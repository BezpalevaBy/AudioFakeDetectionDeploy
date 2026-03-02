# app/core/inference.py
import math

import torch
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
import time
from datetime import datetime
import random
import torchaudio
import soundfile as sf
import uuid
import os
import tempfile
import io

from .audio import AudioProcessor
from .analysis import analyze_artifacts, generate_spectrogram_data

logger = logging.getLogger(__name__)

# Константы для модели spectra_0
SAMPLE_RATE = 16000
TARGET_LENGTH = 64600  # фиксированная длина для модели
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = -1.0625009  # оптимальный порог из модели


def pad_random(x: torch.Tensor, max_len: int = TARGET_LENGTH) -> torch.Tensor:
    """
    Обрезка или повторение аудио до целевой длины

    Args:
        x: Входной тензор аудио
        max_len: Целевая длина

    Returns:
        torch.Tensor: Аудио фиксированной длины
    """
    if x.ndim > 1:
        x = x.squeeze()
    x_len = x.shape[0]

    if x_len >= max_len:
        # Если аудио длиннее, берем случайный отрезок
        start = random.randint(0, x_len - max_len)
        return x[start : start + max_len]
    else:
        # Если короче, повторяем
        num_repeats = int(max_len / x_len) + 1
        return x.repeat(num_repeats)[:max_len]


def load_audio_mono(path: str) -> torch.Tensor:
    """
    Загрузка аудио в моно с частотой 16kHz

    Args:
        path: Путь к аудиофайлу

    Returns:
        torch.Tensor: Аудио тензор
    """
    audio, sr = sf.read(path, dtype="float32")
    audio = torch.from_numpy(audio)

    if audio.ndim > 1:
        # (num_samples, channels) -> mono
        audio = audio.mean(dim=1)

    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)

    return audio


class AudioInference:
    """
    Класс для выполнения предсказаний на аудио с использованием модели spectra_0
    """

    def __init__(self, model, target_length: int = TARGET_LENGTH):
        self.model = model
        self.target_length = target_length
        self.device = DEVICE
        self.threshold = THRESHOLD  # порог для logits bonafide
        self.model = self.model.to(self.device)
        self.model.eval()
        self.total_inferences = 0
        self.total_processing_time = 0.0

    def _prepare_segment_tensor(self, segment):
        segment = torchaudio.functional.preemphasis(segment.unsqueeze(0))
        segment = pad_random(segment.squeeze(0), self.target_length).unsqueeze(0)
        return segment.to(self.device)

    def _analyze_tensor_5x(self, tensor, rangeNumber, max_shift_ratio=0.05):
        """
        range-кратный анализ сегмента со случайным временным смещением.

        max_shift_ratio = 0.15 означает, что сдвиг может быть до ±15% длины сегмента
        """

        spoof_sum = 0.0
        bonafide_sum = 0.0

        segment_length = tensor.shape[-1]
        max_shift = int(segment_length * max_shift_ratio)

        logger.debug("----- START RANDOM SHIFT ANALYSIS -----")

        with torch.inference_mode():
            for i in range(int(rangeNumber)):

                # 🎯 случайный сдвиг
                shift = random.randint(-max_shift, max_shift)

                if shift > 0:
                    shifted = torch.nn.functional.pad(tensor[..., shift:], (0, shift))
                elif shift < 0:
                    shifted = torch.nn.functional.pad(tensor[..., :shift], (-shift, 0))
                else:
                    shifted = tensor

                logits = self.model(shifted)

                spoof = logits[0, 0].item()
                bonafide = logits[0, 1].item()
                delta = bonafide - spoof

                spoof_sum += spoof
                bonafide_sum += bonafide

                logger.debug(
                    f"[Pass {i+1}/{rangeNumber}] "
                    f"Shift: {shift} | "
                    f"Spoof: {spoof:.6f} | "
                    f"Bonafide: {bonafide:.6f} | "
                    f"Delta: {delta:.6f}"
                )

        avg_spoof = spoof_sum / float(rangeNumber)
        avg_bonafide = bonafide_sum / float(rangeNumber)

        logger.debug("----- AVERAGED RESULT -----")
        logger.debug(
            f"Avg Spoof: {avg_spoof:.6f} | "
            f"Avg Bonafide: {avg_bonafide:.6f} | "
            f"Final Delta: {avg_bonafide - avg_spoof:.6f}"
        )
        logger.debug("----- END ANALYSIS -----\n")

        return avg_spoof, avg_bonafide

    def _split_into_segments(self, audio, sr):
        segment_len = int(7 * sr)
        segments = []
        for start in range(0, len(audio), segment_len):
            end = start + segment_len
            if end > len(audio):
                break
            segments.append((start, end, audio[start:end]))
        return segments

    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        Полная предобработка аудио для модели

        Args:
            audio_path: Путь к аудиофайлу

        Returns:
            torch.Tensor: Подготовленный тензор для модели
        """
        # Загрузка аудио
        audio = load_audio_mono(audio_path)

        # Preemphasis фильтр (как в примере модели)
        audio = torchaudio.functional.preemphasis(audio.unsqueeze(0))  # (1, T)

        # Обрезка/повтор до нужной длины
        audio = pad_random(audio.squeeze(0), self.target_length).unsqueeze(
            0
        )  # (1, target_length)

        return audio.to(self.device)

    def preprocess_bytes(self, audio_bytes: bytes) -> torch.Tensor:
        """
        Предобработка аудио из байтов

        Args:
            audio_bytes: Байты аудиофайла

        Returns:
            torch.Tensor: Подготовленный тензор для модели
        """
        # Чтение из байтов
        with io.BytesIO(audio_bytes) as f:
            audio, sr = sf.read(f, dtype="float32")

        audio = torch.from_numpy(audio)

        if audio.ndim > 1:
            audio = audio.mean(dim=1)

        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)

        # Preemphasis
        audio = torchaudio.functional.preemphasis(audio.unsqueeze(0))

        # Обрезка/повтор
        audio = pad_random(audio.squeeze(0), self.target_length).unsqueeze(0)

        return audio.to(self.device)

    # ============================================================
    # ===================== MAIN PREDICT =========================
    # ============================================================

    def predict_from_file(self, audio_path, return_analysis=True, spectrogram_id=None):
        start = time.time()

        try:
            is_valid, error_msg = AudioProcessor.validate_audio(audio_path)
            if not is_valid:
                return {"error": error_msg}

            audio_np, sr = sf.read(audio_path, dtype="float32")
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)

            audio_tensor_full = torch.from_numpy(audio_np)
            duration = len(audio_tensor_full) / sr

            if duration > 15:
                segments = self._split_into_segments(audio_tensor_full, sr)
            else:
                segments = [(0, len(audio_tensor_full), audio_tensor_full)]

            total_spoof = 0.0
            total_bonafide = 0.0
            segments_info = []

            logger.setLevel(logging.DEBUG)

            for idx, (start_idx, end_idx, seg) in enumerate(segments):
                seg_tensor = self._prepare_segment_tensor(seg)
                spoof_score, bonafide_score = self._analyze_tensor_5x(seg_tensor, 25)

                total_spoof += spoof_score
                total_bonafide += bonafide_score

                segments_info.append(
                    {
                        "segment_index": idx,
                        "start_sec": start_idx / sr,
                        "end_sec": end_idx / sr,
                        "score_spoof": spoof_score,
                        "score_bonafide": bonafide_score,
                    }
                )

            avg_spoof = total_spoof / len(segments)
            avg_bonafide = total_bonafide / len(segments)

            is_bonafide = avg_bonafide > self.threshold
            label = "REAL" if is_bonafide else "FAKE"

            margin = avg_bonafide - self.threshold

            prob_real = 1 / (1 + math.exp(-margin))
            prob_fake = 1 - prob_real

            confidence = min((abs(margin) / (abs(avg_bonafide) + 1e-6)), 1)

            processing_time = time.time() - start
            self.total_inferences += 1
            self.total_processing_time += processing_time

            result = {
                "classification": label,
                "is_fake": not is_bonafide,
                "confidence": float(confidence),
                "scores": {
                    "bonafide": avg_bonafide,
                    "spoof": avg_spoof,
                    "threshold_used": self.threshold,
                },
                "probabilities": {"real": float(prob_real), "fake": float(prob_fake)},
                "processing_time": processing_time,
                "audio_info": {
                    "duration_seconds": float(duration),
                    "sample_rate": sr,
                    "segments_used": len(segments),
                },
                "segments": segments_info,
                "timestamp": datetime.now().isoformat(),
                "inference_id": f"inf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.total_inferences}",
            }

            return result

        except Exception as e:
            return {"error": str(e)}

    def predict_from_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        return_analysis: bool = False,
    ) -> Dict:
        """
        Предсказание из байтов аудио

        Args:
            audio_bytes: Байты аудиофайла
            filename: Имя файла (для логирования)
            return_analysis: Возвращать детальный анализ

        Returns:
            Dict: Результаты предсказания
        """
        start_time = time.time()

        try:
            # Предобработка байтов
            audio_tensor = self.preprocess_bytes(audio_bytes)

            # Инференс
            with torch.inference_mode():
                logits = self.model(audio_tensor)
                score_spoof = logits[0, 0].item()
                score_bonafide = logits[0, 1].item()

            # Классификация
            is_bonafide = score_bonafide > self.threshold
            is_fake = not is_bonafide
            confidence = abs(score_bonafide - score_spoof) / 10.0
            confidence = min(1.0, max(0.0, confidence))
            label = "REAL" if is_bonafide else "FAKE"

            # Вероятности
            prob_real = 1.0 / (1.0 + np.exp(-score_bonafide))
            prob_fake = 1.0 / (1.0 + np.exp(-score_spoof))
            total = prob_real + prob_fake
            prob_real /= total
            prob_fake /= total

            processing_time = time.time() - start_time

            self.total_inferences += 1
            self.total_processing_time += processing_time

            result = {
                "classification": label,
                "is_fake": is_fake,
                "confidence": confidence,
                "scores": {"bonafide": score_bonafide, "spoof": score_spoof},
                "probabilities": {"real": float(prob_real), "fake": float(prob_fake)},
                "processing_time": processing_time,
                "filename": filename,
                "timestamp": datetime.now().isoformat(),
            }

            return result

        except Exception as e:
            logger.error(f"Ошибка при обработке байтов: {str(e)}")
            return {
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    # ============================================================
    # ============ NEW METHOD WITH FAKE SEGMENT INFO ============
    # ============================================================

    def predict_with_segments_info(
        self, audio_path, return_analysis=True, spectrogram_id=None
    ):
        res = self.predict_from_file(audio_path, return_analysis, spectrogram_id)
        if "segments" not in res:
            return res

        fake_segments = []
        for seg in res["segments"]:
            if seg["score_spoof"] > seg["score_bonafide"]:
                fake_segments.append(
                    {
                        "segment_index": seg["segment_index"],
                        "start_sec": seg["start_sec"],
                        "end_sec": seg["end_sec"],
                        "confidence": abs(seg["score_bonafide"] - seg["score_spoof"])
                        / max(
                            1.0, abs(seg["score_bonafide"]) + abs(seg["score_spoof"])
                        ),
                    }
                )
        res["fake_segments"] = fake_segments
        return res

    def predict(
        self, audio_path: str, return_analysis: bool = True, spectrogram_id=None
    ):
        return self.predict_from_file(audio_path, return_analysis, spectrogram_id)

    def predict_batch(self, audio_paths: List[str], batch_size: int = 8) -> Dict:
        """
        Пакетная обработка аудиофайлов

        Args:
            audio_paths: Список путей к аудиофайлам
            batch_size: Размер батча (не используется, т.к. модель не поддерживает батчи)

        Returns:
            Dict: Результаты пакетной обработки
        """
        start_time = time.time()
        results = []
        failed_files = []

        logger.info(f"Начало пакетной обработки {len(audio_paths)} файлов")

        for i, audio_path in enumerate(audio_paths):
            try:
                logger.info(f"Обработка файла {i+1}/{len(audio_paths)}: {audio_path}")
                result = self.predict_from_file(audio_path, return_analysis=False)
                results.append(result)
            except Exception as e:
                logger.error(f"Ошибка при обработке файла {audio_path}: {str(e)}")
                failed_files.append({"path": audio_path, "error": str(e)})

        total_time = time.time() - start_time

        # Статистика
        stats = {
            "total_files": len(audio_paths),
            "successful": len(results),
            "failed": len(failed_files),
            "total_processing_time": total_time,
            "avg_time_per_file": total_time / len(audio_paths) if audio_paths else 0,
        }

        # Агрегированные результаты
        fake_count = sum(1 for r in results if r.get("is_fake", False))
        real_count = len(results) - fake_count

        return {
            "batch_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "statistics": stats,
            "summary": {
                "real_count": real_count,
                "fake_count": fake_count,
                "fake_percentage": (fake_count / len(results)) * 100 if results else 0,
            },
            "results": results,
            "failed_files": failed_files,
            "timestamp": datetime.now().isoformat(),
        }

    def get_stats(self) -> Dict:
        """
        Получение статистики работы инференса

        Returns:
            Dict: Статистика
        """
        return {
            "total_inferences": self.total_inferences,
            "total_processing_time": self.total_processing_time,
            "avg_processing_time": (
                self.total_processing_time / self.total_inferences
                if self.total_inferences > 0
                else 0
            ),
            "model_device": str(self.device),
            "threshold": self.threshold,
        }

    def reset_stats(self):
        """Сброс статистики"""
        self.total_inferences = 0
        self.total_processing_time = 0.0


def predict(model, audio_path: str) -> Tuple[str, float]:
    """
    Упрощенная функция для предсказания

    Args:
        model: Загруженная модель
        audio_path: Путь к аудиофайлу

    Returns:
        Tuple[str, float]: (метка, уверенность)
    """
    inference = AudioInference(model)
    result = inference.predict_from_file(audio_path, return_analysis=False)

    if "error" in result:
        raise ValueError(result["error"])

    return result["classification"], result["confidence"]
