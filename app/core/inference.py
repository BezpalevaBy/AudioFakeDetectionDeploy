# app/core/inference.py
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
        """
        Инициализация инференса

        Args:
            model: Загруженная модель spectra_0
            target_length: Целевая длина аудио для модели
        """
        self.model = model
        self.target_length = target_length
        self.device = DEVICE
        self.threshold = THRESHOLD

        # Перемещаем модель на устройство если нужно
        self.model = self.model.to(self.device)
        self.model.eval()

        # Статистика работы
        self.total_inferences = 0
        self.total_processing_time = 0.0

        logger.info(f"AudioInference initialized on {self.device}")
        logger.info(f"Target length: {target_length}, Threshold: {self.threshold}")

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

    def predict_from_file(
        self,
        audio_path: str,
        return_analysis: bool = True,
        spectrogram_id: Optional[uuid.UUID] = None,
    ) -> Dict:
        """
        Выполнение предсказания для аудиофайла

        Args:
            audio_path: Путь к аудиофайлу
            return_analysis: Возвращать детальный анализ
            spectrogram_id: ID для спектрограммы

        Returns:
            Dict: Результаты предсказания
        """
        start_time = time.time()

        try:
            logger.info(f"Начинаем обработку файла: {audio_path}")

            # Валидация аудио
            is_valid, error_msg = AudioProcessor.validate_audio(audio_path)
            if not is_valid:
                return {"error": error_msg, "processing_time": time.time() - start_time}

            # Предобработка аудио
            audio_tensor = self.preprocess_audio(audio_path)

            # Получаем оригинальное аудио для анализа качества
            audio_np, sr = sf.read(audio_path, dtype="float32")
            if len(audio_np.shape) > 1:
                audio_np = audio_np.mean(axis=1)

            # Инференс
            with torch.inference_mode():
                logits = self.model(audio_tensor)  # (1, 2)

                # Индекс 0 = spoof, индекс 1 = bonafide
                score_spoof = logits[0, 0].item()
                score_bonafide = logits[0, 1].item()

            # Классификация на основе порога
            is_bonafide = score_bonafide > self.threshold
            is_fake = not is_bonafide

            # Вычисление уверенности (нормализованная разница логарифмов)
            confidence = abs(score_bonafide - score_spoof) / 10.0
            confidence = min(1.0, max(0.0, confidence))  # ограничиваем [0, 1]

            label = "REAL" if is_bonafide else "FAKE"

            # Вероятности через сигмоиду
            prob_real = 1.0 / (1.0 + np.exp(-score_bonafide))
            prob_fake = 1.0 / (1.0 + np.exp(-score_spoof))
            # Нормализация
            total = prob_real + prob_fake
            prob_real /= total
            prob_fake /= total

            # Анализ качества аудио
            quality_metrics = {}
            if return_analysis:
                quality_metrics = AudioProcessor.analyze_audio_quality(
                    audio_np, sr if "sr" in locals() else SAMPLE_RATE
                )

            # Детальный анализ артефактов
            analysis = None
            if return_analysis:
                analysis = analyze_artifacts(prob_fake, audio_np, sr)

            # Спектрограмма
            spectrogram_data = None
            if spectrogram_id:
                spectrogram_data = generate_spectrogram_data(
                    audio_np, sr, spectrogram_id
                )

            processing_time = time.time() - start_time

            # Обновление статистики
            self.total_inferences += 1
            self.total_processing_time += processing_time

            # Формирование результата
            result = {
                "classification": label,
                "is_fake": is_fake,
                "confidence": confidence,
                "scores": {
                    "bonafide": score_bonafide,
                    "spoof": score_spoof,
                    "threshold_used": self.threshold,
                },
                "probabilities": {"real": float(prob_real), "fake": float(prob_fake)},
                "processing_time": processing_time,
                "audio_quality": quality_metrics,
                "audio_info": {
                    "duration_seconds": float(len(audio_np) / SAMPLE_RATE),
                    "sample_rate": SAMPLE_RATE,
                    "input_length": self.target_length,
                    "original_path": audio_path,
                },
                "model_info": {
                    "name": self.model.__class__.__name__,
                    "device": str(self.device),
                    "target_length": self.target_length,
                    "threshold": self.threshold,
                },
                "timestamp": datetime.now().isoformat(),
                "inference_id": f"inf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.total_inferences}",
            }

            if analysis:
                result["analysis"] = analysis

            if spectrogram_data:
                result["spectrogram"] = spectrogram_data

            logger.info(
                f"Обработка завершена: {label} с уверенностью {confidence:.2%}, "
                f"время: {processing_time:.2f}с"
            )

            return result

        except Exception as e:
            logger.error(f"Ошибка при выполнении предсказания: {str(e)}")
            processing_time = time.time() - start_time

            return {
                "error": str(e),
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
            }

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

    def predict(
        self,
        audio_path: str,
        return_analysis: bool = True,
        spectrogram_id: Optional[uuid.UUID] = None,
    ) -> Dict:
        """
        Алиас для predict_from_file (для обратной совместимости)
        """
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
