# app/core/inference.py
import torch
import numpy as np
from typing import Tuple, Dict, Optional
import logging
import time
from datetime import datetime

from .audio import AudioProcessor
from .model import DEVICE
from .analysis import analyze_artifacts, generate_spectrogram_data

logger = logging.getLogger(__name__)


class AudioInference:
    """Класс для выполнения предсказаний на аудио"""

    def __init__(self, model, target_length: int = 64600):
        """
        Args:
            model: Загруженная модель
            target_length: Целевая длина аудио для модели
        """
        self.model = model
        self.target_length = target_length
        self.model.eval()

        # Статистика работы
        self.total_inferences = 0
        self.total_processing_time = 0.0

    import uuid

    def predict(
        self, audio_path: str, return_analysis: bool = True, spectrogram_id: uuid = 0
    ) -> Dict:
        """
        Выполнение предсказания для аудиофайла

        Args:
            audio_path: Путь к аудиофайлу
            return_analysis: Возвращать детальный анализ
            generate_spectrogram: Генерировать данные спектрограммы

        Returns:
            Dict: Результаты предсказания
        """
        start_time = time.time()

        try:
            # Загрузка и предобработка аудио
            logger.info(f"Начинаем обработку файла: {audio_path}")

            # Валидация аудио
            is_valid, error_msg = AudioProcessor.validate_audio(audio_path)
            if not is_valid:
                return {"error": error_msg, "processing_time": time.time() - start_time}

            # Загрузка аудио
            audio, sr = AudioProcessor.load_audio(audio_path, normalize=True)

            # Анализ качества аудио
            quality_metrics = AudioProcessor.analyze_audio_quality(audio, sr)

            # Предобработка для модели
            processed_audio = AudioProcessor.preprocess_for_model(
                audio, sr, target_length=self.target_length
            )

            # Конвертация в тензор
            tensor = torch.tensor(processed_audio, dtype=torch.float32)
            tensor = tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)

            # Выполнение предсказания
            with torch.no_grad():
                logits = self.model(tensor)  # [B,1] → сигмоида внутри модели
                fake_prob = logits[0, 0].item()  # вероятность фейка
                real_prob = 1.0 - fake_prob
                is_fake = fake_prob > 0.5
                confidence = fake_prob if is_fake else real_prob

            # Определение метки
            label = "FAKE" if is_fake else "REAL"

            # Детальный анализ артефактов
            analysis = None
            if return_analysis:
                analysis = analyze_artifacts(fake_prob, audio, sr)

            # Генерация данных спектрограммы
            spectrogram_data = None
            if spectrogram_id != 0:
                spectrogram_data = generate_spectrogram_data(audio, sr, spectrogram_id)

            processing_time = time.time() - start_time

            # Обновление статистики
            self.total_inferences += 1
            self.total_processing_time += processing_time

            # Формирование результата
            result = {
                "classification": label,
                "is_fake": is_fake,
                "confidence": confidence,
                "probabilities": {"real": float(real_prob), "fake": float(fake_prob)},
                "processing_time": processing_time,
                "audio_quality": quality_metrics,
                "audio_info": {
                    "duration_seconds": len(audio) / sr,
                    "sample_rate": sr,
                    "original_path": audio_path,
                },
                "model_info": {
                    "name": self.model.__class__.__name__,
                    "device": str(DEVICE),
                    "target_length": self.target_length,
                },
                "timestamp": datetime.now().isoformat(),
                "inference_id": f"inf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.total_inferences}",
            }

            if analysis:
                result["analysis"] = analysis

            if spectrogram_data != 0:
                result["spectrogram"] = spectrogram_data

            logger.info(
                f"Обработка завершена: {label} с уверенностью {confidence:.2%}, время: {processing_time:.2f}с"
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

    def predict_batch(self, audio_paths: list, batch_size: int = 8) -> Dict:
        """
        Пакетная обработка аудиофайлов

        Args:
            audio_paths: Список путей к аудиофайлам
            batch_size: Размер батча

        Returns:
            Dict: Результаты пакетной обработки
        """
        start_time = time.time()
        results = []
        failed_files = []

        logger.info(f"Начало пакетной обработки {len(audio_paths)} файлов")

        # Обработка файлов батчами
        for i in range(0, len(audio_paths), batch_size):
            batch = audio_paths[i : i + batch_size]
            batch_start = time.time()

            logger.info(
                f"Обработка батча {i//batch_size + 1}/{(len(audio_paths)-1)//batch_size + 1}"
            )

            for audio_path in batch:
                try:
                    result = self.predict(audio_path, return_analysis=True)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Ошибка при обработке файла {audio_path}: {str(e)}")
                    failed_files.append({"path": audio_path, "error": str(e)})

            batch_time = time.time() - batch_start
            logger.info(f"Батч обработан за {batch_time:.2f}с")

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

    def predict_from_bytes(
        self, audio_bytes: bytes, filename: str = "audio.wav"
    ) -> Dict:
        """
        Предсказание из байтов аудио

        Args:
            audio_bytes: Байты аудиофайла
            filename: Имя файла (для логирования)

        Returns:
            Dict: Результаты предсказания
        """
        import tempfile
        import os

        # Сохранение во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            result = self.predict(tmp_path)
            return result
        finally:
            # Удаление временного файла
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def get_stats(self) -> Dict:
        """Получение статистики работы инференса"""
        return {
            "total_inferences": self.total_inferences,
            "total_processing_time": self.total_processing_time,
            "avg_processing_time": (
                self.total_processing_time / self.total_inferences
                if self.total_inferences > 0
                else 0
            ),
            "model_device": str(DEVICE),
        }


def predict(model, audio_path: str) -> Tuple[str, float]:
    """
    Упрощенная функция для предсказания

    Args:
        model: Загруженная модель
        audio_path: Путь к аудиофайлу

    Returns:
        Tuple[str, float]: (метка, вероятность)
    """
    inference = AudioInference(model)
    result = inference.predict(audio_path, return_analysis=False, spectrogram_id=0)

    if "error" in result:
        raise ValueError(result["error"])

    return result["classification"], result["confidence"]
