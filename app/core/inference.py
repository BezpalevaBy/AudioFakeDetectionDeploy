# app/core/inference.py
import torch
import numpy as np
from typing import Tuple, Dict, Optional, List, Union
import logging
import time
from datetime import datetime
import uuid
import os
import tempfile

from .audio import AudioProcessor
from .model import DEVICE
from .analysis import analyze_artifacts, generate_spectrogram_data
import torchaudio


class AudioInference:
    """Класс для выполнения предсказаний на аудио"""

    def __init__(self, model, target_length: int = 64600):
        """
        Args:
            model: Загруженная модель
            target_length: Целевая длина аудио для модели
        """
        # Настройка логирования
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        self.model = model
        self.target_length = target_length
        self.model.eval()  # Переводим модель в режим инференса

        # Статистика работы
        self.total_inferences = 0
        self.total_processing_time = 0.0

        self.logger.info(f"Инициализирован AudioInference на устройстве {DEVICE}")
        self.logger.info(f"Модель: {model.__class__.__name__}")

    def _setup_logging(self):
        """Настройка логирования"""
        # Получаем корневой логгер
        root_logger = logging.getLogger()

        # Настраиваем только если нет хендлеров
        if not root_logger.handlers:
            # Создаем форматтер
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            # Создаем консольный хендлер
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(logging.INFO)

            # Создаем файловый хендлер (опционально)
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(
                os.path.join(
                    log_dir, f'inference_{datetime.now().strftime("%Y%m%d")}.log'
                )
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)

            # Настраиваем корневой логгер
            root_logger.setLevel(logging.DEBUG)
            root_logger.addHandler(console_handler)
            root_logger.addHandler(file_handler)

    def predict(
        self,
        audio_path: str,
        return_analysis: bool = True,
        spectrogram_id: uuid.UUID = None,
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
            self.logger.info(f"Начинаем обработку файла: {audio_path}")
            self.logger.info(f"Длина целевого аудио: {self.target_length} сэмплов")

            # Валидация аудио
            is_valid, error_msg = AudioProcessor.validate_audio(audio_path)
            if not is_valid:
                self.logger.error(f"Невалидный аудиофайл: {error_msg}")
                return {"error": error_msg, "processing_time": time.time() - start_time}

            # Параметры обработки
            MAX_CHUNK_SEC = 4
            SAMPLE_RATE = 16000
            CHUNK_SAMPLES = MAX_CHUNK_SEC * SAMPLE_RATE

            # Загрузка аудио
            self.logger.debug(f"Загрузка аудио из {audio_path}")
            audio, sr = torchaudio.load(audio_path)
            self.logger.info(f"Загружено аудио: форма={audio.shape}, частота={sr} Гц")

            # Предобработка аудио
            if sr != SAMPLE_RATE:
                self.logger.info(f"Ресемплинг с {sr} Гц на {SAMPLE_RATE} Гц")
                resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                audio = resampler(audio)
                sr = SAMPLE_RATE

            if audio.shape[0] > 1:
                self.logger.debug(f"Преобразование многоканального аудио в моно")
                audio = torch.mean(audio, dim=0, keepdim=True)

            # Обрезка/дополнение до нужной длины
            current_length = audio.shape[1]
            if current_length < self.target_length:
                self.logger.debug(
                    f"Дополнение аудио с {current_length} до {self.target_length} сэмплов"
                )
                padding = self.target_length - current_length
                audio = torch.nn.functional.pad(audio, (0, padding))
            elif current_length > self.target_length:
                self.logger.debug(
                    f"Обрезка аудио с {current_length} до {self.target_length} сэмплов"
                )
                audio = audio[:, : self.target_length]

            self.logger.debug(f"Финальная форма аудио: {audio.shape}")

            # Разбиение на чанки для анализа качества
            chunks = []
            total_len = audio.shape[1]

            for start in range(0, total_len, CHUNK_SAMPLES):
                end = min(start + CHUNK_SAMPLES, total_len)
                chunk = audio[:, start:end]
                if chunk.shape[1] > 0:
                    chunks.append(chunk)

            self.logger.debug(f"Разбито на {len(chunks)} чанков для анализа качества")

            # Анализ качества для каждого чанка
            real_probs = []
            fake_probs = []
            quality_metrics_list = []

            # Инференс модели
            self.logger.info("Запуск инференса модели...")
            with torch.no_grad():
                # Перемещаем аудио на нужное устройство
                audio = audio.to(DEVICE)
                model_out = self.model(audio)

                if isinstance(model_out, torch.Tensor):
                    logits = model_out
                elif hasattr(model_out, "logits"):
                    logits = model_out.logits
                else:
                    raise TypeError(f"Unsupported model output type: {type(model_out)}")

                probs = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probs, dim=1)
                probabilities = torch.softmax(logits, dim=1)

                self.logger.info(
                    f"Результат модели: {'Bonafide' if prediction.item() == 0 else 'Spoof'}"
                )
                self.logger.info(f"Уверенность: {probabilities.max().item():.3f}")

                real_probs.append(probs[0][0].item())
                fake_probs.append(probs[0][1].item())

            # Расчет итоговых вероятностей
            real_prob = float(np.mean(real_probs))
            fake_prob = float(np.mean(fake_probs))

            is_fake = fake_prob > 0.5
            confidence = fake_prob if is_fake else real_prob
            label = "FAKE" if is_fake else "REAL"

            self.logger.info(
                f"Итоговое предсказание: {label} (REAL: {real_prob:.3f}, FAKE: {fake_prob:.3f})"
            )

            # Агрегация метрик качества
            quality_metrics = {}
            if quality_metrics_list:
                for key in quality_metrics_list[0].keys():
                    quality_metrics[key] = float(
                        np.mean([qm[key] for qm in quality_metrics_list])
                    )
                self.logger.debug(f"Метрики качества: {quality_metrics}")

            # Анализ артефактов (если требуется)
            analysis = None
            if return_analysis:
                self.logger.debug("Запуск анализа артефактов...")
                analysis = analyze_artifacts(fake_prob, audio, sr)
                if analysis:
                    self.logger.debug(
                        f"Анализ артефактов завершен: {analysis.get('verdict', 'N/A')}"
                    )

            # Генерация данных спектрограммы (если требуется)
            spectrogram_data = None
            if spectrogram_id:
                self.logger.debug(f"Генерация спектрограммы с ID: {spectrogram_id}")
                spectrogram_data = generate_spectrogram_data(
                    audio.cpu(), sr, spectrogram_id
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
                "probabilities": {"real": float(real_prob), "fake": float(fake_prob)},
                "processing_time": processing_time,
                "audio_quality": quality_metrics,
                "audio_info": {
                    "duration_seconds": audio.shape[1] / sr,
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

            if spectrogram_data:
                result["spectrogram"] = spectrogram_data

            self.logger.info(
                f"Обработка завершена: {label} с уверенностью {confidence:.2%}, "
                f"время: {processing_time:.2f}с"
            )

            return result

        except FileNotFoundError as e:
            self.logger.error(f"Файл не найден: {str(e)}")
            return {
                "error": f"Файл не найден: {audio_path}",
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(
                f"Ошибка при выполнении предсказания: {str(e)}", exc_info=True
            )
            processing_time = time.time() - start_time

            return {
                "error": str(e),
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
            }

    def predict_batch(self, audio_paths: List[str], batch_size: int = 8) -> Dict:
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

        self.logger.info(f"Начало пакетной обработки {len(audio_paths)} файлов")

        # Обработка файлов батчами
        total_batches = (len(audio_paths) + batch_size - 1) // batch_size
        for i in range(0, len(audio_paths), batch_size):
            batch = audio_paths[i : i + batch_size]
            batch_start = time.time()
            batch_num = i // batch_size + 1

            self.logger.info(
                f"Обработка батча {batch_num}/{total_batches} " f"({len(batch)} файлов)"
            )

            for j, audio_path in enumerate(batch):
                file_num = i + j + 1
                try:
                    self.logger.debug(
                        f"Обработка файла {file_num}/{len(audio_paths)}: {audio_path}"
                    )
                    result = self.predict(audio_path, return_analysis=True)
                    results.append(result)
                except Exception as e:
                    self.logger.error(
                        f"Ошибка при обработке файла {audio_path}: {str(e)}"
                    )
                    failed_files.append(
                        {"path": audio_path, "error": str(e), "file_number": file_num}
                    )

            batch_time = time.time() - batch_start
            self.logger.info(f"Батч {batch_num} обработан за {batch_time:.2f}с")

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

        self.logger.info(
            f"Пакетная обработка завершена: "
            f"успешно {len(results)}, ошибок {len(failed_files)}, "
            f"общее время {total_time:.2f}с"
        )
        self.logger.info(
            f"Итоги: REAL={real_count}, FAKE={fake_count} "
            f"({fake_count/len(results)*100:.1f}% подделок)"
        )

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
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        return_analysis: bool = True,
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
        self.logger.info(
            f"Обработка аудио из байтов ({len(audio_bytes)} байт), имя: {filename}"
        )

        # Сохранение во временный файл
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".wav", prefix="temp_audio_"
        ) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            result = self.predict(tmp_path, return_analysis=return_analysis)
            # Добавляем информацию о имени файла
            if "audio_info" in result:
                result["audio_info"]["original_filename"] = filename
            return result
        except Exception as e:
            self.logger.error(
                f"Ошибка при обработке байтов аудио: {str(e)}", exc_info=True
            )
            raise
        finally:
            # Удаление временного файла
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                    self.logger.debug(f"Временный файл удален: {tmp_path}")
                except Exception as e:
                    self.logger.warning(
                        f"Не удалось удалить временный файл {tmp_path}: {str(e)}"
                    )

    def get_stats(self) -> Dict:
        """Получение статистики работы инференса"""
        stats = {
            "total_inferences": self.total_inferences,
            "total_processing_time": self.total_processing_time,
            "avg_processing_time": (
                self.total_processing_time / self.total_inferences
                if self.total_inferences > 0
                else 0
            ),
            "model_device": str(DEVICE),
            "model_name": self.model.__class__.__name__,
            "target_length": self.target_length,
        }

        self.logger.debug(f"Запрошена статистика: {stats}")
        return stats


def predict(model, audio_path: str) -> Tuple[str, float]:
    """
    Упрощенная функция для предсказания

    Args:
        model: Загруженная модель
        audio_path: Путь к аудиофайлу

    Returns:
        Tuple[str, float]: (метка, вероятность)
    """
    # Настраиваем логгер для этой функции
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Запуск упрощенного предсказания для: {audio_path}")
        inference = AudioInference(model)
        result = inference.predict(
            audio_path, return_analysis=False, spectrogram_id=None
        )

        if "error" in result:
            logger.error(f"Ошибка в упрощенном предсказании: {result['error']}")
            raise ValueError(result["error"])

        logger.info(
            f"Упрощенное предсказание завершено: {result['classification']}, {result['confidence']:.3f}"
        )
        return result["classification"], result["confidence"]
    except Exception as e:
        logger.error(f"Ошибка в функции predict: {str(e)}", exc_info=True)
        raise
