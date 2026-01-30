# app/core/batch_processor.py
import asyncio
import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import UploadFile

from app.core.inference import AudioInference
from app.core.audio import AudioProcessor
from app.core.report import generate_report

logger = logging.getLogger(__name__)

# Глобальное хранилище результатов пакетной обработки
_batch_results = {}
_batch_lock = asyncio.Lock()


class BatchProcessor:
    """Класс для пакетной обработки аудиофайлов"""

    def __init__(self, model, max_workers: int = 4):
        """
        Args:
            model: Загруженная модель для инференса
            max_workers: Максимальное количество потоков для обработки
        """
        self.model = model
        self.max_workers = max_workers
        self.inference = AudioInference(model)

        # Статистика
        self.total_batches = 0
        self.total_files_processed = 0

    async def process_batch_async(
        self, task_id: str, file_paths: List[str], temp_dir: str
    ):
        """
        Асинхронная обработка пакета файлов

        Args:
            task_id: Уникальный идентификатор задачи
            file_paths: Список путей к файлам
            temp_dir: Временная директория для хранения файлов
        """
        try:
            # Обновление статуса задачи
            await self._update_task_status(
                task_id,
                {
                    "status": "processing",
                    "started_at": datetime.now().isoformat(),
                    "total_files": len(file_paths),
                    "processed_files": 0,
                },
            )

            # Обработка файлов в пуле потоков
            results = []
            failed_files = []

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Создание задач для каждого файла
                future_to_file = {
                    executor.submit(self._process_single_file, file_path): file_path
                    for file_path in file_paths
                }

                # Обработка завершенных задач
                for i, future in enumerate(as_completed(future_to_file), 1):
                    file_path = future_to_file[future]

                    try:
                        result = future.result()
                        results.append(result)

                        # Обновление прогресса
                        if i % 5 == 0 or i == len(file_paths):
                            await self._update_task_progress(
                                task_id, i, len(file_paths)
                            )

                    except Exception as e:
                        logger.error(f"Failed to process file {file_path}: {str(e)}")
                        failed_files.append({"path": file_path, "error": str(e)})

                        # Обновление прогресса даже при ошибке
                        await self._update_task_progress(task_id, i, len(file_paths))

            # Формирование итогового результата
            final_result = await self._create_final_result(
                task_id, results, failed_files, temp_dir
            )

            # Сохранение результата
            await self._save_batch_result(task_id, final_result)

            # Обновление статистики
            self.total_batches += 1
            self.total_files_processed += len(results)

            logger.info(f"Batch task {task_id} completed successfully")

        except Exception as e:
            logger.error(f"Batch processing failed for task {task_id}: {str(e)}")
            await self._update_task_status(
                task_id,
                {
                    "status": "failed",
                    "error": str(e),
                    "completed_at": datetime.now().isoformat(),
                },
            )

        finally:
            # Очистка временных файлов
            await self._cleanup_temp_files(temp_dir)

    def _process_single_file(self, file_path: str) -> Dict:
        try:
            # Валидация файла
            is_valid, error_msg = AudioProcessor.validate_audio(file_path)
            if not is_valid:
                raise ValueError(f"Invalid audio file: {error_msg}")

            # Выполнение анализа
            result = self.inference.predict(
                file_path, return_analysis=False, spectrogram_id=0
            )

            if not result or not isinstance(result, dict):
                raise ValueError(f"Inference result invalid for file: {file_path}")

            if "error" in result:
                raise ValueError(result["error"])

            # Генерация отчета
            report = generate_report(
                classification=result["classification"],
                confidence=result["confidence"],
                artifacts=result.get("analysis", {}),
                processing_time=result["processing_time"],
                audio_quality=result.get("audio_quality", {}),
                spectrogram_data=result.get("spectrogram"),
                model_info=result.get("model_info", {}),
            )

            return {
                "filename": file_path,
                "success": True,
                "result": report.get(
                    "api_format", {"is_fake": None, "confidence": 0.0}
                ),
                "processing_time": result.get("processing_time", 0.0),
            }

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return {
                "filename": os.path.basename(file_path),
                "success": False,
                "result": {"is_fake": None, "confidence": 0.0},
                "processing_time": 0.0,
                "error": str(e),
                "file_size": (
                    os.path.getsize(file_path) if os.path.exists(file_path) else 0
                ),
            }

    async def _create_final_result(
        self, task_id: str, results: List[Dict], failed_files: List[Dict], temp_dir: str
    ) -> Dict:
        """Создание финального результата пакетной обработки с безопасной агрегацией"""

        successful_results = [r for r in results if r.get("success")]
        total = len(results) + len(failed_files)
        successful = len(successful_results)

        # Считаем fake/real безопасно
        fake_count = sum(
            1 for r in successful_results if r["result"].get("is_fake") == True
        )
        real_count = sum(
            1 for r in successful_results if r["result"].get("is_fake") == False
        )

        total_processing_time = sum(
            r.get("processing_time", 0.0) for r in successful_results
        )

        # Определяем общий вердикт
        overall_verdict = "MIXED"
        if fake_count == 0 and real_count > 0:
            overall_verdict = "ALL_REAL"
        elif real_count == 0 and fake_count > 0:
            overall_verdict = "ALL_FAKE"
        elif fake_count > real_count:
            overall_verdict = "MOSTLY_FAKE"
        elif real_count > fake_count:
            overall_verdict = "MOSTLY_REAL"

        avg_confidence = (
            sum(r["result"].get("confidence", 0.0) for r in successful_results)
            / successful
            if successful > 0
            else 0.0
        )

        return {
            "task_id": task_id,
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat(),
            "statistics": {
                "total_files": total,
                "successful": successful,
                "failed": len(failed_files),
                "success_rate": (successful / total * 100) if total > 0 else 0,
                "total_processing_time": total_processing_time,
                "avg_time_per_file": (
                    total_processing_time / successful if successful > 0 else 0
                ),
            },
            "summary": {
                "real_count": real_count,
                "fake_count": fake_count,
                "fake_percentage": (
                    (fake_count / successful * 100) if successful > 0 else 0
                ),
                "overall_verdict": overall_verdict,
                "avg_confidence": avg_confidence,
            },
            "results": results,
            "failed_files": failed_files,
            "metadata": {
                "temp_dir": temp_dir,
                "model": self.model.__class__.__name__,
                "max_workers": self.max_workers,
            },
        }

    async def _update_task_status(self, task_id: str, status: Dict):
        """Обновление статуса задачи"""
        async with _batch_lock:
            if task_id not in _batch_results:
                _batch_results[task_id] = {}
            _batch_results[task_id].update(status)

    async def _update_task_progress(self, task_id: str, processed: int, total: int):
        """Обновление прогресса обработки"""
        progress = {
            "processed_files": processed,
            "progress_percentage": (processed / total * 100) if total > 0 else 0,
            "last_update": datetime.now().isoformat(),
        }
        await self._update_task_status(task_id, progress)

    async def _save_batch_result(self, task_id: str, result: Dict):
        """Сохранение результата пакетной обработки"""
        async with _batch_lock:
            _batch_results[task_id] = result

        # Также сохраняем в файл для надежности
        try:
            result_dir = os.path.join(tempfile.gettempdir(), "audio_detection_batches")
            os.makedirs(result_dir, exist_ok=True)

            result_file = os.path.join(result_dir, f"{task_id}.json")
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)

            logger.debug(f"Batch result saved to {result_file}")
        except Exception as e:
            logger.error(f"Failed to save batch result to file: {str(e)}")

    async def _cleanup_temp_files(self, temp_dir: str):
        """Очистка временных файлов"""
        try:
            if os.path.exists(temp_dir):
                import shutil

                shutil.rmtree(temp_dir)
                logger.debug(f"Temporary directory removed: {temp_dir}")
        except Exception as e:
            logger.error(f"Failed to cleanup temp directory {temp_dir}: {str(e)}")

    def get_stats(self) -> Dict:
        """Получение статистики пакетного процессора"""
        return {
            "total_batches": self.total_batches,
            "total_files_processed": self.total_files_processed,
            "max_workers": self.max_workers,
            "model": self.model.__class__.__name__,
        }


# Глобальные функции для работы с пакетной обработкой


async def process_batch_async(task_id: str, file_paths: List[str], temp_dir: str):
    """Асинхронная обработка пакета файлов"""
    from app.main import get_model, get_metrics

    model = get_model()
    metrics = get_metrics()

    processor = BatchProcessor(model)

    try:
        metrics.increment_batch_requests()
        await processor.process_batch_async(task_id, file_paths, temp_dir)
    finally:
        metrics.decrement_batch_tasks(len(file_paths))


def get_batch_results(task_id: str) -> Optional[Dict]:
    """Получение результатов пакетной обработки"""
    return _batch_results.get(task_id)


async def process_batch(files: List[UploadFile], task_id: str, metrics):
    """Синхронная обработка пакета (для обратной совместимости)"""
    # Этот метод оставлен для обратной совместимости
    # В реальном использовании следует использовать process_batch_async

    temp_dir = os.path.join(tempfile.gettempdir(), task_id)
    os.makedirs(temp_dir, exist_ok=True)

    file_paths = []
    for i, file in enumerate(files):
        temp_path = os.path.join(temp_dir, f"{i}_{file.filename}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_paths.append(temp_path)

    await process_batch_async(task_id, file_paths, temp_dir)


def get_batch_status(task_id: str) -> Optional[Dict]:
    """Получение статуса пакетной задачи"""
    result = get_batch_results(task_id)
    if result:
        return {
            "task_id": task_id,
            "status": result.get("status", "unknown"),
            "progress": result.get("progress_percentage", 0),
            "processed_files": result.get("processed_files", 0),
            "total_files": result.get("total_files", 0),
            "created_at": result.get("created_at"),
            "last_update": result.get("last_update"),
        }
    return None
