# app/api/routes.py
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
    FileResponse,
)
from typing import List, Optional, Dict
from fastapi import Path
import shutil
import os
import tempfile
import uuid
import logging
from datetime import datetime
import json
import io

import numpy as np

from app.core.inference import AudioInference
from app.core.audio import AudioProcessor
from app.core.report import generate_report, export_report
from app.core.model import get_model_info
from app.api.schemas import (
    AnalysisResponse,
    BatchAnalysisResponse,
    StreamingAnalysisResponse,
    SystemStatusResponse,
    ErrorResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Глобальный объект инференса
_inference = None


def get_inference():
    """Ленивая инициализация инференса"""
    global _inference
    if _inference is None:
        from app.main import get_model

        model = get_model()
        _inference = AudioInference(model)
        logger.info("Inference engine initialized")
    return _inference


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Анализ аудиофайла",
    description="""Анализ аудиофайла на наличие признаков deepfake.
    
    Поддерживаемые форматы: WAV, MP3, FLAC, M4A, OGG
    Диапазон длительностей: 5 секунд - 5 минут
    Частота дискретизации: 8-48 kHz (автоматическая конвертация к 16kHz)
    
    Возвращает детализированный отчет с вероятностью подлинности,
    обнаруженными артефактами и рекомендациями.""",
)
async def analyze_audio(
    file: UploadFile = File(..., description="Аудиофайл для анализа"),
    detailed: bool = False,
    generate_spectrogram: bool = Query(
        True, description="Генерировать данные спектрограммы"
    ),
    return_format: str = Query("json", description="Формат отчета (json, text, html)"),
):
    """
    Анализ одиночного аудиофайла
    """
    logger.info(f"Received analysis request for file: {file.filename}")

    # Получаем движок инференса
    inference = get_inference()

    # Валидация файла
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Сохраняем временный файл
    temp_dir = tempfile.gettempdir()
    temp_filename = f"audio_{uuid.uuid4()}_{file.filename}"
    temp_path = os.path.join(temp_dir, temp_filename)

    try:
        # Сохраняем загруженный файл
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"File saved to temporary location: {temp_path}")

        global output_path

        SPECTROGRAM_DIR = os.path.join(tempfile.gettempdir(), "spectrograms")
        os.makedirs(SPECTROGRAM_DIR, exist_ok=True)
        spectrogram_id = str(uuid.uuid4())
        output_path = os.path.join(SPECTROGRAM_DIR, f"spectrogram_{spectrogram_id}.png")

        if generate_spectrogram == False:
            spectrogram_id = 0

        # Выполнение анализа
        result = inference.predict(
            temp_path, return_analysis=detailed, spectrogram_id=spectrogram_id
        )

        # Проверка на ошибки
        if "error" in result:
            logger.error(f"Analysis error: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])

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

        logger.info(
            f"Analysis complete: {result['classification']} with {result['confidence']:.2%} confidence"
        )

        if return_format == "json":
            return export_report(result)
        elif return_format == "text":
            return PlainTextResponse(report["human_readable"])

        else:
            # По умолчанию возвращаем комплексный отчет
            return report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # Очистка временного файла
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            logger.debug(f"Temporary file removed: {temp_path}")


@router.post(
    "/analyze/batch",
    response_model=BatchAnalysisResponse,
    summary="Пакетная обработка аудиофайлов",
    description="""Пакетный анализ нескольких аудиофайлов.
    
    Обрабатывает до 100 файлов за запрос.
    Возвращает статистику и индивидуальные результаты для каждого файла.
    Поддерживает асинхронную обработку через task_id.""",
)
async def analyze_batch(
    files: List[UploadFile] = File(..., description="Список аудиофайлов для анализа"),
    max_files: int = Query(
        100, description="Максимальное количество файлов", ge=1, le=1000
    ),
    background_tasks: BackgroundTasks = None,
):
    """
    Пакетная обработка аудиофайлов
    """
    logger.info(f"Received batch request with {len(files)} files")

    # Проверка количества файлов
    if len(files) > max_files:
        raise HTTPException(
            status_code=400, detail=f"Too many files. Maximum allowed: {max_files}"
        )

    # Создаем уникальный ID для задачи
    task_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # Сохраняем файлы во временную директорию
    temp_dir = os.path.join(tempfile.gettempdir(), task_id)
    os.makedirs(temp_dir, exist_ok=True)

    file_paths = []
    for i, file in enumerate(files):
        temp_path = os.path.join(temp_dir, f"{i}_{file.filename}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_paths.append(temp_path)

    logger.info(f"Batch task {task_id} created with {len(file_paths)} files")

    # Запускаем обработку в фоне
    from app.core.batch_processor import process_batch_async

    background_tasks.add_task(process_batch_async, task_id, file_paths, temp_dir)

    return {
        "task_id": task_id,
        "status": "processing",
        "message": "Batch processing started",
        "files_count": len(files),
        "created_at": datetime.now().isoformat(),
        "status_endpoint": f"/api/batch/status/{task_id}",
        "statistics": {},
        "summary": {},
        "results": [],
    }


@router.get(
    "/batch/status/{task_id}",
    summary="Статус пакетной обработки",
    description="Получение статуса и результатов пакетной обработки",
)
async def get_batch_status(task_id: str):
    """
    Получение статуса пакетной задачи
    """
    from app.core.batch_processor import get_batch_results

    results = get_batch_results(task_id)

    if not results:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return results


@router.post(
    "/analyze/stream",
    response_model=StreamingAnalysisResponse,
    summary="Стриминговая обработка аудио",
    description="""Анализ аудио в реальном времени через WebSocket.
    
    Принимает аудио поток через WebSocket.
    Возвращает промежуточные результаты каждые 5 секунд.
    Подходит для анализа живых трансляций и звонков.""",
)
async def analyze_stream():
    """
    Endpoint для стриминговой обработки (через WebSocket)
    """
    return {
        "message": "Use WebSocket connection at /ws/audio",
        "protocol": "WebSocket",
        "supported_formats": ["PCM", "WAV"],
        "chunk_duration": "5 seconds",
        "sample_rate": "16000 Hz",
    }


@router.get(
    "/model/info",
    summary="Информация о модели",
    description="Получение информации о загруженной модели детекции",
)
async def get_model_information():
    """
    Получение информации о модели
    """
    from app.main import get_model

    model = get_model()

    model_info = get_model_info(model)
    inference = get_inference()

    return {
        "model": model_info,
        "inference_stats": inference.get_stats(),
        "system": {
            "device": str(model_info["device"]),
            "audio_target_length": inference.target_length,
            "supported_formats": [".wav", ".mp3", ".flac", ".m4a", ".ogg"],
            "max_duration_seconds": 300,
            "min_duration_seconds": 5,
        },
    }


@router.get(
    "/system/status",
    response_model=SystemStatusResponse,
    summary="Статус системы",
    description="Полная информация о состоянии системы детекции",
)
async def get_system_status():
    """
    Получение статуса системы
    """
    from app.main import get_model, get_metrics

    model = get_model()
    metrics = get_metrics()
    inference = get_inference()

    # Информация о модели
    model_info = get_model_info(model)

    # Статистика инференса
    inference_stats = inference.get_stats()

    # Информация о системе
    import psutil
    import torch

    system_info = {
        "cpu_usage_percent": psutil.cpu_percent(),
        "memory_usage_percent": psutil.virtual_memory().percent,
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory_allocated": (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        ),
        "gpu_memory_reserved": (
            torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        ),
        "uptime_seconds": metrics.get_uptime() if metrics else 0,
    }

    return {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "model": {"loaded": model is not None, **model_info},
        "performance": {
            **inference_stats,
            "total_requests": metrics.total_requests if metrics else 0,
            "success_rate": (
                metrics.successful_requests / metrics.total_requests * 100
                if metrics and metrics.total_requests > 0
                else 100
            ),
        },
        "system": system_info,
        "limits": {
            "max_file_size_mb": 100,
            "max_batch_files": 100,
            "max_duration_seconds": 300,
            "supported_formats": [".wav", ".mp3", ".flac", ".m4a", ".ogg"],
        },
    }


@router.post(
    "/validate",
    summary="Валидация аудиофайла",
    description="""Полная проверка аудиофайла на соответствие требованиям системы.
    
    Проверяет:
    - Поддерживаемый формат (WAV, MP3, FLAC, M4A, OGG)
    - Длительность (5 секунд - 5 минут)
    - Частоту дискретизации (8-48 kHz)
    - Количество каналов (моно/стерео)
    - Качество записи (SNR, динамический диапазон)
    - Отсутствие повреждений файла
    
    Возвращает детальную информацию о файле и рекомендации.""",
)
async def validate_audio_file(
    file: UploadFile = File(..., description="Аудиофайл для валидации")
):
    """
    Расширенная валидация аудиофайла перед анализом
    """
    logger.info(f"Validation request for file: {file.filename}")

    # Проверка наличия имени файла
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Проверка размера файла
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    file.file.seek(0, 2)  # Перейти в конец файла
    file_size = file.file.tell()
    file.file.seek(0)  # Вернуться в начало

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size ({file_size / (1024 * 1024):.2f} MB) exceeds maximum limit of 100 MB",
        )

    if file_size == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    # Сохраняем временный файл
    temp_dir = tempfile.gettempdir()
    temp_filename = f"validate_{uuid.uuid4()}_{file.filename}"
    temp_path = os.path.join(temp_dir, temp_filename)

    try:
        # Сохраняем загруженный файл
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.debug(f"Validation file saved to: {temp_path}")

        # Расширенная валидация с обработкой всех ошибок
        validation_result = await perform_comprehensive_validation(
            temp_path, file.filename
        )

        # Логирование результата валидации
        if validation_result["valid"]:
            logger.info(f"File validation successful: {file.filename}")
        else:
            logger.warning(
                f"File validation failed: {file.filename} - {validation_result['error']}"
            )

        return to_json_safe(validation_result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during validation: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error during validation: {str(e)}"
        )
    finally:
        # Очистка временного файла
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.debug(f"Temporary file removed: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_path}: {str(e)}")


async def perform_comprehensive_validation(file_path: str, filename: str) -> Dict:
    """
    Выполнение комплексной валидации аудиофайла
    """
    # Базовая валидация через AudioProcessor
    is_valid, error_msg = AudioProcessor.validate_audio(file_path)

    if not is_valid:
        return {
            "valid": False,
            "filename": filename,
            "error": error_msg,
            "compatibility": {
                "supported": False,
                "recommended_action": "fix_issues_before_analysis",
                "issues": [error_msg],
            },
            "metadata": {
                "file_size_bytes": os.path.getsize(file_path),
                "validation_timestamp": datetime.now().isoformat(),
            },
        }

    try:
        # Детальный анализ файла
        file_stats = os.stat(file_path)

        # Загрузка аудио для детального анализа
        audio, sr = AudioProcessor.load_audio(file_path, normalize=False)

        # Анализ качества аудио
        quality_metrics = AudioProcessor.analyze_audio_quality(audio, sr)

        # Дополнительные проверки
        additional_checks = perform_additional_checks(audio, sr, file_path)

        # Определение рекомендаций
        recommendations = generate_validation_recommendations(
            quality_metrics, sr, additional_checks
        )

        # Расчет оценки качества
        quality_score = calculate_quality_score(quality_metrics, additional_checks)

        # Формирование детального ответа
        return {
            "valid": True,
            "filename": filename,
            "file_info": {
                "duration_seconds": quality_metrics["duration_seconds"],
                "sample_rate": sr,
                "original_sample_rate": sr,
                "channels": 1,  # Всегда конвертируется в моно
                "size_bytes": file_stats.st_size,
                "bit_depth": estimate_bit_depth(audio),
                "format": get_file_format(filename),
                **quality_metrics,
            },
            "quality_assessment": {
                "score": quality_score,
                "level": get_quality_level(quality_score),
                "snr_rating": get_snr_rating(quality_metrics["snr_db"]),
                "dynamic_range_rating": get_dynamic_range_rating(
                    quality_metrics["dynamic_range_db"]
                ),
                "harmonic_rating": get_harmonic_rating(
                    quality_metrics["harmonic_ratio"]
                ),
            },
            "compatibility": {
                "supported": True,
                "needs_conversion": sr != 16000,
                "needs_normalization": np.max(np.abs(audio)) > 0.95
                or np.max(np.abs(audio)) < 0.1,
                "recommended_action": recommendations["action"],
                "optimization_suggestions": recommendations["suggestions"],
                "estimated_processing_time": estimate_processing_time(
                    quality_metrics["duration_seconds"]
                ),
            },
            "analysis_readiness": {
                "ready_for_basic_analysis": True,
                "ready_for_detailed_analysis": quality_score > 50,
                "confidence_factor": min(quality_score / 100, 1.0),
                "potential_issues": additional_checks.get("warnings", []),
            },
            "metadata": {
                "validation_id": f"val_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "validation_timestamp": datetime.now().isoformat(),
                "system_version": "2.0",
                "checks_performed": [
                    "format_support",
                    "duration_check",
                    "sample_rate_check",
                    "file_integrity",
                    "quality_metrics",
                    "additional_checks",
                ],
            },
        }

    except Exception as e:
        logger.error(f"Error during detailed validation: {str(e)}")
        return {
            "valid": False,
            "filename": filename,
            "error": f"Detailed validation failed: {str(e)}",
            "compatibility": {
                "supported": False,
                "recommended_action": "contact_support",
                "issues": [f"Validation error: {str(e)}"],
            },
        }


def perform_additional_checks(audio: np.ndarray, sr: int, file_path: str) -> Dict:
    """
    Выполнение дополнительных проверок аудиофайла
    """
    checks = {"warnings": [], "passed": [], "failed": []}

    try:
        # Проверка на тишину (более 50% файла)
        energy = np.mean(audio**2)
        if energy < 1e-6:
            checks["warnings"].append("Very low audio energy - possible silence")
            checks["failed"].append("minimum_energy")
        else:
            checks["passed"].append("minimum_energy")

        # Проверка на клиппинг
        clipping_samples = np.sum(np.abs(audio) > 0.95)
        clipping_percentage = (clipping_samples / len(audio)) * 100
        if clipping_percentage > 1.0:  # Более 1% клиппинга
            checks["warnings"].append(
                f"Audio clipping detected ({clipping_percentage:.1f}%)"
            )
            checks["failed"].append("clipping_check")
        else:
            checks["passed"].append("clipping_check")

        # Проверка на постоянный шум (DC offset)
        dc_offset = np.mean(audio)
        if abs(dc_offset) > 0.01:
            checks["warnings"].append(f"DC offset detected ({dc_offset:.3f})")
            checks["failed"].append("dc_offset_check")
        else:
            checks["passed"].append("dc_offset_check")

        # Проверка на наличие нестандартных частот
        if sr not in [8000, 16000, 22050, 44100, 48000]:
            checks["warnings"].append(f"Non-standard sample rate: {sr} Hz")

        # Проверка длительности
        duration = len(audio) / sr
        if duration < 5:
            checks["warnings"].append(f"Very short duration: {duration:.1f} seconds")
        elif duration > 300:
            checks["warnings"].append(f"Very long duration: {duration:.1f} seconds")

        # Проверка на резкие переходы (артефакты)
        audio_diff = np.diff(audio)
        sharp_transitions = np.sum(np.abs(audio_diff) > 0.5) / len(audio_diff)
        if sharp_transitions > 0.01:
            checks["warnings"].append(
                f"Excessive sharp transitions ({sharp_transitions:.1%})"
            )
            checks["failed"].append("transition_smoothness")
        else:
            checks["passed"].append("transition_smoothness")

        return checks

    except Exception as e:
        logger.warning(f"Additional checks failed: {str(e)}")
        checks["warnings"].append(f"Additional checks incomplete: {str(e)}")
        return checks


def generate_validation_recommendations(
    quality_metrics: Dict, sr: int, additional_checks: Dict
) -> Dict:
    """
    Генерация рекомендаций на основе результатов валидации
    """
    recommendations = {"action": "ready_for_analysis", "suggestions": []}

    # Рекомендации по частоте дискретизации
    if sr != 16000:
        recommendations["suggestions"].append(
            f"Sample rate will be automatically converted from {sr} Hz to 16000 Hz"
        )

    # Рекомендации по качеству
    if quality_metrics["snr_db"] < 20:
        recommendations["suggestions"].append(
            f"Low SNR ({quality_metrics['snr_db']:.1f} dB) may affect analysis accuracy"
        )

    if quality_metrics["dynamic_range_db"] < 30:
        recommendations["suggestions"].append(
            f"Limited dynamic range ({quality_metrics['dynamic_range_db']:.1f} dB)"
        )

    if quality_metrics["harmonic_ratio"] < 0.3:
        recommendations["suggestions"].append(
            f"Low harmonic content ({(quality_metrics['harmonic_ratio'] * 100):.1f}%)"
        )

    # Рекомендации из дополнительных проверок
    if additional_checks.get("warnings"):
        for warning in additional_checks["warnings"][
            :3
        ]:  # Ограничиваем 3 предупреждениями
            recommendations["suggestions"].append(warning)

    # Рекомендации по длительности
    duration = quality_metrics["duration_seconds"]
    if duration < 10:
        recommendations["suggestions"].append(
            "Very short audio - consider using longer recordings for better accuracy"
        )
    elif duration > 180:
        recommendations["suggestions"].append(
            "Long audio - analysis may take additional time"
        )

    # Если есть серьезные проблемы
    if additional_checks.get("failed"):
        failed_checks = additional_checks["failed"]
        if "clipping_check" in failed_checks:
            recommendations["action"] = "fix_before_analysis"
            recommendations["suggestions"].append(
                "CRITICAL: Audio clipping detected - fix before analysis"
            )

        if len(failed_checks) > 2:
            recommendations["action"] = "review_before_analysis"

    return recommendations


def calculate_quality_score(quality_metrics: Dict, additional_checks: Dict) -> float:
    """
    Расчет комплексной оценки качества аудио (0-100)
    """
    score = 0
    max_score = 0

    # SNR (макс 30 баллов)
    snr = quality_metrics["snr_db"]
    snr_score = min(max(snr / 40 * 30, 0), 30)  # 40 dB = отлично
    score += snr_score
    max_score += 30

    # Динамический диапазон (макс 25 баллов)
    dynamic_range = quality_metrics["dynamic_range_db"]
    dr_score = min(max(dynamic_range / 60 * 25, 0), 25)  # 60 dB = отлично
    score += dr_score
    max_score += 25

    # Гармонический коэффициент (макс 20 баллов)
    harmonic = quality_metrics["harmonic_ratio"]
    harmonic_score = harmonic * 20
    score += harmonic_score
    max_score += 20

    # Длительность (макс 15 баллов)
    duration = quality_metrics["duration_seconds"]
    if 10 <= duration <= 180:  # Идеальная длительность
        duration_score = 15
    elif 5 <= duration < 10 or 180 < duration <= 300:
        duration_score = 10
    else:
        duration_score = 5
    score += duration_score
    max_score += 15

    # Дополнительные проверки (макс 10 баллов)
    additional_score = 10
    if additional_checks.get("failed"):
        additional_score -= len(additional_checks["failed"]) * 2
    if additional_checks.get("warnings"):
        additional_score -= min(len(additional_checks["warnings"]), 5)
    additional_score = max(additional_score, 0)
    score += additional_score
    max_score += 10

    return (score / max_score) * 100


def get_quality_level(score: float) -> str:
    """Определение уровня качества на основе оценки"""
    if score >= 85:
        return "EXCELLENT"
    elif score >= 70:
        return "GOOD"
    elif score >= 50:
        return "FAIR"
    elif score >= 30:
        return "POOR"
    else:
        return "UNACCEPTABLE"


def get_snr_rating(snr_db: float) -> str:
    """Рейтинг SNR"""
    if snr_db >= 30:
        return "EXCELLENT"
    elif snr_db >= 20:
        return "GOOD"
    elif snr_db >= 10:
        return "FAIR"
    else:
        return "POOR"


def get_dynamic_range_rating(dynamic_range_db: float) -> str:
    """Рейтинг динамического диапазона"""
    if dynamic_range_db >= 50:
        return "EXCELLENT"
    elif dynamic_range_db >= 40:
        return "GOOD"
    elif dynamic_range_db >= 30:
        return "FAIR"
    else:
        return "POOR"


def get_harmonic_rating(harmonic_ratio: float) -> str:
    """Рейтинг гармонического содержания"""
    if harmonic_ratio >= 0.7:
        return "EXCELLENT"
    elif harmonic_ratio >= 0.5:
        return "GOOD"
    elif harmonic_ratio >= 0.3:
        return "FAIR"
    else:
        return "POOR"


def estimate_bit_depth(audio: np.ndarray) -> int:
    """Оценка битности аудио"""
    # Простая эвристика для определения битности
    unique_values = len(np.unique(np.round(audio * 32768)))
    if unique_values > 65536:  # 2^16
        return 24
    elif unique_values > 256:  # 2^8
        return 16
    else:
        return 8


def get_file_format(filename: str) -> str:
    """Определение формата файла по расширению"""
    ext = os.path.splitext(filename)[1].lower()
    formats = {
        ".wav": "WAV",
        ".mp3": "MP3",
        ".flac": "FLAC",
        ".m4a": "M4A/AAC",
        ".ogg": "OGG/Vorbis",
        ".opus": "Opus",
        ".aac": "AAC",
        ".wma": "WMA",
    }
    return formats.get(ext, "UNKNOWN")


def estimate_processing_time(duration_seconds: float) -> float:
    """Оценка времени обработки"""
    # Базовое время + время пропорциональное длительности
    base_time = 1.0  # секунды
    processing_rate = 0.5  # секунды обработки на секунду аудио
    return base_time + (duration_seconds * processing_rate)


@router.get(
    "/export/report/{format}",
    summary="Экспорт шаблона отчета",
    description="Получение шаблона отчета в различных форматах",
)
async def export_report_template(
    format: str = Path(
        ...,
        description="Формат отчета (json, text, html, pdf)",
        regex="^(json|text|html|pdf)$",
    )
):
    """
    Экспорт шаблона отчета
    """
    # Создаем пример отчета
    example_report = {
        "api_format": {
            "classification": "REAL",
            "is_fake": False,
            "confidence": 0.85,
            "confidence_percent": 85.0,
            "confidence_level": "HIGH",
            "analysis": {
                "artifacts": {
                    "detected": ["Низкое соотношение гармоник/шум"],
                    "confidence_level": "MEDIUM",
                },
                "audio_quality": {
                    "snr_db": 25.5,
                    "dynamic_range_db": 45.2,
                    "harmonic_ratio": 0.7,
                    "duration_seconds": 30.5,
                    "sample_rate": 16000,
                },
            },
            "processing": {
                "processing_time_seconds": 1.23,
                "timestamp": datetime.now().isoformat(),
                "model_info": {"name": "RawNet3", "device": "cuda:0"},
            },
            "recommendations": [
                "Высокая вероятность подлинности аудио",
                "Рекомендуется проверить источник записи",
            ],
        }
    }

    # Генерация полного отчета
    from app.core.report import generate_report, export_report as export_report_func

    full_report = generate_report(
        classification="REAL",
        confidence=0.85,
        artifacts={"detected": ["Низкое соотношение гармоник/шум"]},
        processing_time=1.23,
        audio_quality={
            "snr_db": 25.5,
            "dynamic_range_db": 45.2,
            "harmonic_ratio": 0.7,
            "duration_seconds": 30.5,
            "sample_rate": 16000,
        },
    )

    if format == "json":
        return JSONResponse(content=example_report)

    elif format == "text":
        return export_report_func(full_report, "text")

    elif format == "html":
        html_content = export_report_func(full_report, "html")
        return HTMLResponse(content=html_content)

    elif format == "pdf":
        # Для PDF нужны дополнительные библиотеки
        raise HTTPException(status_code=501, detail="PDF export not implemented yet")

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")


# Дополнительные утилитарные endpoints
@router.get("/formats")
async def get_supported_formats():
    """Получение списка поддерживаемых форматов"""
    return {
        "audio_formats": [".wav", ".mp3", ".flac", ".m4a", ".ogg"],
        "export_formats": ["json", "text", "html"],
        "analysis_types": ["fast", "detailed", "comprehensive"],
    }


@router.get("/statistics")
async def get_statistics():
    """Получение статистики работы API"""
    from app.main import get_metrics

    metrics = get_metrics()
    inference = get_inference()

    return {
        "requests": {
            "total": metrics.total_requests if metrics else 0,
            "successful": metrics.successful_requests if metrics else 0,
            "failed": metrics.failed_requests if metrics else 0,
            "last_hour": metrics.get_requests_last_hour() if metrics else [],
        },
        "inference": inference.get_stats() if inference else {},
        "performance": {
            "avg_response_time": metrics.avg_processing_time if metrics else 0,
            "uptime_seconds": metrics.get_uptime() if metrics else 0,
        },
    }


from fastapi.responses import FileResponse
from fastapi import BackgroundTasks


@router.get("/spectrogram/{spectrogram_id}")
def get_spectrogram(spectrogram_id: str, background_tasks: BackgroundTasks):
    SPECTROGRAM_DIR = os.path.join(tempfile.gettempdir(), "spectrograms")
    os.makedirs(SPECTROGRAM_DIR, exist_ok=True)
    path = os.path.join(SPECTROGRAM_DIR, f"spectrogram_{spectrogram_id}.png")

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Spectrogram not found")

    background_tasks.add_task(os.remove, path)

    return FileResponse(path, media_type="image/png")


# Обработка ошибок
@router.get("/error/test")
async def test_error():
    """Endpoint для тестирования обработки ошибок"""
    raise HTTPException(status_code=418, detail="I'm a teapot - это тестовая ошибка")


import numpy as np
import torch


def to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, torch.Tensor):
        return to_json_safe(obj.tolist())
    return obj


# Нужно добавить в конец файла:
from fastapi.responses import HTMLResponse
