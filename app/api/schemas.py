# app/api/schemas.py
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum


class ClassificationResult(str, Enum):
    """Результаты классификации"""

    REAL = "REAL"
    FAKE = "FAKE"


class ConfidenceLevel(str, Enum):
    """Уровни уверенности"""

    VERY_HIGH = "VERY HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ArtifactType(str, Enum):
    """Типы артефактов"""

    SPECTRAL = "spectral_anomalies"
    PHONEME = "phoneme_transitions"
    VOCODER = "vocoder_artifacts"
    STATISTICAL = "statistical_anomalies"


class ProcessingStatus(str, Enum):
    """Статусы обработки"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ArtifactDetail(BaseModel):
    """Детали артефакта"""

    type: ArtifactType
    description: str
    severity: float = Field(ge=0.0, le=1.0, description="Серьезность артефакта (0-1)")
    timestamp: Optional[float] = Field(
        None, description="Временная метка артефакта в секундах"
    )
    frequency: Optional[float] = Field(None, description="Частота артефакта в Hz")


class AudioQualityMetrics(BaseModel):
    """Метрики качества аудио"""

    snr_db: float = Field(..., description="Signal-to-Noise Ratio в dB")
    dynamic_range_db: float = Field(..., description="Динамический диапазон в dB")
    harmonic_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Коэффициент гармоник"
    )
    duration_seconds: float = Field(
        ..., gt=0, description="Длительность аудио в секундах"
    )
    sample_rate: int = Field(..., description="Частота дискретизации в Hz")
    samples_count: int = Field(..., description="Количество сэмплов")
    max_amplitude: float = Field(..., description="Максимальная амплитуда")


class SpectrogramData(BaseModel):
    """Данные спектрограммы"""

    has_data: bool = Field(..., description="Наличие данных спектрограммы")
    time_frames: Optional[List[float]] = Field(None, description="Временные метки")
    frequency_bins: Optional[List[float]] = Field(None, description="Частотные бины")
    anomalies: Optional[List[Dict[str, Any]]] = Field(
        None, description="Обнаруженные аномалии"
    )
    image_base64: Optional[str] = Field(None, description="Спектрограмма в base64")


class ModelInfo(BaseModel):
    """Информация о модели"""

    name: str = Field(..., description="Название модели")
    device: str = Field(..., description="Устройство выполнения")
    total_parameters: int = Field(..., description="Общее количество параметров")
    trainable_parameters: int = Field(
        ..., description="Количество обучаемых параметров"
    )
    input_shape: List[int] = Field(..., description="Форма входных данных")


class AnalysisResponse(BaseModel):
    """Ответ анализа аудио"""

    # Основные результаты
    classification: ClassificationResult = Field(
        ..., description="Результат классификации"
    )
    is_fake: bool = Field(..., description="Флаг синтезированного аудио")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Уверенность модели (0-1)"
    )
    confidence_percent: float = Field(
        ..., ge=0.0, le=100.0, description="Уверенность в процентах"
    )
    confidence_level: ConfidenceLevel = Field(..., description="Уровень уверенности")

    # Детали анализа
    analysis: Dict[str, Any] = Field(
        ...,
        description="Детализированные результаты анализа",
        example={
            "artifacts": {
                "detected": ["Аномалии в частотном спектре"],
                "confidence_level": "MEDIUM",
            },
            "audio_quality": {"snr_db": 25.5, "duration_seconds": 30.2},
        },
    )

    # Вероятности классов
    probabilities: Dict[str, float] = Field(
        ...,
        description="Вероятности для каждого класса",
        example={"REAL": 0.85, "FAKE": 0.15},
    )

    # Производительность
    processing_time_seconds: float = Field(
        ..., gt=0, description="Время обработки в секундах"
    )
    timestamp: datetime = Field(..., description="Время выполнения анализа")
    inference_id: str = Field(..., description="Уникальный идентификатор инференса")

    # Информация о модели
    model_info: Optional[ModelInfo] = Field(
        None, description="Информация о используемой модели"
    )

    # Рекомендации
    recommendations: List[str] = Field(
        ..., description="Рекомендации по дальнейшим действиям", min_items=1
    )

    # Дополнительные данные
    spectrogram: Optional[SpectrogramData] = Field(
        None, description="Данные спектрограммы"
    )
    audio_metadata: Optional[Dict[str, Any]] = Field(
        None, description="Метаданные аудиофайла"
    )

    class Config:
        schema_extra = {
            "example": {
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
                "probabilities": {"REAL": 0.85, "FAKE": 0.15},
                "processing_time_seconds": 1.23,
                "timestamp": "2024-01-15T10:30:00.000Z",
                "inference_id": "inf_20240115_103000_1",
                "recommendations": [
                    "Высокая вероятность подлинности аудио",
                    "Рекомендуется проверить источник записи",
                ],
            }
        }

    @validator("probabilities")
    def validate_probabilities(cls, v):
        """Валидация вероятностей"""
        if not all(0.0 <= p <= 1.0 for p in v.values()):
            raise ValueError("Probabilities must be between 0 and 1")
        if abs(sum(v.values()) - 1.0) > 0.01:
            raise ValueError("Probabilities must sum to approximately 1")
        return v


class BatchFileResult(BaseModel):
    """Результат обработки файла в пакете"""

    filename: str = Field(..., description="Имя файла")
    success: bool = Field(..., description="Успешность обработки")
    result: Optional[AnalysisResponse] = Field(None, description="Результат анализа")
    error: Optional[str] = Field(None, description="Ошибка при обработке")
    processing_time: float = Field(..., description="Время обработки файла")


class BatchAnalysisResponse(BaseModel):
    """Ответ пакетной обработки"""

    task_id: str = Field(..., description="Идентификатор задачи")
    status: ProcessingStatus = Field(..., description="Статус обработки")
    created_at: datetime = Field(..., description="Время создания задачи")
    completed_at: Optional[datetime] = Field(None, description="Время завершения")

    # Статистика
    statistics: Dict[str, Any] = Field(
        ...,
        description="Статистика обработки",
        example={
            "total_files": 10,
            "successful": 8,
            "failed": 2,
            "total_processing_time": 12.5,
            "avg_time_per_file": 1.25,
        },
    )

    # Сводка
    summary: Dict[str, Any] = Field(
        ...,
        description="Сводка результатов",
        example={"real_count": 7, "fake_count": 1, "fake_percentage": 12.5},
    )

    # Результаты
    results: List[BatchFileResult] = Field(
        ..., description="Результаты обработки файлов"
    )

    # Неудачные файлы
    failed_files: List[Dict[str, str]] = Field(
        default_factory=list, description="Информация о неудачных файлах"
    )

    # Дополнительная информация
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Дополнительные метаданные"
    )


class StreamingChunkResult(BaseModel):
    """Результат анализа чанка аудио"""

    chunk_id: str = Field(..., description="Идентификатор чанка")
    timestamp: datetime = Field(..., description="Время анализа")
    classification: ClassificationResult = Field(
        ..., description="Результат классификации"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность")
    duration_seconds: float = Field(..., description="Длительность чанка")
    artifacts_detected: List[str] = Field(
        default_factory=list, description="Обнаруженные артефакты"
    )


class StreamingAnalysisResponse(BaseModel):
    """Ответ стримингового анализа"""

    session_id: str = Field(..., description="Идентификатор сессии")
    start_time: datetime = Field(..., description="Время начала анализа")
    status: ProcessingStatus = Field(..., description="Статус стриминга")

    # Результаты по чанкам
    chunks: List[StreamingChunkResult] = Field(
        default_factory=list, description="Результаты анализа чанков"
    )

    # Агрегированные результаты
    aggregated_result: Optional[AnalysisResponse] = Field(
        None, description="Агрегированный результат всей сессии"
    )

    # Статистика
    statistics: Dict[str, Any] = Field(
        ...,
        description="Статистика стриминга",
        example={
            "total_chunks": 10,
            "total_duration_seconds": 50.0,
            "avg_confidence": 0.78,
            "real_percentage": 80.0,
            "fake_percentage": 20.0,
        },
    )

    # Информация о подключении
    connection_info: Dict[str, Any] = Field(
        ...,
        description="Информация о подключении",
        example={
            "protocol": "WebSocket",
            "sample_rate": 16000,
            "chunk_size_seconds": 5.0,
            "bit_depth": 16,
        },
    )


class SystemMetrics(BaseModel):
    """Метрики системы"""

    requests_total: int = Field(..., description="Общее количество запросов")
    requests_successful: int = Field(..., description="Успешные запросы")
    requests_failed: int = Field(..., description="Неудачные запросы")
    avg_processing_time: float = Field(..., description="Среднее время обработки")
    uptime_seconds: float = Field(..., description="Время работы системы")
    memory_usage_percent: float = Field(..., description="Использование памяти")
    cpu_usage_percent: float = Field(..., description="Использование CPU")


class ModelMetrics(BaseModel):
    """Метрики модели"""

    total_inferences: int = Field(..., description="Общее количество инференсов")
    avg_inference_time: float = Field(..., description="Среднее время инференса")
    accuracy_estimate: float = Field(..., ge=0.0, le=1.0, description="Оценка точности")
    precision_estimate: float = Field(
        ..., ge=0.0, le=1.0, description="Оценка precision"
    )
    recall_estimate: float = Field(..., ge=0.0, le=1.0, description="Оценка recall")
    f1_score_estimate: float = Field(..., ge=0.0, le=1.0, description="Оценка F1-score")


class SystemStatusResponse(BaseModel):
    """Ответ с статусом системы"""

    status: str = Field(..., description="Общий статус системы")
    timestamp: datetime = Field(..., description="Время проверки")

    # Состояние системы
    system: SystemMetrics = Field(..., description="Метрики системы")
    model: ModelMetrics = Field(..., description="Метрики модели")

    # Информация о загрузке
    load: Dict[str, float] = Field(
        ...,
        description="Загрузка системы",
        example={
            "current_requests": 5,
            "max_concurrent_requests": 100,
            "queue_size": 0,
        },
    )

    # Подсистемы
    subsystems: Dict[str, bool] = Field(
        ...,
        description="Статус подсистем",
        example={
            "model_loaded": True,
            "database_connected": True,
            "cache_available": True,
        },
    )

    # Ограничения
    limits: Dict[str, Any] = Field(
        ...,
        description="Ограничения системы",
        example={
            "max_file_size_mb": 100,
            "max_duration_seconds": 300,
            "max_batch_files": 100,
        },
    )


class ErrorResponse(BaseModel):
    """Ответ с ошибкой"""

    error: str = Field(..., description="Сообщение об ошибке")
    code: int = Field(..., description="Код ошибки")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Время ошибки"
    )
    details: Optional[Dict[str, Any]] = Field(None, description="Детали ошибки")
    suggestion: Optional[str] = Field(None, description="Предложение по исправлению")

    class Config:
        schema_extra = {
            "example": {
                "error": "File format not supported",
                "code": 400,
                "timestamp": "2024-01-15T10:30:00.000Z",
                "details": {"filename": "audio.mp4"},
                "suggestion": "Convert to WAV, MP3, or FLAC format",
            }
        }


class ValidationResponse(BaseModel):
    """Ответ валидации файла"""

    valid: bool = Field(..., description="Результат валидации")
    filename: str = Field(..., description="Имя файла")
    file_info: Optional[Dict[str, Any]] = Field(None, description="Информация о файле")
    error: Optional[str] = Field(None, description="Ошибка валидации")
    compatibility: Dict[str, Any] = Field(
        ...,
        description="Совместимость с системой",
        example={
            "supported": True,
            "needs_conversion": False,
            "recommended_action": "ready_for_analysis",
        },
    )


class ExportFormat(str, Enum):
    """Форматы экспорта"""

    JSON = "json"
    TEXT = "text"
    HTML = "html"
    PDF = "pdf"


class ExportRequest(BaseModel):
    """Запрос на экспорт отчета"""

    format: ExportFormat = Field(..., description="Формат экспорта")
    include_spectrogram: bool = Field(True, description="Включать спектрограмму")
    include_details: bool = Field(True, description="Включать детали анализа")
    language: str = Field("ru", description="Язык отчета")


# Дополнительные схемы для WebSocket
class WebSocketMessage(BaseModel):
    """Сообщение WebSocket"""

    type: str = Field(..., description="Тип сообщения")
    data: Dict[str, Any] = Field(..., description="Данные сообщения")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Время сообщения"
    )


class AudioChunkMessage(BaseModel):
    """Сообщение с аудио чанком"""

    chunk_id: str = Field(..., description="Идентификатор чанка")
    audio_data: str = Field(..., description="Аудио данные в base64")
    sample_rate: int = Field(..., description="Частота дискретизации")
    channels: int = Field(1, description="Количество каналов")
    duration_ms: float = Field(..., description="Длительность чанка в миллисекундах")


class AnalysisResultMessage(BaseModel):
    """Сообщение с результатом анализа"""

    chunk_id: str = Field(..., description="Идентификатор чанка")
    classification: ClassificationResult = Field(
        ..., description="Результат классификации"
    )
    confidence: float = Field(..., description="Уверенность")
    artifacts: List[str] = Field(
        default_factory=list, description="Обнаруженные артефакты"
    )
    processing_time_ms: float = Field(
        ..., description="Время обработки в миллисекундах"
    )
