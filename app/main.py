# app/main.py
from fastapi import FastAPI, WebSocket, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import UploadFile
import os
import asyncio
from contextlib import asynccontextmanager
import logging
import torch
from app.api.routes import router
from app.api.streaming import audio_stream
from app.core.model import load_model
from app.core.metrics import SystemMetrics
from app.core.inference import AudioInference
import base64
import asyncio
import time
from datetime import datetime
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from .model import aasist3

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Глобальная модель и метрики
_model = None
_metrics = None

model_name = "MTUCI/AASIST3"
model = aasist3.from_pretrained(model_name)

inference = AudioInference(model)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # """Управление жизненным циклом приложения"""
    global _model, _metrics

    # Загрузка при старте
    logger.info("Starting Deepfake Audio Detection System")

    # Инициализация метрик
    _metrics = SystemMetrics()

    # Ленивая загрузка модели (будет загружена при первом использовании)
    logger.info("Application startup complete")

    yield

    # Завершение работы
    logger.info("Shutting down Deepfake Audio Detection System")
    if _model:
        # Очистка ресурсов модели
        del _model


def get_model():
    """Функция для получения модели (ленивая загрузка)"""
    global _model
    if _model is None:
        try:
            _model = load_model()
            logger.info(
                f"Model loaded successfully on device: {next(_model.parameters()).device}"
            )
            _metrics.model_load_time = datetime.now()
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    return _model


def get_metrics():
    """Получение метрик системы"""
    global _metrics
    return _metrics


app = FastAPI(
    title="Deepfake Audio Detection System",
    description="AI-based detection of synthesized audio with comprehensive analysis",
    version="2.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API маршруты
app.include_router(router, prefix="/api", tags=["API"])

# Web интерфейс
app.mount("/static", StaticFiles(directory="app/web/static"), name="static")
templates = Jinja2Templates(directory="app/web/templates")


@app.get("/")
async def index(request: Request):
    """Главная страница веб-интерфейса"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "system_status": "online", "api_version": "2.0"},
    )


@app.get("/dashboard")
async def dashboard(request: Request):
    """Панель управления системой"""
    model_path = os.getenv("MODEL_PATH", "models/rawnet_lite.pt")
    model = get_model(model_path)
    metrics = get_metrics()

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "model_loaded": model is not None,
            "model_device": (
                str(next(model.parameters()).device) if model else "Not loaded"
            ),
            "total_requests": metrics.total_requests,
            "avg_processing_time": metrics.avg_processing_time,
            "accuracy": metrics.accuracy,
        },
    )


import time
import base64
import asyncio
import logging
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect

# ================= LOGGING =================
logger = logging.getLogger("audio_ws")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# ================= CONFIG =================
SAMPLE_RATE = 48000
BYTES_PER_SAMPLE = 2  # int16
CHUNK_SECONDS = 5  # сколько секунд на один анализ

# ================= IMPORTS =================
import time
import io
import base64
import asyncio
import logging
from datetime import datetime
import numpy as np
from scipy.io.wavfile import write as wav_write
from fastapi import WebSocket

# ================= LOGGER =================
logger = logging.getLogger("audio_ws")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# ================= WAV CONVERTER =================
def pcm_bytes_to_wav(audio_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """
    Конвертирует PCM int16 в валидный WAV в памяти
    """
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

    # Создаём BytesIO
    wav_io = io.BytesIO()
    wav_write(wav_io, sample_rate, audio_array)
    return wav_io.getvalue()


# ================= WEBSOCKET HANDLER =================
@app.websocket("/ws/audio")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    audio_buffer = bytearray()
    chunks_processed = 0
    stream_start = time.time()

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            logger.debug(f"Received message type: {msg_type}")

            if msg_type != "audio_chunk":
                logger.warning(f"Ignoring message of type {msg_type}")
                continue

            # Декодируем PCM из base64 и буферизуем
            audio_bytes = base64.b64decode(data["data"]["audio_data"])
            audio_buffer.extend(audio_bytes)
            buffer_duration = len(audio_buffer) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
            logger.debug(
                f"Current buffer duration: {buffer_duration:.3f}s, bytes: {len(audio_buffer)}"
            )

            # Ждём, пока наберётся CHUNK_SECONDS
            if buffer_duration < CHUNK_SECONDS:
                continue

            # Конвертируем в валидный WAV
            wav_bytes = pcm_bytes_to_wav(bytes(audio_buffer))
            audio_buffer.clear()
            chunks_processed += 1
            logger.info(
                f"Processing chunk {chunks_processed}, duration {buffer_duration:.3f}s, bytes {len(wav_bytes)}"
            )

            # Асинхронный запуск инференса
            start = time.time()
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, inference.predict_from_bytes, wav_bytes
            )
            processing_time = time.time() - start

            if "error" in result:
                logger.error(f"Chunk prediction error: {result['error']}")
                await websocket.send_json(
                    {"type": "analysis_error", "message": result["error"]}
                )
                continue

            payload = {
                "type": "analysis_result",
                "data": {
                    "chunk_id": chunks_processed,
                    "classification": result["classification"],
                    "confidence": result["confidence"],
                    "is_fake": result["is_fake"],
                    "probabilities": result["probabilities"],
                    "chunk_duration": round(buffer_duration, 2),
                    "processing_time": round(processing_time, 3),
                    "timestamp": datetime.now().isoformat(),
                },
            }

            logger.info(f"Sending analysis result for chunk {chunks_processed}")
            await websocket.send_json(payload)

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Сводка по сессии
        duration = time.time() - stream_start
        await websocket.send_json(
            {
                "type": "session_summary",
                "data": {
                    "duration_seconds": round(duration, 3),
                    "chunks_processed": chunks_processed,
                },
            }
        )
        await websocket.close()
        logger.info("WebSocket connection closed")


# Batch processing endpoint
@app.post("/api/batch")
async def batch_process(files: list[UploadFile], background_tasks: BackgroundTasks):
    """Пакетная обработка аудиофайлов"""
    from app.core.batch_processor import process_batch

    metrics = get_metrics()
    metrics.increment_batch_requests()

    # Запуск обработки в фоне
    task_id = f"batch_{datetime.now().timestamp()}"
    background_tasks.add_task(process_batch, files, task_id, metrics)

    return {
        "task_id": task_id,
        "status": "processing",
        "message": "Batch processing started",
        "files_count": len(files),
    }


@app.get("/api/batch/{task_id}")
async def get_batch_status(task_id: str):
    """Получение статуса пакетной обработки"""
    from app.core.batch_processor import get_batch_status as get_status

    status = get_status(task_id)
    return status or {"error": "Task not found"}


# Health check с расширенной информацией
@app.get("/health")
async def health_check():
    """Расширенная проверка здоровья системы"""
    try:
        model = get_model()
        metrics = get_metrics()

        # Проверка доступности GPU
        gpu_available = torch.cuda.is_available() if torch else False
        gpu_info = None

        if gpu_available:
            gpu_info = {
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.device_count() > 0
                    else None
                ),
            }

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system": {
                "model_loaded": model is not None,
                "model_device": str(next(model.parameters()).device) if model else None,
                "model_name": model.__class__.__name__ if model else None,
            },
            "resources": {
                "gpu_available": gpu_available,
                "gpu_info": gpu_info,
                "memory_allocated": (
                    torch.cuda.memory_allocated() if gpu_available else None
                ),
                "memory_reserved": (
                    torch.cuda.memory_reserved() if gpu_available else None
                ),
            },
            "metrics": {
                "total_requests": metrics.total_requests,
                "successful_requests": metrics.successful_requests,
                "failed_requests": metrics.failed_requests,
                "avg_response_time": metrics.avg_processing_time,
                "streaming_connections": metrics.streaming_connections,
            },
            "version": {"api": "2.0", "model": os.getenv("MODEL_VERSION", "1.0.0")},
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            },
        )


# System metrics endpoint
@app.get("/api/metrics")
async def get_system_metrics():
    """Получение метрик системы"""
    metrics = get_metrics()

    return {
        "requests": {
            "total": metrics.total_requests,
            "successful": metrics.successful_requests,
            "failed": metrics.failed_requests,
            "last_hour": metrics.get_requests_last_hour(),
        },
        "performance": {
            "avg_processing_time": metrics.avg_processing_time,
            "min_processing_time": metrics.min_processing_time,
            "max_processing_time": metrics.max_processing_time,
            "current_queue_size": metrics.current_queue_size,
        },
        "system": {
            "streaming_connections": metrics.streaming_connections,
            "batch_tasks_running": metrics.batch_tasks_running,
            "model_load_time": (
                metrics.model_load_time.isoformat() if metrics.model_load_time else None
            ),
            "uptime_seconds": metrics.get_uptime(),
        },
        "accuracy": {
            "estimated_accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
        },
    }


# Clear metrics endpoint (для отладки)
@app.delete("/api/metrics")
async def clear_metrics():
    """Очистка метрик системы"""
    metrics = get_metrics()
    metrics.reset()
    return {"status": "metrics_cleared"}
