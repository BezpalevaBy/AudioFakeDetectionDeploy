# app/api/streaming.py
import json
import base64
import numpy as np
import io
import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional

import torch
import torchaudio
from fastapi import WebSocket, WebSocketDisconnect

from app.core.audio import AudioProcessor
from app.api.schemas import AudioChunkMessage, ClassificationResult

# ================= LOGGING SETUP =================
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# ================= DEVICE =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= AUDIO SETTINGS =================
SAMPLE_RATE = 48000  # 48 kHz
BYTES_PER_SAMPLE = 2  # int16
CHUNK_SECONDS = 5  # сколько секунд на один анализ


# ================= AUDIO INFERENCE =================
class AudioInference:
    """Обёртка для RawNetLite с поддержкой predict_from_bytes"""

    def __init__(self, model, target_length: int = 64600):
        self.model = model.to(DEVICE)
        self.model.eval()
        self.target_length = target_length

    def _preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Подгоняем аудио к длине target_length и делаем тензор"""
        if len(audio) > self.target_length:
            audio = audio[: self.target_length]
        elif len(audio) < self.target_length:
            padding = self.target_length - len(audio)
            audio = torch.nn.functional.pad(audio, (0, padding))
        tensor = audio.unsqueeze(0).unsqueeze(0).to(DEVICE)
        logger.debug(f"Preprocessed tensor shape: {tensor.shape}")
        return tensor

    def predict_from_bytes(self, audio_bytes: bytes) -> dict:
        logger.debug(f"Predicting audio chunk, bytes size: {len(audio_bytes)}")
        try:
            audio_file = io.BytesIO(audio_bytes)
            waveform, sr = torchaudio.load(audio_file)
            audio = waveform[0]  # первый канал
            logger.debug(f"Loaded waveform, shape: {waveform.shape}, sample_rate: {sr}")
            tensor = self._preprocess_audio(audio)
            with torch.no_grad():
                out = self.model(tensor)
            fake_prob = torch.sigmoid(out[0, 0]).item()
            real_prob = 1.0 - fake_prob
            is_fake = fake_prob > 0.5
            label = "FAKE" if is_fake else "REAL"
            logger.debug(
                f"Prediction result: {label}, confidence: {fake_prob if is_fake else real_prob}"
            )
            return {
                "classification": label,
                "is_fake": is_fake,
                "confidence": fake_prob if is_fake else real_prob,
                "probabilities": {"real": real_prob, "fake": fake_prob},
            }
        except Exception as e:
            logger.error(f"Error in predict_from_bytes: {str(e)}")
            return {"error": str(e)}


# ================= STREAMING SESSION =================
class StreamingSession:
    """Сессия стримингового анализа аудио"""

    def __init__(self, websocket: WebSocket, model, metrics):
        self.websocket = websocket
        self.model = model
        self.metrics = metrics
        self.session_id = f"stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = datetime.now()

        # Статистика
        self.chunks_received = 0
        self.chunks_processed = 0
        self.total_audio_duration = 0.0
        self.chunk_results = []

        # Буфер
        self.audio_buffer = []
        self.buffer_duration = 0.0
        self.target_chunk_duration = CHUNK_SECONDS

        # Инференс
        self.inference = AudioInference(model)
        self.is_active = True

        logger.info(f"Streaming session created: {self.session_id}")

    async def process_chunk(
        self, chunk_data: bytes, sample_rate: int
    ) -> Optional[Dict]:
        try:
            chunk_start_time = datetime.now()

            # Декодируем и нормализуем
            audio_array = (
                np.frombuffer(chunk_data, dtype=np.int16).astype(np.float32) / 32768.0
            )
            chunk_duration = len(audio_array) / sample_rate
            self.total_audio_duration += chunk_duration
            self.audio_buffer.append(audio_array)
            self.buffer_duration += chunk_duration

            logger.info(
                f"Processing chunk {self.chunks_processed+1}, duration {chunk_duration:.3f}s, bytes {len(chunk_data)}"
            )
            logger.debug(f"Current buffer duration: {self.buffer_duration:.3f}s")

            if self.buffer_duration >= self.target_chunk_duration:
                # Собираем полный буфер
                full_chunk = np.concatenate(self.audio_buffer)

                # Пересэмплируем в 48000 Hz
                if sample_rate != 48000:
                    full_chunk = AudioProcessor.resample_audio(
                        full_chunk, sample_rate, 48000
                    )
                    sample_rate = 48000

                # конвертируем в корректный WAV
                wav_bytes = AudioProcessor.convert_to_wav(full_chunk, sample_rate)

                if wav_bytes:
                    result = await self.analyze_audio_chunk(wav_bytes, sample_rate)
                    if result:
                        result["chunk_duration"] = self.buffer_duration
                        result["buffer_size"] = len(self.audio_buffer)
                        result["processing_time"] = (
                            datetime.now() - chunk_start_time
                        ).total_seconds()
                        self.audio_buffer.clear()
                        self.buffer_duration = 0.0
                        logger.info(f"Chunk analysis completed: {result['chunk_id']}")
                        return result

            return None
        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            return None

    async def analyze_audio_chunk(
        self, wav_bytes: bytes, sample_rate: int
    ) -> Optional[Dict]:
        logger.debug(
            f"Analyzing audio chunk, size={len(wav_bytes)} bytes, sample_rate={sample_rate}"
        )
        try:
            result = self.inference.predict_from_bytes(wav_bytes)
            if "error" in result:
                logger.warning(f"Chunk analysis error: {result['error']}")
                return None

            chunk_result = {
                "chunk_id": f"chunk_{self.chunks_processed}",
                "timestamp": datetime.now().isoformat(),
                "classification": result["classification"],
                "confidence": result["confidence"],
                "is_fake": result["is_fake"],
                "probabilities": result["probabilities"],
                "artifacts": [],
                "audio_quality": {},
            }
            self.chunks_processed += 1
            self.chunk_results.append(chunk_result)

            if self.metrics:
                self.metrics.increment_streaming_chunks()

            logger.debug(f"Chunk processed: {chunk_result}")
            return chunk_result
        except Exception as e:
            logger.error(f"Error in chunk analysis: {str(e)}")
            return None

    def get_session_summary(self) -> Dict:
        if not self.chunk_results:
            return {}
        real_count = sum(1 for r in self.chunk_results if not r["is_fake"])
        fake_count = len(self.chunk_results) - real_count
        avg_confidence = np.mean([r["confidence"] for r in self.chunk_results])
        overall_classification = (
            ClassificationResult.FAKE
            if fake_count > real_count
            else ClassificationResult.REAL
        )
        overall_confidence = np.mean(
            [
                r["confidence"]
                for r in self.chunk_results
                if (
                    r["is_fake"]
                    if overall_classification == ClassificationResult.FAKE
                    else not r["is_fake"]
                )
            ]
        )
        all_artifacts = [a for r in self.chunk_results for a in r.get("artifacts", [])]

        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": self.total_audio_duration,
            "chunks_processed": self.chunks_processed,
            "overall_classification": overall_classification,
            "overall_confidence": overall_confidence,
            "statistics": {
                "real_count": real_count,
                "fake_count": fake_count,
                "fake_percentage": (fake_count / len(self.chunk_results)) * 100,
                "avg_confidence": avg_confidence,
                "total_artifacts": len(set(all_artifacts)),
            },
        }

    async def close(self):
        self.is_active = False
        logger.info(f"Streaming session closed: {self.session_id}")


# ================= STREAMING MANAGER =================
class StreamingManager:
    def __init__(self):
        self.sessions: Dict[str, StreamingSession] = {}

    def create_session(self, websocket: WebSocket, model, metrics) -> StreamingSession:
        session = StreamingSession(websocket, model, metrics)
        self.sessions[session.session_id] = session
        return session

    def remove_session(self, session_id: str):
        self.sessions.pop(session_id, None)

    def get_session(self, session_id: str) -> Optional[StreamingSession]:
        return self.sessions.get(session_id)


streaming_manager = StreamingManager()


# ================= STREAMING HANDLERS =================
async def audio_stream(websocket: WebSocket, model, metrics):
    logger.debug("Starting audio_stream handler")
    session = streaming_manager.create_session(websocket, model, metrics)
    await websocket.accept()

    await websocket.send_json(
        {
            "type": "session_start",
            "session_id": session.session_id,
            "message": "Streaming session started",
            "config": {
                "target_chunk_duration": session.target_chunk_duration,
                "sample_rate": SAMPLE_RATE,
                "format": "16-bit PCM",
            },
        }
    )

    try:
        while session.is_active:
            try:
                msg = await websocket.receive_json()
                logger.debug(f"Received message: {msg}")
                msg_type = msg.get("type")

                if msg_type == "audio_chunk":
                    await handle_audio_chunk(session, msg)
                elif msg_type == "control":
                    await handle_control_message(session, msg)
                elif msg_type == "heartbeat":
                    await websocket.send_json(
                        {
                            "type": "heartbeat_response",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                else:
                    await websocket.send_json(
                        {"type": "error", "error": f"Unknown type: {msg_type}"}
                    )
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break
            except Exception as e:
                logger.error(f"Error in audio_stream loop: {str(e)}")
                await websocket.send_json({"type": "error", "error": str(e)})
    finally:
        logger.debug("Finishing streaming session")
        await finish_streaming_session(session)


async def handle_audio_chunk(session: StreamingSession, message_data: dict):
    try:
        chunk_msg = AudioChunkMessage(**message_data["data"])
        logger.debug(f"Handling audio chunk: {chunk_msg.chunk_id}")
        audio_bytes = base64.b64decode(chunk_msg.audio_data)
        result = await session.process_chunk(audio_bytes, SAMPLE_RATE)
        if result:
            await session.websocket.send_json(
                {"type": "analysis_result", "data": result}
            )
        session.chunks_received += 1
        await session.websocket.send_json(
            {
                "type": "chunk_received",
                "chunk_id": chunk_msg.chunk_id,
                "chunks_received": session.chunks_received,
            }
        )
    except Exception as e:
        logger.error(f"Error handling audio chunk: {e}")
        await session.websocket.send_json({"type": "error", "error": str(e)})


async def handle_control_message(session: StreamingSession, message_data: dict):
    command = message_data.get("data", {}).get("command")
    if command == "pause":
        session.is_active = False
        await session.websocket.send_json(
            {"type": "control_response", "message": "Paused"}
        )
    elif command == "resume":
        session.is_active = True
        await session.websocket.send_json(
            {"type": "control_response", "message": "Resumed"}
        )
    elif command == "status":
        await session.websocket.send_json(
            {"type": "status_response", "data": session.get_session_summary()}
        )
    elif command == "stop":
        session.is_active = False
        await finish_streaming_session(session)
    else:
        await session.websocket.send_json(
            {"type": "error", "error": f"Unknown command: {command}"}
        )


async def finish_streaming_session(session: StreamingSession):
    logger.debug(f"Finishing session {session.session_id}, processing remaining buffer")
    try:
        if session.audio_buffer:
            logger.debug(
                f"Buffer reached target duration, converting to WAV at 48kHz..."
            )
            full_chunk = np.concatenate(session.audio_buffer)

            if sample_rate != 48000:
                full_chunk = AudioProcessor.resample_audio(
                    full_chunk, sample_rate, 48000
                )
                sample_rate = 48000
            wav_bytes = AudioProcessor.convert_to_wav(full_chunk, SAMPLE_RATE)
            if wav_bytes:
                await session.analyze_audio_chunk(wav_bytes, SAMPLE_RATE)
        summary = session.get_session_summary()
        logger.debug(f"Session summary: {summary}")
        await session.websocket.send_json({"type": "session_summary", "data": summary})
    finally:
        try:
            await session.websocket.close()
        except:
            pass
        streaming_manager.remove_session(session.session_id)
        await session.close()
