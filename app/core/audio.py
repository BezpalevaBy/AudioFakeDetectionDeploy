# app/core/audio.py
import librosa
import numpy as np
import soundfile as sf
import wave
import io
import logging
from typing import Tuple, Optional, Union
from scipy import signal
import resampy
import torch
import torchaudio
from scipy.io.wavfile import write as wav_write

logger = logging.getLogger(__name__)

# Константы
TARGET_SR = 16000
MIN_DURATION = 5.0  # 5 секунд
MAX_DURATION = 300.0  # 5 минут
SUPPORTED_FORMATS = [".wav", ".flac", ".mp3", ".m4a", ".ogg"]


class AudioProcessor:
    """Класс для обработки аудиофайлов с поддержкой различных форматов"""

    @staticmethod
    def validate_audio(file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Валидация аудиофайла

        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        try:
            # Проверка формата файла
            if not any(file_path.lower().endswith(fmt) for fmt in SUPPORTED_FORMATS):
                return (
                    False,
                    f"Неподдерживаемый формат. Поддерживаемые: {', '.join(SUPPORTED_FORMATS)}",
                )

            # Проверка через soundfile
            with sf.SoundFile(file_path) as f:
                duration = len(f) / f.samplerate

                if duration < MIN_DURATION:
                    return (
                        False,
                        f"Длительность ({duration:.1f}с) меньше минимальной ({MIN_DURATION}с)",
                    )

                if duration > MAX_DURATION:
                    return (
                        False,
                        f"Длительность ({duration:.1f}с) превышает максимальную ({MAX_DURATION}с)",
                    )

                if f.channels not in [1, 2]:
                    return (
                        False,
                        f"Неподдерживаемое количество каналов: {f.channels}. Поддерживаются моно (1) или стерео (2)",
                    )

                if f.samplerate < 8000 or f.samplerate > 48000:
                    return (
                        False,
                        f"Частота дискретизации ({f.samplerate} Hz) вне допустимого диапазона (8-48 kHz)",
                    )

            return True, None

        except Exception as e:
            return False, f"Ошибка при валидации аудиофайла: {str(e)}"

    @staticmethod
    def load_audio(
        path: Union[str, bytes, io.BytesIO],
        target_sr: int = TARGET_SR,
        normalize: bool = True,
        duration: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Загрузка аудиофайла с автоматической конвертацией

        Args:
            path: Путь к файлу или bytes-like объект
            target_sr: Целевая частота дискретизации
            normalize: Нормализовать аудио
            duration: Обрезать до указанной длительности (в секундах)

        Returns:
            Tuple[np.ndarray, int]: (аудиосигнал, частота дискретизации)
        """
        try:
            # Загрузка аудио
            if isinstance(path, (bytes, io.BytesIO)):
                # Загрузка из bytes или BytesIO
                audio, sr = sf.read(
                    io.BytesIO(path if isinstance(path, bytes) else path.getvalue())
                )
            else:
                # Загрузка из файла
                audio, sr = librosa.load(path, sr=None, mono=False)

            # Конвертация в моно если стерео
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=0)
                logger.debug(f"Конвертирован стерео в моно")

            # Проверка длительности
            current_duration = len(audio) / sr
            if current_duration < MIN_DURATION:
                logger.warning(
                    f"Аудио короче минимальной длительности: {current_duration:.1f}с"
                )

            if duration and current_duration > duration:
                # Обрезка до указанной длительности
                samples_to_keep = int(duration * sr)
                audio = audio[:samples_to_keep]
                logger.debug(f"Аудио обрезано до {duration} секунд")

            # Ресемплинг если необходимо
            if sr != target_sr:
                logger.debug(f"Ресемплинг с {sr} Hz до {target_sr} Hz")

                # Используем resampy для высококачественного ресемплинга
                audio = resampy.resample(audio, sr, target_sr)
                sr = target_sr

            # Нормализация
            if normalize:
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    audio = audio / max_val
                    logger.debug("Аудио нормализовано")

            # Удаление тишины в начале и конце
            audio = AudioProcessor.trim_silence(audio, sr)

            return audio, sr

        except Exception as e:
            logger.error(f"Ошибка при загрузке аудио: {str(e)}")
            raise

    @staticmethod
    def trim_silence(audio: np.ndarray, sr: int, top_db: int = 30) -> np.ndarray:
        """Удаление тишины в начале и конце аудио"""
        try:
            trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
            if len(trimmed) > 0:
                logger.debug(f"Удалена тишина: {len(audio) - len(trimmed)} samples")
                return trimmed
            return audio
        except:
            return audio

    @staticmethod
    def preprocess_for_model(
        audio: np.ndarray,
        sr: int,
        target_length: int = 64600,
        padding_mode: str = "constant",
    ) -> np.ndarray:
        """
        Предобработка аудио для модели

        Args:
            audio: Аудиосигнал
            sr: Частота дискретизации
            target_length: Целевая длина
            padding_mode: Режим паддинга ('constant', 'edge', 'reflect')

        Returns:
            np.ndarray: Обработанный аудиосигнал фиксированной длины
        """
        # Обрезка до целевой длины если длиннее
        if len(audio) > target_length:
            audio = audio[:target_length]
            logger.debug(f"Аудио обрезано до {target_length} samples")

        # Паддинг если короче
        if len(audio) < target_length:
            pad_length = target_length - len(audio)

            if padding_mode == "constant":
                audio = np.pad(audio, (0, pad_length), mode="constant")
            elif padding_mode == "edge":
                audio = np.pad(audio, (0, pad_length), mode="edge")
            elif padding_mode == "reflect":
                audio = np.pad(audio, (0, pad_length), mode="reflect")

            logger.debug(f"Добавлен паддинг: {pad_length} samples")

        return audio

    @staticmethod
    def extract_features(
        audio: np.ndarray, sr: int, feature_type: str = "mfcc"
    ) -> np.ndarray:
        """
        Извлечение аудио-фич

        Args:
            audio: Аудиосигнал
            sr: Частота дискретизации
            feature_type: Тип фич ('mfcc', 'melspectrogram', 'spectral')

        Returns:
            np.ndarray: Матрица фич
        """
        if feature_type == "mfcc":
            # MFCC с дельтами и дельта-дельтами
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])

        elif feature_type == "melspectrogram":
            # Mel-спектрограмма
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            features = mel_spec_db

        elif feature_type == "spectral":
            # Спектральные фичи
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

            features = np.vstack(
                [
                    spectral_centroid,
                    spectral_bandwidth,
                    spectral_rolloff,
                    spectral_contrast,
                ]
            )

        else:
            raise ValueError(f"Неизвестный тип фич: {feature_type}")

        return features

    @staticmethod
    def analyze_audio_quality(audio: np.ndarray, sr: int) -> dict:
        """
        Анализ качества аудио

        Returns:
            Dict: Метрики качества аудио
        """
        # SNR оценка
        noise_floor = np.percentile(np.abs(audio), 5)
        signal_power = np.mean(audio**2)
        noise_power = noise_floor**2
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))

        # Динамический диапазон
        dynamic_range = 20 * np.log10(np.max(np.abs(audio)) / (np.std(audio) + 1e-10))

        # Гармонический анализ
        try:
            harmonic, percussive = librosa.effects.hpss(audio)
            harmonic_ratio = np.sum(harmonic**2) / (np.sum(audio**2) + 1e-10)
        except:
            harmonic_ratio = 0.5

        return {
            "snr_db": float(snr),
            "dynamic_range_db": float(dynamic_range),
            "harmonic_ratio": float(harmonic_ratio),
            "duration_seconds": len(audio) / sr,
            "sample_rate": sr,
            "samples_count": len(audio),
            "max_amplitude": float(np.max(np.abs(audio))),
        }

    @staticmethod
    def resample_audio(
        audio_array: np.ndarray, orig_sr: int, target_sr: int = 48000
    ) -> np.ndarray:
        tensor = torch.from_numpy(audio_array).float().unsqueeze(0)
        resampled = torchaudio.transforms.Resample(orig_sr, target_sr)(tensor)
        return resampled.squeeze(0).numpy()

    @staticmethod
    def convert_to_wav(audio_array: np.ndarray, sample_rate: int = 48000) -> bytes:
        """
        Конвертирует float32 numpy array (-1.0..1.0) в WAV 16-bit PCM
        и возвращает байты.
        """
        try:
            # Приведение float32 -> int16
            audio_int16 = np.clip(audio_array * 32767, -32768, 32767).astype(np.int16)

            # Запись в BytesIO через scipy
            buffer = io.BytesIO()
            wav_write(buffer, sample_rate, audio_int16)
            buffer.seek(0)
            return buffer.read()
        except Exception as e:
            logger.error(f"convert_to_wav error: {e}")
            return None

    @staticmethod
    def stream_to_wav(audio_chunks: list[np.ndarray], sr: int) -> bytes:
        """
        Конвертация потока аудио чанков в WAV

        Args:
            audio_chunks: Список аудио чанков
            sr: Частота дискретизации

        Returns:
            bytes: WAV файл в байтах
        """
        # Конкатенация чанков
        full_audio = np.concatenate(audio_chunks)

        # Конвертация в WAV
        return AudioProcessor.convert_to_wav(full_audio, sr)


def load_audio(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """
    Упрощенная функция для загрузки аудио

    Args:
        path: Путь к аудиофайлу
        target_sr: Целевая частота дискретизации

    Returns:
        np.ndarray: Аудиосигнал
    """
    audio, _ = AudioProcessor.load_audio(path, target_sr)
    return audio
