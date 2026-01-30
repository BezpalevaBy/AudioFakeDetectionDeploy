# app/core/analysis.py
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
import scipy.signal as signal
from scipy.stats import kurtosis, skew
import logging
import tempfile
import os

logger = logging.getLogger(__name__)

import tempfile
import os
import uuid

SPECTROGRAM_DIR = os.path.join(tempfile.gettempdir(), "spectrograms")
os.makedirs(SPECTROGRAM_DIR, exist_ok=True)

output_path = None


def analyze_artifacts(prob: float, audio: np.ndarray, sr: int = 16000) -> Dict:
    """
    Расширенный анализ артефактов в аудио

    Args:
        prob: Вероятность подлинности (0-1)
        audio: Аудиосигнал
        sr: Частота дискретизации

    Returns:
        Словарь с детализированными результатами анализа
    """
    findings = {
        "spectral_anomalies": [],
        "phoneme_transitions": [],
        "vocoder_artifacts": [],
        "statistical_anomalies": [],
        "confidence_level": "LOW",
    }

    # Уровень уверенности
    if prob > 0.8:
        findings["confidence_level"] = "HIGH"
    elif prob > 0.6:
        findings["confidence_level"] = "MEDIUM"
    else:
        findings["confidence_level"] = "LOW"

    # 1. Анализ спектральных аномалий
    spectral_features = analyze_spectral_features(audio, sr)

    if spectral_features["harmonic_noise_ratio"] < 0.1:
        findings["spectral_anomalies"].append("Низкое соотношение гармоник/шум")

    if spectral_features["spectral_flatness_mean"] > 0.8:
        findings["spectral_anomalies"].append(
            "Высокая спектральная плоскостность (возможный шум)"
        )

    if spectral_features["spectral_centroid_std"] > 500:
        findings["spectral_anomalies"].append("Нестабильный спектральный центроид")

    # 2. Анализ переходов между фонемами
    phoneme_features = analyze_phoneme_transitions(audio, sr)

    if phoneme_features["transition_smoothness"] < 0.3:
        findings["phoneme_transitions"].append("Резкие переходы между звуками")

    if phoneme_features["formant_stability"] < 0.7:
        findings["phoneme_transitions"].append("Нестабильные форманты")

    # 3. Анализ артефактов вокодера
    vocoder_features = analyze_vocoder_artifacts(audio, sr)

    if vocoder_features["phase_coherence"] < 0.6:
        findings["vocoder_artifacts"].append("Низкая когерентность фазы")

    if vocoder_features["quantization_noise"] > 0.05:
        findings["vocoder_artifacts"].append("Шум квантования обнаружен")

    if vocoder_features["robotic_effect"] > 0.3:
        findings["vocoder_artifacts"].append("Признаки роботизированного голоса")

    # 4. Статистический анализ
    stats = analyze_statistical_features(audio)

    if stats["kurtosis"] > 5:
        findings["statistical_anomalies"].append(
            "Высокий эксцесс (островершинное распределение)"
        )

    if abs(stats["skewness"]) > 1:
        findings["statistical_anomalies"].append(
            f"Асимметрия распределения: {stats['skewness']:.2f}"
        )

    # 5. Дополнительные проверки для high-confidence фейков
    if prob > 0.7:
        # Проверка на артефакты перехода
        transition_artifacts = check_transition_artifacts(audio, sr)
        findings["vocoder_artifacts"].extend(transition_artifacts)

        # Проверка спектральной целостности
        spectral_integrity = check_spectral_integrity(audio, sr)
        if not spectral_integrity:
            findings["spectral_anomalies"].append("Нарушена спектральная целостность")

    # Удаление пустых категорий
    for key in list(findings.keys()):
        if isinstance(findings[key], list) and len(findings[key]) == 0:
            findings[f"no_{key.replace('_', ' ')}"] = True
            del findings[key]

    return findings


def analyze_spectral_features(audio: np.ndarray, sr: int) -> Dict:
    """Анализ спектральных характеристик"""
    # Вычисление спектрограммы
    n_fft = 2048
    hop_length = 512

    # STFT
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)

    # Спектральный центроид
    spectral_centroid = librosa.feature.spectral_centroid(
        S=magnitude, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]

    # Спектральная плоскостность
    spectral_flatness = librosa.feature.spectral_flatness(
        S=magnitude, n_fft=n_fft, hop_length=hop_length
    )[0]

    # Спектральная полоса пропускания
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        S=magnitude, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]

    # Гармонический анализ
    harmonic = librosa.effects.harmonic(audio)
    percussive = librosa.effects.percussive(audio)
    harmonic_noise_ratio = np.mean(np.abs(harmonic)) / (
        np.mean(np.abs(percussive)) + 1e-10
    )

    return {
        "spectral_centroid_mean": float(np.mean(spectral_centroid)),
        "spectral_centroid_std": float(np.std(spectral_centroid)),
        "spectral_flatness_mean": float(np.mean(spectral_flatness)),
        "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
        "harmonic_noise_ratio": float(harmonic_noise_ratio),
    }


def analyze_phoneme_transitions(audio: np.ndarray, sr: int) -> Dict:
    """Анализ плавности переходов между фонемами"""
    # Вычисление MFCC для анализа фонем
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    # Анализ изменений MFCC (градиенты)
    mfcc_gradients = np.gradient(mfcc, axis=1)
    gradient_magnitudes = np.sqrt(np.sum(mfcc_gradients**2, axis=0))

    # Плавность переходов (чем меньше градиент, тем плавнее)
    transition_smoothness = 1.0 / (1.0 + np.mean(gradient_magnitudes))

    # Анализ формант (через LPC)
    try:
        from scipy.signal import lfilter

        # Линейное предсказание для анализа формант
        order = 12
        a = librosa.lpc(audio, order=order)

        # Нахождение корней полинома
        roots = np.roots(a)
        roots = roots[np.imag(roots) >= 0]

        # Вычисление частот формант
        angs = np.arctan2(np.imag(roots), np.real(roots))
        formant_freqs = angs * (sr / (2 * np.pi))

        # Стабильность формант
        formant_stability = (
            1.0 - (np.std(formant_freqs) / np.mean(formant_freqs))
            if len(formant_freqs) > 0
            else 0.5
        )
    except:
        formant_stability = 0.5

    return {
        "transition_smoothness": float(transition_smoothness),
        "formant_stability": float(formant_stability),
        "mfcc_variance": float(np.var(mfcc)),
    }


def analyze_vocoder_artifacts(audio: np.ndarray, sr: int) -> Dict:
    """Анализ артефактов, характерных для вокодеров"""
    # Анализ фазовой когерентности
    n_fft = 2048
    hop_length = 512

    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    phase = np.angle(D)

    # Когерентность фазы между соседними кадрами
    phase_diff = np.diff(phase, axis=1)
    phase_coherence = np.exp(-np.mean(np.abs(phase_diff)))

    # Поиск характерных для вокодера паттернов (периодичность в ошибках)
    # Анализ остаточного сигнала
    try:
        # Линейное предсказание
        order = 16
        a = librosa.lpc(audio, order=order)

        # Остаточный сигнал
        residual = lfilter(a, 1, audio)

        # Анализ остатков на предмет артефактов квантования
        residual_autocorr = np.correlate(residual, residual, mode="full")
        residual_autocorr = residual_autocorr[len(residual_autocorr) // 2 :]

        # Поиск периодичности в автокорреляции
        peaks, _ = signal.find_peaks(residual_autocorr[:1000], height=0.1)
        quantization_noise = len(peaks) / 1000.0 if len(peaks) > 0 else 0.0
    except:
        quantization_noise = 0.0

    # Проверка на "роботизированный" эффект (чрезмерная стабильность)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spectral_centroid_variation = np.std(spectral_centroid) / np.mean(spectral_centroid)
    robotic_effect = max(0, 1.0 - spectral_centroid_variation)

    return {
        "phase_coherence": float(phase_coherence),
        "quantization_noise": float(quantization_noise),
        "robotic_effect": float(robotic_effect),
    }


def analyze_statistical_features(audio: np.ndarray) -> Dict:
    """Статистический анализ аудиосигнала"""
    return {
        "mean": float(np.mean(audio)),
        "std": float(np.std(audio)),
        "kurtosis": float(kurtosis(audio)),
        "skewness": float(skew(audio)),
        "dynamic_range": float(np.max(np.abs(audio)) - np.min(np.abs(audio))),
    }


def check_transition_artifacts(audio: np.ndarray, sr: int) -> List[str]:
    """Проверка артефактов на переходах"""
    artifacts = []

    # Энергия сигнала
    energy = librosa.feature.rms(y=audio)[0]
    energy_gradient = np.gradient(energy)

    # Резкие скачки энергии
    sharp_transitions = np.where(np.abs(energy_gradient) > np.std(energy_gradient) * 3)[
        0
    ]
    if len(sharp_transitions) > len(energy) * 0.1:  # Более 10% резких переходов
        artifacts.append("Частые резкие скачки энергии")

    # Проверка на клиппинг
    clipping_ratio = np.sum(np.abs(audio) > 0.95) / len(audio)
    if clipping_ratio > 0.01:  # Более 1% клиппинга
        artifacts.append("Обнаружено клиппирование сигнала")

    return artifacts


def check_spectral_integrity(audio: np.ndarray, sr: int) -> bool:
    """Проверка спектральной целостности сигнала"""
    n_fft = 2048
    hop_length = 512

    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)

    # Проверка на наличие "дыр" в спектре
    spectral_gaps = np.sum(magnitude < 1e-6, axis=0) / magnitude.shape[0]

    # Если в более чем 20% кадров есть значительные спектральные провалы
    if np.mean(spectral_gaps > 0.3) > 0.2:
        return False

    return True


import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import uuid
import base64


def generate_spectrogram_data(audio: np.ndarray, sr: int, id: uuid) -> dict:
    n_fft = 2048
    hop_length = 512

    # STFT
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)

    # Log scale
    log_magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)

    # Временная и частотная оси
    times = librosa.frames_to_time(
        np.arange(log_magnitude.shape[1]), sr=sr, hop_length=hop_length
    )
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Генерация PNG спектрограммы
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(
        log_magnitude,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        cmap="magma",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.tight_layout()

    # Создаём уникальный временный файл
    output_path = os.path.join(tempfile.gettempdir(), f"spectrogram_{id}.png")
    plt.savefig(output_path, dpi=200)
    plt.close()

    # Читаем PNG и конвертируем в Base64
    with open(output_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Можно удалить временный файл
    os.unlink(output_path)

    # Возвращаем данные спектрограммы + Base64
    return {
        "spectrogram": log_magnitude.tolist(),
        "frequencies": freqs.tolist(),
        "times": times.tolist(),
        "sr": sr,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "spectrogram_id": id,
        "spectrogram_image": image_data,  # Base64 для фронта
    }
