# app/core/report.py
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import base64
from fastapi import Response, logger
import numpy as np


def generate_report(
    classification: str,
    confidence: float,
    artifacts: Dict,
    processing_time: float,
    audio_quality: Dict,
    spectrogram_data: Optional[Dict] = None,
    model_info: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Генерация детализированного отчета анализа (безопасно, всегда возвращает dict)
    """
    try:
        # Основная классификация
        is_fake = classification == "FAKE"
        confidence_percent = round(confidence * 100, 2)

        if confidence_percent >= 85:
            confidence_level = "VERY HIGH"
        elif confidence_percent >= 70:
            confidence_level = "HIGH"
        elif confidence_percent >= 55:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        # Детализация артефактов
        artifact_categories = {
            "spectral_anomalies": "Аномалии в частотном спектре",
            "phoneme_transitions": "Переходы между фонемами",
            "vocoder_artifacts": "Артефакты вокодера",
            "statistical_anomalies": "Статистические аномалии",
        }

        detailed_artifacts = []
        for key, description in artifact_categories.items():
            if key in artifacts and artifacts[key]:
                detailed_artifacts.extend(
                    [f"{description}: {a}" for a in artifacts[key]]
                )

        # Рекомендации
        try:
            recommendations = generate_recommendations(
                is_fake, confidence_percent, artifacts
            )
        except Exception as e:
            recommendations = []
            logger.warning(f"generate_recommendations failed: {str(e)}")

        # Основной JSON
        json_report = {
            "classification": classification,
            "is_fake": is_fake,
            "confidence": confidence,
            "confidence_percent": confidence_percent,
            "confidence_level": confidence_level,
            "analysis": {
                "artifacts": {
                    "detected": detailed_artifacts,
                    "confidence_level": artifacts.get("confidence_level", "LOW"),
                },
                "audio_quality": {
                    "snr_db": audio_quality.get("snr_db", 0),
                    "dynamic_range_db": audio_quality.get("dynamic_range_db", 0),
                    "harmonic_ratio": audio_quality.get("harmonic_ratio", 0),
                    "duration_seconds": audio_quality.get("duration_seconds", 0),
                    "sample_rate": audio_quality.get("sample_rate", 0),
                },
            },
            "processing": {
                "processing_time_seconds": round(processing_time, 3),
                "timestamp": datetime.now().isoformat(),
                "model_info": model_info or {},
            },
            "recommendations": recommendations,
            "metadata": {
                "report_version": "2.0",
                "system": "Deepfake Audio Detection System",
                "threshold": 0.5,
            },
        }

        # Безопасная работа со спектрограммой
        if spectrogram_data:
            json_report["spectrogram"] = {
                "has_data": True,
                "anomaly_count": len(spectrogram_data.get("anomalies", [])),
                "size": {
                    "time_frames": len(spectrogram_data.get("times", [])),
                    "frequency_bins": len(spectrogram_data.get("frequencies", [])),
                },
            }
            json_report["spectrogram_image"] = (
                f"/api/spectrogram/{spectrogram_data.get('spectrogram_id', 0)}"
            )
        else:
            json_report["spectrogram_image"] = None

        # Human readable и visual отчеты с защитой try/except
        try:
            human_readable = generate_human_readable_report(json_report) or {}
        except Exception as e:
            logger.warning(f"generate_human_readable_report failed: {str(e)}")
            human_readable = {}

        try:
            visual_report = generate_visual_report(json_report, spectrogram_data) or {}
        except Exception as e:
            logger.warning(f"generate_visual_report failed: {str(e)}")
            visual_report = {}

        # Комплексный отчет
        comprehensive_report = {
            "api_format": json_report,
            "human_readable": human_readable,
            "visual": visual_report,
            "technical": {
                "classification_score": confidence,
                "artifact_score": calculate_artifact_score(artifacts),
                "quality_score": calculate_quality_score(audio_quality),
                "final_score": calculate_final_score(
                    confidence, artifacts, audio_quality
                ),
            },
        }

        return comprehensive_report

    except Exception as e:
        logger.error(f"generate_report fatal error: {str(e)}")
        # Всегда возвращаем словарь
        return {
            "api_format": {"is_fake": None, "confidence": 0.0},
            "human_readable": {},
            "visual": {},
            "technical": {},
        }


def generate_human_readable_report(json_report: Dict) -> str:
    """Генерация человеко-читаемого текстового отчета"""

    classification = json_report["classification"]
    confidence = json_report["confidence_percent"]
    artifacts = json_report["analysis"]["artifacts"]["detected"]
    recommendations = json_report["recommendations"]
    processing_time = json_report["processing"]["processing_time_seconds"]

    report_lines = [
        "=" * 60,
        "ОТЧЕТ ОБ АНАЛИЗЕ АУДИО НА ПРИЗНАКИ DEEPFAKE",
        "=" * 60,
        f"\n📊 ОСНОВНЫЕ РЕЗУЛЬТАТЫ:",
        f"   Вердикт: {'⚠️ СИНТЕЗИРОВАННОЕ (FAKE) аудио' if classification == 'FAKE' else '✅ НАСТОЯЩЕЕ (REAL) аудио'}",
        f"   Уверенность модели: {confidence}% ({json_report['confidence_level']})",
        f"   Время анализа: {processing_time:.2f} секунд",
    ]

    # Качество аудио
    quality = json_report["analysis"]["audio_quality"]
    report_lines.extend(
        [
            f"\n🎵 КАЧЕСТВО АУДИО:",
            f"   Длительность: {quality['duration_seconds']:.1f} секунд",
            f"   Частота дискретизации: {quality['sample_rate']} Hz",
            f"   SNR: {quality['snr_db']:.1f} dB",
            f"   Динамический диапазон: {quality['dynamic_range_db']:.1f} dB",
        ]
    )

    # Рекомендации
    report_lines.append(f"\n💡 РЕКОМЕНДАЦИИ:")
    for i, recommendation in enumerate(recommendations, 1):
        report_lines.append(f"   {i}. {recommendation}")

    report_lines.extend(
        [
            f"\n📋 ТЕХНИЧЕСКАЯ ИНФОРМАЦИЯ:",
            f"   Модель: {json_report['processing']['model_info'].get('name', 'rawnet_lite')}",
            f"   Порог классификации: {json_report['metadata']['threshold']}",
            f"   Время анализа: {json_report['processing']['timestamp']}",
            f"\n" + "=" * 60,
            f"Система обнаружения аудио-фейков v{json_report['metadata']['report_version']}",
            "=" * 60,
        ]
    )

    return "\n".join(report_lines)


def generate_visual_report(json_report: Dict, spectrogram_data: Optional[Dict]) -> Dict:
    """Генерация данных для визуализации"""

    visual_data = {
        # Данные для графиков
        "charts": {
            "confidence_gauge": {
                "value": json_report["confidence_percent"],
                "max": 100,
                "levels": [
                    {"from": 0, "to": 50, "color": "#dc3545", "label": "Низкая"},
                    {"from": 50, "to": 75, "color": "#ffc107", "label": "Средняя"},
                    {"from": 75, "to": 90, "color": "#28a745", "label": "Высокая"},
                    {
                        "from": 90,
                        "to": 100,
                        "color": "#20c997",
                        "label": "Очень высокая",
                    },
                ],
            },
            "artifact_radar": generate_artifact_radar_data(json_report),
            "quality_metrics": generate_quality_metrics_data(json_report),
        },
        # Данные для спектрограммы
        "spectrogram": (
            spectrogram_data
            if spectrogram_data
            else {"has_data": False, "message": "Спектрограмма не сгенерирована"}
        ),
        # Статусные индикаторы
        "indicators": {
            "classification": {
                "status": "danger" if json_report["is_fake"] else "success",
                "icon": "⚠️" if json_report["is_fake"] else "✅",
                "text": "FAKE" if json_report["is_fake"] else "REAL",
            },
            "confidence": {
                "status": (
                    "success"
                    if json_report["confidence_percent"] > 70
                    else (
                        "warning"
                        if json_report["confidence_percent"] > 50
                        else "danger"
                    )
                ),
                "level": json_report["confidence_level"],
            },
            "quality": {
                "status": (
                    "success"
                    if json_report["analysis"]["audio_quality"]["snr_db"] > 20
                    else (
                        "warning"
                        if json_report["analysis"]["audio_quality"]["snr_db"] > 10
                        else "danger"
                    )
                ),
                "rating": (
                    "Высокое"
                    if json_report["analysis"]["audio_quality"]["snr_db"] > 20
                    else (
                        "Среднее"
                        if json_report["analysis"]["audio_quality"]["snr_db"] > 10
                        else "Низкое"
                    )
                ),
            },
        },
    }

    return visual_data


def generate_recommendations(
    is_fake: bool, confidence: float, artifacts: Dict
) -> List[str]:
    """Генерация рекомендаций на основе результатов анализа"""

    recommendations = []

    if is_fake:
        if confidence > 80:
            recommendations.extend(
                [
                    "Высокая вероятность синтезированного аудио. Рекомендуется полная верификация источника.",
                    "Провести анализ дополнительными методами (спектральный, статистический).",
                    "Сравнить с другими записями того же говорящего при наличии.",
                    "Проверить метаданные файла на наличие несоответствий.",
                    "Рассмотреть возможность использования профессиональных инструментов экспертизы.",
                ]
            )
        elif confidence > 60:
            recommendations.extend(
                [
                    "Умеренная вероятность синтезированного аудио. Требуется дополнительная проверка.",
                    "Использовать несколько независимых систем детекции для подтверждения.",
                    "Анализировать контекст записи и обстоятельства получения.",
                    "Проверить историю файла и его происхождение.",
                    "Рассмотреть возможность человеческой экспертизы.",
                ]
            )
        else:
            recommendations.extend(
                [
                    "Низкая уверенность в результатах. Требуется углубленный анализ.",
                    "Использовать расширенные методы спектрального анализа.",
                    "Проверить качество исходной записи (шум, артефакты сжатия).",
                    "Рассмотреть возможность ложного срабатывания системы.",
                    "Повторить анализ с другими настройками модели.",
                ]
            )
    else:
        if confidence > 80:
            recommendations.extend(
                [
                    "Высокая вероятность подлинности аудио. Дополнительная проверка может не потребоваться.",
                    "Для максимальной уверенности можно провести экспресс-проверку альтернативными методами.",
                    "Убедиться в надежности источника записи.",
                    "Сохранить метаданные файла для возможной будущей верификации.",
                ]
            )
        elif confidence > 60:
            recommendations.extend(
                [
                    "Умеренная уверенность в подлинности. Рекомендуется базовая проверка.",
                    "Проверить целостность файла и отсутствие модификаций.",
                    "Убедиться в соответствии технических характеристик заявленным.",
                    "Рассмотреть возможность дополнительной проверки при критичности задачи.",
                ]
            )
        else:
            recommendations.extend(
                [
                    "Низкая уверенность в результатах. Рекомендуется полная проверка.",
                    "Использовать несколько методов детекции для получения консенсуса.",
                    "Провести экспертизу качества записи и технических параметров.",
                    "Анализировать запись в контексте других доступных данных.",
                    "Рассмотреть возможность проведения профессиональной аудио-экспертизы.",
                ]
            )

    # Дополнительные рекомендации на основе артефактов
    if "spectral_anomalies" in artifacts and artifacts["spectral_anomalies"]:
        recommendations.append(
            "Обнаружены спектральные аномалии - рекомендован спектральный анализ."
        )

    if "vocoder_artifacts" in artifacts and artifacts["vocoder_artifacts"]:
        recommendations.append(
            "Найдены артефакты вокодера - проверьте использование TTS систем."
        )

    return recommendations


def generate_artifact_radar_data(json_report: Dict) -> Dict:
    """Генерация данных для радар-диаграммы артефактов"""

    artifacts = json_report["analysis"]["artifacts"]["detected"]

    # Категории артефактов
    categories = {
        "spectral": ["спектр", "частот", "гармоник"],
        "temporal": ["переход", "время", "темп"],
        "vocoder": ["вокодер", "фаза", "квант"],
        "statistical": ["статистик", "распредел", "асимметр"],
        "quality": ["шум", "качество", "snr"],
    }

    scores = {}
    for category, keywords in categories.items():
        score = 0
        for artifact in artifacts:
            if any(keyword in artifact.lower() for keyword in keywords):
                score += 1
        scores[category] = min(score * 20, 100)  # Нормализация до 100

    return {
        "categories": [
            "Спектральные",
            "Временные",
            "Вокодер",
            "Статистические",
            "Качество",
        ],
        "scores": list(scores.values()),
        "max_score": 100,
    }


def generate_quality_metrics_data(json_report: Dict) -> Dict:
    """Генерация данных для отображения метрик качества"""

    quality = json_report["analysis"]["audio_quality"]

    return {
        "snr": {
            "value": quality["snr_db"],
            "optimal": ">20 dB",
            "status": (
                "good"
                if quality["snr_db"] > 20
                else "acceptable" if quality["snr_db"] > 10 else "poor"
            ),
        },
        "dynamic_range": {
            "value": quality["dynamic_range_db"],
            "optimal": ">40 dB",
            "status": (
                "good"
                if quality["dynamic_range_db"] > 40
                else "acceptable" if quality["dynamic_range_db"] > 20 else "poor"
            ),
        },
        "harmonic_ratio": {
            "value": quality["harmonic_ratio"] * 100,
            "optimal": ">60%",
            "status": (
                "good"
                if quality["harmonic_ratio"] > 0.6
                else "acceptable" if quality["harmonic_ratio"] > 0.3 else "poor"
            ),
        },
    }


def calculate_artifact_score(artifacts: Dict) -> float:
    """Расчет оценки на основе обнаруженных артефактов"""

    if not artifacts:
        return 0.0

    # Веса разных типов артефактов
    weights = {
        "spectral_anomalies": 0.3,
        "vocoder_artifacts": 0.4,
        "phoneme_transitions": 0.2,
        "statistical_anomalies": 0.1,
    }

    score = 0.0
    total_weight = 0.0

    for artifact_type, weight in weights.items():
        if artifact_type in artifacts and artifacts[artifact_type]:
            score += weight * len(artifacts[artifact_type])
            total_weight += weight

    if total_weight > 0:
        return min(score / total_weight * 10, 1.0)

    return 0.0


def calculate_quality_score(audio_quality: Dict) -> float:
    """Расчет оценки качества аудио"""

    # Нормализованные метрики качества
    snr_score = min(audio_quality.get("snr_db", 0) / 30, 1.0)
    dynamic_range_score = min(audio_quality.get("dynamic_range_db", 0) / 50, 1.0)
    harmonic_score = audio_quality.get("harmonic_ratio", 0)

    # Композитная оценка
    quality_score = snr_score * 0.4 + dynamic_range_score * 0.3 + harmonic_score * 0.3

    return quality_score


def calculate_final_score(
    confidence: float, artifacts: Dict, audio_quality: Dict
) -> float:
    """Расчет финальной комплексной оценки"""

    artifact_score = calculate_artifact_score(artifacts)
    quality_score = calculate_quality_score(audio_quality)

    # Если качество аудио низкое, снижаем общую уверенность
    quality_adjustment = 1.0 if quality_score > 0.5 else quality_score

    final_score = confidence * artifact_score * quality_adjustment

    return min(final_score, 1.0)


def export_report(report: Dict, format: str = "json") -> str:
    """
    Экспорт отчета в различных форматах

    Args:
        report: Отчет
        format: Формат экспорта ('json', 'text', 'html')

    Returns:
        str: Отчет в запрошенном формате
    """

    if format == "json":
        return Response(
            content=json.dumps(report, ensure_ascii=False, indent=2),
            media_type="application/json",
        )

    elif format == "text":
        return report["human_readable"]

    elif format == "html":
        # Базовая HTML версия отчета
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Анализ аудио на Deepfake</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .result {{ padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .fake {{ background-color: #f8d7da; color: #721c24; }}
                .real {{ background-color: #d4edda; color: #155724; }}
                .section {{ margin: 20px 0; }}
                .artifact {{ margin: 5px 0; padding: 5px; background-color: #e9ecef; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Отчет анализа аудио на Deepfake</h1>
                <p>Время анализа: {report['api_format']['processing']['timestamp']}</p>
            </div>
            
            <div class="result {'fake' if report['api_format']['is_fake'] else 'real'}">
                <h2>Результат: {report['api_format']['classification']}</h2>
                <p>Уверенность: {report['api_format']['confidence_percent']}%</p>
                <p>Уровень уверенности: {report['api_format']['confidence_level']}</p>
            </div>
            
            <div class="section">
                <h3>Обнаруженные артефакты:</h3>
                {''.join(f'<div class="artifact">{artifact}</div>' 
                        for artifact in report['api_format']['analysis']['artifacts']['detected']) 
                        if report['api_format']['analysis']['artifacts']['detected'] 
                        else '<p>Значительные артефакты не обнаружены</p>'}
            </div>
            
            <div class="section">
                <h3>Рекомендации:</h3>
                <ul>
                    {''.join(f'<li>{rec}</li>' for rec in report['api_format']['recommendations'])}
                </ul>
            </div>
        </body>
        </html>
        """
        return html_report

    else:
        raise ValueError(f"Unsupported format: {format}")
