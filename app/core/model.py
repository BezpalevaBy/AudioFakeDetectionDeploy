# app/core/model.py
import torch
import logging
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# Импортируем оригинальный RawNetLite из репозитория
from app.rawnet.rawnet_lite import RawNetLite


def load_model(
    weights_path: Optional[str] = None, device: torch.device = DEVICE
) -> RawNetLite:
    """
    Загрузка модели RawNetLite с весами из .pt

    Args:
        weights_path: путь к файлу весов rawnet_lite.pt
        device: CPU или GPU

    Returns:
        model: загруженная модель
    """
    model = RawNetLite()
    model.to(device)

    if weights_path and os.path.exists(weights_path):
        try:
            logger.info(f"Loading weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location=device)

            # Если это чекпоинт с state_dict внутри
            if isinstance(state_dict, dict):
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]

            # Загружаем веса
            model.load_state_dict(state_dict)
            logger.info("✅ Weights loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            logger.info("Using randomly initialized weights")
    else:
        logger.info(
            "No weights path provided or file does not exist, using random weights"
        )

    model.eval()
    return model


def get_model_info(model: RawNetLite) -> Dict[str, Any]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "name": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "device": str(next(model.parameters()).device),
        "state": "eval" if not model.training else "train",
    }
