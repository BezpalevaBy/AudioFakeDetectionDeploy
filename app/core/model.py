# app/core/model.py
import torch
import logging
from typing import Dict, Any
from transformers import AutoModelForAudioClassification
from app.model import spectra_0
from pathlib import Path

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_DIR = Path(__file__).parent.parent

    print(f"Загружаем модель из: {MODEL_DIR}")
    print(f"Файл safetensors: {MODEL_DIR / 'model.safetensors'}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = (
        spectra_0.from_pretrained(pretrained_model_name_or_path=str(MODEL_DIR))
        .eval()
        .to(device)
    )
    model.eval()

    return model


def get_model_info(model) -> Dict[str, Any]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "name": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "device": str(next(model.parameters()).device),
        "state": "eval" if not model.training else "train",
    }
