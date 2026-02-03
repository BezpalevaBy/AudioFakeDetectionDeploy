# app/core/model.py
import torch
import logging
from typing import Dict, Any
from transformers import AutoModelForAudioClassification

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


def load_model():
    model_name = "garystafford/wav2vec2-deepfake-voice-detector"
    model = AutoModelForAudioClassification.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
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
