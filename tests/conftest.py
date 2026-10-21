from __future__ import annotations

import pytest
import torch

from irodori_tts.config import ModelConfig
from irodori_tts.model import TextToLatentRFDiT


def _build_tiny_model(flow_parameterization: str = "rf_velocity") -> TextToLatentRFDiT:
    cfg = ModelConfig(
        flow_parameterization=flow_parameterization,
        latent_dim=4,
        model_dim=32,
        num_layers=2,
        num_heads=2,
        mlp_ratio=2.0,
        text_mlp_ratio=2.0,
        speaker_mlp_ratio=2.0,
        text_vocab_size=50,
        text_dim=32,
        text_layers=1,
        text_heads=2,
        use_speaker_condition=True,
        speaker_dim=32,
        speaker_layers=1,
        speaker_heads=2,
        timestep_embed_dim=16,
        adaln_rank=8,
    )
    torch.manual_seed(0)
    model = TextToLatentRFDiT(cfg)
    # The model zero-initializes its output and gate projections; non-zero weights make leaks visible.
    for param in model.parameters():
        torch.nn.init.normal_(param, std=0.2)
    return model.eval()


@pytest.fixture
def tiny_model():
    return _build_tiny_model
