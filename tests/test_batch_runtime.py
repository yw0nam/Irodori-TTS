from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    batch_compatibility_key,
)
from irodori_tts.watermark import SilentCipherWatermarker

HOP_LENGTH = 8


class FakeTokenizer:
    def batch_encode(self, texts: list[str], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.zeros((len(texts), max_length), dtype=torch.long)
        mask = torch.zeros((len(texts), max_length), dtype=torch.bool)
        for row, text in enumerate(texts):
            tokens = [ord(ch) % 49 + 1 for ch in text][:max_length]
            ids[row, : len(tokens)] = torch.tensor(tokens)
            mask[row, : len(tokens)] = True
        return ids, mask


class FakeCodec:
    sample_rate = 80
    model = SimpleNamespace(hop_length=HOP_LENGTH)

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        return z.mean(-1).repeat_interleave(HOP_LENGTH, dim=-1).unsqueeze(1)


@pytest.fixture
def runtime(tiny_model, monkeypatch) -> InferenceRuntime:
    # Keep the test offline: the watermarker reports not-ready instead of loading SilentCipher.
    monkeypatch.setattr(SilentCipherWatermarker, "_load_backend", staticmethod(lambda **_: None))
    model = tiny_model()
    return InferenceRuntime(
        key=RuntimeKey(checkpoint="tiny", model_device="cpu"),
        model_cfg=model.cfg,
        train_cfg=None,
        model=model,
        tokenizer=FakeTokenizer(),
        caption_tokenizer=None,
        codec=FakeCodec(),
        default_text_max_len=16,
        default_caption_max_len=16,
    )


@pytest.fixture
def two_requests(runtime, tmp_path) -> list[SamplingRequest]:
    gen = torch.Generator().manual_seed(3)
    paths = []
    for name, steps in (("short", 6), ("long", 11)):
        path = tmp_path / f"{name}.pt"
        torch.save(torch.randn((steps, runtime.model_cfg.latent_dim), generator=gen), path)
        paths.append(str(path))
    # 0.5 s and 0.9 s at 80 Hz / hop 8 give 5 and 9 latent steps.
    return [
        SamplingRequest(
            text="hi", ref_latent=paths[0], num_candidates=2, seconds=0.5, seed=1, num_steps=4
        ),
        SamplingRequest(
            text="a longer line", ref_latent=paths[1], seconds=0.9, seed=2, num_steps=4
        ),
    ]


def test_synthesize_batch_matches_single_requests(runtime, two_requests) -> None:
    batched = runtime.synthesize_batch(two_requests)
    singles = [runtime.synthesize(req) for req in two_requests]

    assert len(batched) == len(singles)
    for got, want in zip(batched, singles, strict=True):
        assert got.used_seed == want.used_seed
        assert [a.shape for a in got.audios] == [a.shape for a in want.audios]
        for got_audio, want_audio in zip(got.audios, want.audios, strict=True):
            torch.testing.assert_close(got_audio, want_audio)


def test_synthesize_batch_rejects_incompatible_requests(runtime, two_requests) -> None:
    with pytest.raises(ValueError, match="1"):
        runtime.synthesize_batch([two_requests[0], replace(two_requests[1], num_steps=8)])


def test_batch_compatibility_key() -> None:
    base = SamplingRequest(text="a", ref_wav="a.wav", seed=1, seconds=1.0)

    per_request_only = replace(base, text="b", ref_wav="b.wav", seed=2, seconds=2.0)
    assert batch_compatibility_key(base) == batch_compatibility_key(per_request_only)
    assert batch_compatibility_key(base) != batch_compatibility_key(replace(base, num_steps=8))
    assert batch_compatibility_key(base) != batch_compatibility_key(
        replace(base, ref_wav=None, no_ref=True)
    )
    assert batch_compatibility_key(replace(base, speaker_uncond_mode="noise")) is None
