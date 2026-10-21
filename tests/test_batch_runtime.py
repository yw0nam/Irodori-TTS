from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import soundfile as sf
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
    latent_dim = 4

    def __init__(self) -> None:
        self.encode_calls = 0

    def encode_waveform(self, wav: torch.Tensor, **_) -> torch.Tensor:
        self.encode_calls += 1
        steps = wav.shape[-1] // HOP_LENGTH
        frames = wav[0, 0, : steps * HOP_LENGTH].reshape(steps, HOP_LENGTH).mean(-1)
        return frames[None, :, None].repeat(1, 1, self.latent_dim)

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        return z.mean(-1).repeat_interleave(HOP_LENGTH, dim=-1).unsqueeze(1)


@pytest.fixture
def make_runtime(tiny_model, monkeypatch):
    # Keep the test offline: the watermarker reports not-ready instead of loading SilentCipher.
    monkeypatch.setattr(SilentCipherWatermarker, "_load_backend", staticmethod(lambda **_: None))

    def make(**model_overrides) -> InferenceRuntime:
        model = tiny_model(**model_overrides)
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

    return make


@pytest.fixture
def runtime(make_runtime) -> InferenceRuntime:
    return make_runtime()


def write_ref_latents(runtime: InferenceRuntime, tmp_path, lengths: list[int]) -> list[str]:
    gen = torch.Generator().manual_seed(3)
    paths = []
    for index, steps in enumerate(lengths):
        path = tmp_path / f"ref{index}.pt"
        torch.save(torch.randn((steps, runtime.model_cfg.latent_dim), generator=gen), path)
        paths.append(str(path))
    return paths


@pytest.fixture
def two_requests(runtime, tmp_path) -> list[SamplingRequest]:
    paths = write_ref_latents(runtime, tmp_path, [6, 11])
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
    with pytest.raises(ValueError, match=r"Request 1 cannot share"):
        runtime.synthesize_batch([two_requests[0], replace(two_requests[1], num_steps=8)])
    with pytest.raises(ValueError, match=r"Request 0 cannot run"):
        runtime.synthesize_batch(
            [replace(two_requests[0], speaker_uncond_mode="noise"), two_requests[1]]
        )


def test_synthesize_batch_rejects_reference_shorter_than_speaker_patch(
    make_runtime, tmp_path
) -> None:
    runtime = make_runtime(speaker_patch_size=4)
    requests = [
        SamplingRequest(text="hi", ref_latent=path, seconds=0.5, seed=1, num_steps=4)
        for path in write_ref_latents(runtime, tmp_path, [2, 13])
    ]
    with pytest.raises(ValueError, match="Reference sequence too short for speaker_patch_size"):
        runtime.synthesize_batch(requests)


def test_batch_compatibility_key() -> None:
    base = SamplingRequest(text="a", ref_wav="a.wav", seed=1, seconds=1.0)

    per_request_only = replace(base, text="b", ref_wav="b.wav", seed=2, seconds=2.0)
    assert batch_compatibility_key(base) == batch_compatibility_key(per_request_only)
    assert batch_compatibility_key(base) != batch_compatibility_key(replace(base, num_steps=8))
    assert batch_compatibility_key(base) != batch_compatibility_key(
        replace(base, ref_wav=None, no_ref=True)
    )
    assert batch_compatibility_key(replace(base, speaker_uncond_mode="noise")) is None


def write_ref_wav(path, seed: int) -> str:
    audio = torch.randn(80, generator=torch.Generator().manual_seed(seed)).numpy() * 0.1
    sf.write(path, audio, 80)
    return str(path)


def test_reference_wavs_are_encoded_once_per_unchanged_file(runtime, tmp_path) -> None:
    paths = [write_ref_wav(tmp_path / f"ref{i}.wav", seed=i) for i in range(2)]
    req = SamplingRequest(text="hi", ref_wavs=paths, seconds=0.5, seed=1, num_steps=4)

    first = runtime.synthesize(req)
    second = runtime.synthesize(req)

    assert runtime.codec.encode_calls == 2
    assert second.messages == first.messages
    torch.testing.assert_close(second.audios[0], first.audios[0])

    write_ref_wav(tmp_path / "ref1.wav", seed=9)
    third = runtime.synthesize(req)

    assert runtime.codec.encode_calls == 4
    assert not torch.equal(third.audios[0], first.audios[0])
