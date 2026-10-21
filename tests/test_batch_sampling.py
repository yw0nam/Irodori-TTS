from __future__ import annotations

import pytest
import torch

from irodori_tts.meanflow import sample_euler_meanflow
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.rf import draw_initial_noise, sample_euler_rf_cfg

# (text_len, ref_len, patched_steps, seed) per request.
REQUESTS = [(3, 4, 5, 11), (6, 7, 9, 22)]


def request_inputs(model: TextToLatentRFDiT) -> list[tuple[torch.Tensor, torch.Tensor, int, int]]:
    gen = torch.Generator().manual_seed(1)
    return [
        (
            torch.randint(1, model.cfg.text_vocab_size, (1, text_len), generator=gen),
            torch.randn((1, ref_len, model.cfg.patched_latent_dim), generator=gen),
            patched_steps,
            seed,
        )
        for text_len, ref_len, patched_steps, seed in REQUESTS
    ]


def right_pad(x: torch.Tensor, length: int) -> torch.Tensor:
    pad = x.new_zeros((x.shape[0], length - x.shape[1], *x.shape[2:]))
    return torch.cat([x, pad], dim=1)


def valid_mask(lengths: list[int], width: int) -> torch.Tensor:
    return torch.arange(width)[None, :] < torch.tensor(lengths)[:, None]


def run_single_and_batched(model: TextToLatentRFDiT, sampler, **kwargs):
    inputs = request_inputs(model)
    singles = [
        sampler(
            model=model,
            text_input_ids=text_ids,
            text_mask=torch.ones_like(text_ids, dtype=torch.bool),
            ref_latent=ref,
            ref_mask=torch.ones(ref.shape[:2], dtype=torch.bool),
            sequence_length=steps,
            seed=seed,
            **kwargs,
        )
        for text_ids, ref, steps, seed in inputs
    ]

    text_lens = [text_ids.shape[1] for text_ids, _, _, _ in inputs]
    ref_lens = [ref.shape[1] for _, ref, _, _ in inputs]
    steps = [s for _, _, s, _ in inputs]
    max_steps = max(steps)
    noise = torch.cat(
        [
            right_pad(
                draw_initial_noise(
                    seed,
                    (1, s, model.cfg.patched_latent_dim),
                    device=model.device,
                    dtype=model.dtype,
                )[0],
                max_steps,
            )
            for _, _, s, seed in inputs
        ]
    )
    batched_inputs = dict(
        model=model,
        text_input_ids=torch.cat([right_pad(t, max(text_lens)) for t, _, _, _ in inputs]),
        text_mask=valid_mask(text_lens, max(text_lens)),
        ref_latent=torch.cat([right_pad(r, max(ref_lens)) for _, r, _, _ in inputs]),
        ref_mask=valid_mask(ref_lens, max(ref_lens)),
        sequence_length=max_steps,
        initial_noise=noise,
        **kwargs,
    )
    batched = sampler(latent_mask=valid_mask(steps, max_steps), **batched_inputs)
    unmasked = sampler(latent_mask=None, **batched_inputs)
    return singles, steps, batched, unmasked


def assert_rows_match_singles(singles, steps, batched, unmasked) -> None:
    for row, (single, s) in enumerate(zip(singles, steps, strict=True)):
        torch.testing.assert_close(batched[row : row + 1, :s], single)
    # Without the mask the shorter row attends to padding, so the test can see a leak.
    with pytest.raises(AssertionError):
        torch.testing.assert_close(unmasked[0:1, : steps[0]], singles[0])


def test_rf_batch_rows_match_single_runs(tiny_model) -> None:
    results = run_single_and_batched(
        tiny_model(),
        sample_euler_rf_cfg,
        num_steps=6,
        cfg_scale_text=3.0,
        cfg_scale_speaker=5.0,
        cfg_guidance_mode="independent",
    )
    assert_rows_match_singles(*results)


def test_meanflow_batch_rows_match_single_runs(tiny_model) -> None:
    results = run_single_and_batched(tiny_model("meanflow"), sample_euler_meanflow, num_steps=4)
    assert_rows_match_singles(*results)


def test_rf_initial_noise_rejects_speaker_noise_uncond(tiny_model) -> None:
    model = tiny_model()
    text_ids, ref, steps, _ = request_inputs(model)[0]
    with pytest.raises(ValueError, match="noise"):
        sample_euler_rf_cfg(
            model=model,
            text_input_ids=text_ids,
            text_mask=torch.ones_like(text_ids, dtype=torch.bool),
            ref_latent=ref,
            ref_mask=torch.ones(ref.shape[:2], dtype=torch.bool),
            sequence_length=steps,
            speaker_uncond_mode="noise",
            initial_noise=torch.zeros((1, steps, model.cfg.patched_latent_dim)),
        )
