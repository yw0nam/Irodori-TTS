"""fast-bunkai wrapper exposed via the HTTP API for streaming-TTS clients."""

from __future__ import annotations

from functools import lru_cache

from fast_bunkai import FastBunkai


@lru_cache(maxsize=1)
def _splitter() -> FastBunkai:
    return FastBunkai()


def find_eos(text: str) -> list[int]:
    if not text or not text.strip():
        return []
    return list(_splitter().find_eos(text))


def split_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    return list(_splitter()(text))


def find_eos_and_split(text: str) -> tuple[list[int], list[str]]:
    if not text or not text.strip():
        return [], []
    sentences = list(_splitter()(text))
    positions: list[int] = []
    cursor = 0
    for sentence in sentences:
        cursor += len(sentence)
        positions.append(cursor)
    return positions, sentences


def warmup() -> None:
    _splitter().find_eos("ウォームアップ。")
