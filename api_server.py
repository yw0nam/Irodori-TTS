#!/usr/bin/env python3
"""
FastAPI server for Irodori-TTS — multi-worker, async.

Endpoint:
  POST /synthesize
    - text              : desired synthesis text
    - reference_audio   : audio file (wav/mp3/flac/…) for voice cloning (optional;
                          omit to run in no-reference / voice-design mode)
    - reference_id      : optional client id to cache the encoded reference. Send
                          it with reference_audio once; later requests with the
                          same id may omit the audio and reuse the cached latent
                          (skips the codec encode). See --ref-cache-size.
    - seconds           : output duration in seconds; omit to let the v3 duration
                          predictor choose a natural length (default: None = auto)
    - duration_scale    : multiplier on the predicted duration (auto mode only)
    - min_seconds       : lower clamp for output duration       (default: 0.5)
    - max_seconds       : upper clamp for output duration       (default: 30.0)
    - num_steps         : diffusion steps              (default: 40)
    - cfg_scale_text    : text CFG scale               (default: 3.0)
    - cfg_scale_speaker : speaker CFG scale            (default: 5.0)
    - seed              : integer seed; omit=random    (default: None)

  Returns: audio/wav (mono, codec sample-rate). Response headers also carry
    timing telemetry: `Server-Timing` (per-stage + total ms), `X-Used-Seed`,
    and `X-RTF` (real-time factor = compute-time / audio-length).

  Errors: 422 (bad params), 503 (overloaded — includes `Retry-After`),
    504 (synthesis timed out), 500 (internal).

  Voice registry (OpenAI-style — register once, reuse by id, no re-upload):
    POST   /voices         form: reference_audio, voice_id? → create (409 if exists)
    PUT    /voices/{id}    form: reference_audio → create-or-replace (update)
    GET    /voices         → {"voices":[…]}
    GET    /voices/{id}    → voice metadata (404 if missing)
    DELETE /voices/{id}    → {"deleted": id}
  A registered voice_id can be passed as `reference_id` to /synthesize.
  Voices are persisted to --voices-dir and reloaded on startup.

  GET /health  →  {"status":"ok","pool_size":N,"available":K,"voices":V}

Usage (single GPU, 1 worker):
  python api_server.py --checkpoint Aratako/Irodori-TTS-500M-v3 --devices cuda:0

Usage (two GPUs, 2 workers):
  python api_server.py --checkpoint Aratako/Irodori-TTS-500M-v3 --devices cuda:0,cuda:1

Usage (one GPU, 2 workers — both share the same device):
  python api_server.py --checkpoint Aratako/Irodori-TTS-500M-v3 --devices cuda:0 --num-workers 2
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import logging
import tempfile
import threading
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Callable

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field
from safetensors.torch import load_file as safetensors_load
from safetensors.torch import save_file as safetensors_save

from irodori_tts.eos import find_eos_and_split, warmup as warmup_eos
from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    resolve_runtime_device,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runtime pool
# ---------------------------------------------------------------------------

class RuntimePool:
    """Async pool of InferenceRuntime workers.

    Worker checkout is decoupled from the request coroutine: a worker is only
    returned to the pool once its background thread *actually finishes* (via a
    done-callback), never when the awaiting request gives up. This keeps a
    timed-out (504) request from leaking its worker. The same callback performs
    CUDA OOM cache cleanup before the worker is reused.

    An in-flight counter provides backpressure: callers `try_reserve()` before
    acquiring a worker and are rejected (503) once the configured queue depth is
    exceeded, instead of piling up unbounded and holding upload bytes in memory.
    """

    def __init__(
        self,
        runtimes: list[InferenceRuntime],
        executor: ThreadPoolExecutor,
        max_inflight: int,
    ) -> None:
        if not runtimes:
            raise ValueError("RuntimePool requires at least one runtime.")
        self._queue: asyncio.Queue[InferenceRuntime] = asyncio.Queue()
        for rt in runtimes:
            self._queue.put_nowait(rt)
        self._size = len(runtimes)
        self._executor = executor
        self._max_inflight = max_inflight
        self._inflight = 0

    @property
    def size(self) -> int:
        return self._size

    @property
    def available(self) -> int:
        return self._queue.qsize()

    @property
    def inflight(self) -> int:
        return self._inflight

    def try_reserve(self) -> bool:
        """Reserve an in-flight slot for backpressure. Runs in the loop thread,
        so the read-modify-write needs no lock. Returns False when overloaded."""
        if self._inflight >= self._max_inflight:
            return False
        self._inflight += 1
        return True

    def release_reservation(self) -> None:
        """Release a slot taken by try_reserve() when no submit() will follow."""
        self._inflight = max(0, self._inflight - 1)

    async def acquire_raw(self) -> InferenceRuntime:
        """Block until a worker is free and hand it out (no auto-return)."""
        return await self._queue.get()

    def submit(
        self, runtime: InferenceRuntime, fn: Callable[[InferenceRuntime], object]
    ) -> asyncio.Future:
        """Run `fn(runtime)` in the thread pool. The returned asyncio future
        resolves with fn's result. A done-callback (executed in the loop thread
        thanks to wrap_future) returns the worker to the pool, cleans up after
        OOM, and decrements the in-flight counter — independent of whether the
        awaiting request still cares about the result."""
        cf: Future = self._executor.submit(fn, runtime)
        afut = asyncio.wrap_future(cf)
        afut.add_done_callback(lambda f: self._on_done(runtime, f))
        return afut

    def _on_done(self, runtime: InferenceRuntime, fut: asyncio.Future) -> None:
        try:
            exc = fut.exception() if not fut.cancelled() else None
        except Exception:  # noqa: BLE001 — defensive; never let cleanup raise
            exc = None
        if exc is not None and _is_oom(exc):
            _empty_device_cache(runtime, exc)
        self._queue.put_nowait(runtime)
        self._inflight = max(0, self._inflight - 1)

    def unload_all(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait().unload()
            except asyncio.QueueEmpty:
                break


def _is_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return "out of memory" in str(exc).lower()


def _empty_device_cache(runtime: InferenceRuntime, exc: BaseException) -> None:
    """Free cached CUDA memory on a worker's device after an OOM so the next
    request on that worker has a clean allocator. Best-effort; never raises."""
    dev = getattr(runtime, "model_device", None)
    logger.warning("CUDA OOM on worker (%s): clearing cache — %s", dev, exc)
    try:
        if dev is not None and getattr(dev, "type", None) == "cuda":
            with torch.cuda.device(dev):
                torch.cuda.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        logger.exception("failed to clear CUDA cache after OOM")


# ---------------------------------------------------------------------------
# Reference latent cache
# ---------------------------------------------------------------------------

class ReferenceLatentCache:
    """Thread-safe LRU cache of precomputed reference latents, keyed by a
    client-supplied ``reference_id``.

    The cached value is the CPU latent from `InferenceRuntime.encode_reference_latent`
    — the expensive part of reference preprocessing. Latents are device-agnostic
    (CPU), so workers on different GPUs can share one cache. Repeat requests with
    the same id skip the codec encode (and the client may omit the audio upload
    entirely). Get/put run inside worker threads, hence the lock.
    """

    def __init__(self, max_size: int) -> None:
        self._max = max_size
        self._store: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._max > 0

    def get(self, key: str) -> torch.Tensor | None:
        if not self.enabled:
            return None
        with self._lock:
            value = self._store.get(key)
            if value is not None:
                self._store.move_to_end(key)
            return value

    def put(self, key: str, value: torch.Tensor) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# ---------------------------------------------------------------------------
# Voice registry (OpenAI-style named voices)
# ---------------------------------------------------------------------------

def _valid_voice_id(voice_id: str) -> bool:
    # Allow Unicode (e.g. Japanese names like 茉子) but keep the id a safe
    # filename: reject path separators, traversal, control chars, and hidden /
    # whitespace-edged names.
    if not voice_id or len(voice_id) > 128:
        return False
    if voice_id in (".", ".."):
        return False
    if voice_id[0] == "." or voice_id != voice_id.strip():
        return False
    if any(ch in voice_id for ch in ("/", "\\", "\x00")):
        return False
    return all(ord(ch) >= 0x20 for ch in voice_id)  # no control characters


class VoiceStore:
    """Disk-backed registry of named voices (pinned reference latents).

    Unlike the transient LRU ``ReferenceLatentCache``, registered voices are
    never evicted and survive restarts: each voice's CPU latent is persisted to
    ``<dir>/<voice_id>.safetensors`` and reloaded on startup. ``/synthesize``
    resolves a ``reference_id`` against this store first, then the LRU cache.
    """

    _TENSOR_KEY = "ref_latent"

    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._lock = threading.Lock()
        self._latents: dict[str, torch.Tensor] = {}
        self._meta: dict[str, dict] = {}

    def _path(self, voice_id: str) -> Path:
        return self._dir / f"{voice_id}.safetensors"

    @staticmethod
    def _meta_for(voice_id: str, latent: torch.Tensor, path: Path) -> dict:
        return {
            "voice_id": voice_id,
            "latent_steps": int(latent.shape[1]) if latent.dim() >= 2 else 0,
            "latent_dim": int(latent.shape[-1]),
            "created": path.stat().st_mtime if path.exists() else None,
        }

    def load_existing(self) -> int:
        self._dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for path in sorted(self._dir.glob("*.safetensors")):
            voice_id = path.stem
            if not _valid_voice_id(voice_id):
                logger.warning("skipping voice file with invalid id: %s", path.name)
                continue
            try:
                latent = safetensors_load(str(path))[self._TENSOR_KEY]
            except Exception:  # noqa: BLE001
                logger.exception("failed to load voice file %s", path.name)
                continue
            with self._lock:
                self._latents[voice_id] = latent
                self._meta[voice_id] = self._meta_for(voice_id, latent, path)
            count += 1
        return count

    def exists(self, voice_id: str) -> bool:
        with self._lock:
            return voice_id in self._latents

    def register(self, voice_id: str, latent: torch.Tensor) -> dict:
        latent = latent.detach().cpu().contiguous()
        path = self._path(voice_id)
        safetensors_save({self._TENSOR_KEY: latent}, str(path))
        meta = self._meta_for(voice_id, latent, path)
        with self._lock:
            self._latents[voice_id] = latent
            self._meta[voice_id] = meta
        return meta

    def get(self, voice_id: str) -> torch.Tensor | None:
        with self._lock:
            return self._latents.get(voice_id)

    def get_meta(self, voice_id: str) -> dict | None:
        with self._lock:
            return self._meta.get(voice_id)

    def list(self) -> list[dict]:
        with self._lock:
            return sorted(self._meta.values(), key=lambda m: m["voice_id"])

    def delete(self, voice_id: str) -> bool:
        with self._lock:
            existed = self._latents.pop(voice_id, None) is not None
            self._meta.pop(voice_id, None)
        self._path(voice_id).unlink(missing_ok=True)
        return existed

    def __len__(self) -> int:
        with self._lock:
            return len(self._latents)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_checkpoint(checkpoint: str) -> str:
    checkpoint = checkpoint.strip()
    suffix = Path(checkpoint).suffix.lower()
    if suffix in {".pt", ".safetensors"} or Path(checkpoint).exists():
        return checkpoint
    resolved = hf_hub_download(repo_id=checkpoint, filename="model.safetensors")
    logger.info("downloaded checkpoint: hf://%s → %s", checkpoint, resolved)
    return resolved


def _build_runtime(
    checkpoint: str,
    device: str,
    precision: str,
    codec_device: str,
    codec_precision: str,
    enable_watermark: bool,
) -> InferenceRuntime:
    key = RuntimeKey(
        checkpoint=checkpoint,
        model_device=device,
        model_precision=precision,
        codec_device=codec_device,
        codec_precision=codec_precision,
        enable_watermark=enable_watermark,
    )
    logger.info("loading worker on %s (%s) …", device, precision)
    rt = InferenceRuntime.from_key(key)
    logger.info("worker ready on %s.", device)
    return rt


def _audio_to_wav_bytes(audio: torch.Tensor, sample_rate: int) -> bytes:
    # Encode straight to memory with soundfile — no per-request disk round-trip.
    # (torchaudio's torchcodec backend can't write to BytesIO; soundfile can.)
    # `audio` is (channels, samples); soundfile expects frames-first, so .T.
    data = audio.float().cpu().numpy().T  # → (samples, channels)
    buf = io.BytesIO()
    sf.write(buf, data, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Irodori-TTS API")


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI, args: argparse.Namespace) -> AsyncIterator[None]:
    checkpoint = _resolve_checkpoint(args.checkpoint)
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    num_workers = args.num_workers
    worker_devices = [devices[i % len(devices)] for i in range(num_workers)]
    codec_precision = args.codec_precision or args.precision

    runtimes = [
        _build_runtime(
            checkpoint=checkpoint,
            device=dev,
            precision=args.precision,
            codec_device=dev,
            codec_precision=codec_precision,
            enable_watermark=args.enable_watermark,
        )
        for dev in worker_devices
    ]

    executor = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="tts-worker")
    max_queue = args.max_queue if args.max_queue is not None else num_workers
    max_inflight = num_workers + max_queue
    pool = RuntimePool(runtimes, executor, max_inflight=max_inflight)
    logger.info(
        "pool ready: %d worker(s) on %s | max in-flight=%d (workers+%d queued) | timeout=%.1fs",
        num_workers, worker_devices, max_inflight, max_queue, args.request_timeout,
    )

    warmup_eos()
    logger.info("fast-bunkai splitter warmed up.")

    ref_cache = ReferenceLatentCache(args.ref_cache_size)
    logger.info(
        "reference latent cache: %s",
        f"LRU max {args.ref_cache_size}" if ref_cache.enabled else "disabled",
    )

    voice_store = VoiceStore(Path(args.voices_dir).expanduser())
    loaded = voice_store.load_existing()
    logger.info("voice registry: %d voice(s) loaded from %s", loaded, voice_store._dir)

    app.state.pool = pool
    app.state.executor = executor
    app.state.request_timeout = args.request_timeout
    app.state.ref_cache = ref_cache
    app.state.voice_store = voice_store

    yield

    executor.shutdown(wait=True)
    pool.unload_all()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post(
    "/synthesize",
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}}}},
)
async def synthesize(
    request: Request,
    text: Annotated[str, Form()],
    reference_audio: Annotated[UploadFile | None, File()] = None,
    reference_id: Annotated[str | None, Form()] = None,
    seconds: Annotated[float | None, Form()] = None,
    duration_scale: Annotated[float, Form()] = 1.0,
    min_seconds: Annotated[float, Form()] = 0.5,
    max_seconds: Annotated[float, Form()] = 30.0,
    num_steps: Annotated[int, Form()] = 40,
    cfg_scale_text: Annotated[float, Form()] = 3.0,
    cfg_scale_speaker: Annotated[float, Form()] = 5.0,
    seed: Annotated[int | None, Form()] = None,
) -> Response:
    pool: RuntimePool = request.app.state.pool
    request_timeout: float = request.app.state.request_timeout
    ref_cache: ReferenceLatentCache = request.app.state.ref_cache
    voice_store: VoiceStore = request.app.state.voice_store

    # ---- validation (no in-flight slot held yet) ----
    if not text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty.")
    if seconds is not None and seconds <= 0:
        raise HTTPException(status_code=422, detail="seconds must be > 0 when provided.")
    if duration_scale <= 0:
        raise HTTPException(status_code=422, detail="duration_scale must be > 0.")
    if min_seconds <= 0 or max_seconds < min_seconds:
        raise HTTPException(status_code=422, detail="require 0 < min_seconds <= max_seconds.")
    if num_steps < 1:
        raise HTTPException(status_code=422, detail="num_steps must be >= 1.")
    reference_id = reference_id.strip() if reference_id else None
    if reference_id is not None and not reference_id:
        reference_id = None

    # Read upload bytes before handing off to the thread.
    has_upload = reference_audio is not None
    audio_bytes: bytes = b""
    ref_suffix = ".wav"
    if has_upload:
        audio_bytes = await reference_audio.read()  # type: ignore[union-attr]
        ref_suffix = Path(reference_audio.filename or "ref.wav").suffix or ".wav"  # type: ignore[union-attr]

    # ---- backpressure: reject early when overloaded instead of piling up ----
    if not pool.try_reserve():
        return JSONResponse(
            status_code=503,
            content={"detail": "server overloaded; retry shortly."},
            headers={"Retry-After": "1"},
        )

    def _run_synthesis(runtime: InferenceRuntime) -> tuple:
        # Resolve the reference latent, in priority order:
        #   1. registered voice (pinned, disk-backed) by reference_id
        #   2. transient LRU cache by reference_id
        #   3. fresh upload → encode (and cache under reference_id if given)
        #   4. reference_id with none of the above → error
        ref_latent_tensor = voice_store.get(reference_id) if reference_id else None
        cache_status = "none"
        if ref_latent_tensor is not None:
            cache_status = "voice"
        elif reference_id and (cached := ref_cache.get(reference_id)) is not None:
            ref_latent_tensor = cached
            cache_status = "hit"
        elif has_upload:
            cache_status = "miss"
            with tempfile.NamedTemporaryFile(suffix=ref_suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                ref_wav_path = tmp.name
            try:
                ref_latent_tensor = runtime.encode_reference_latent(
                    ref_wav_path,
                    normalize_db=-16.0,
                    ensure_max=True,
                    max_ref_seconds=30.0,
                )
            finally:
                Path(ref_wav_path).unlink(missing_ok=True)
            if reference_id:
                ref_cache.put(reference_id, ref_latent_tensor)
        elif reference_id:
            raise ValueError(
                f"unknown reference_id '{reference_id}'; register it via "
                "POST /voices, or send reference_audio with the request."
            )

        req = SamplingRequest(
            text=text,
            ref_latent_tensor=ref_latent_tensor,
            no_ref=ref_latent_tensor is None,
            seconds=seconds,
            duration_scale=duration_scale,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            num_steps=num_steps,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_speaker=cfg_scale_speaker,
            seed=seed,
        )
        result = runtime.synthesize(req, log_fn=logger.debug)
        audio_len_sec = result.audio.shape[-1] / result.sample_rate
        wav_bytes = _audio_to_wav_bytes(result.audio, result.sample_rate)
        return (wav_bytes, result.stage_timings, result.total_to_decode,
                result.used_seed, audio_len_sec, cache_status)

    # Acquire a worker, then submit. Once submit() is called, the pool's
    # done-callback owns returning the worker and decrementing the in-flight
    # counter — so a 504 timeout below cannot leak the worker.
    try:
        runtime = await pool.acquire_raw()
    except BaseException:
        pool.release_reservation()
        raise

    afut = pool.submit(runtime, _run_synthesis)
    try:
        wav_bytes, stage_timings, total_s, used_seed, audio_len_s, cache_status = (
            await asyncio.wait_for(asyncio.shield(afut), timeout=request_timeout)
        )
    except asyncio.TimeoutError:
        logger.warning(
            "synthesis timed out after %.1fs; worker reclaimed on completion",
            request_timeout,
        )
        raise HTTPException(
            status_code=504, detail=f"synthesis timed out after {request_timeout:.0f}s."
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("synthesis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ---- timing telemetry ----
    rtf = (total_s / audio_len_s) if audio_len_s > 0 else 0.0
    stage_parts = [
        f"{name.replace(' ', '_')};dur={sec * 1000.0:.1f}" for name, sec in stage_timings
    ]
    stage_parts.append(f"total;dur={total_s * 1000.0:.1f}")
    logger.info(
        "synthesized: RTF=%.3f total=%.0fms seed=%d audio=%.2fs ref_cache=%s",
        rtf, total_s * 1000.0, used_seed, audio_len_s, cache_status,
    )

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=output.wav",
            "Server-Timing": ", ".join(stage_parts),
            "X-Used-Seed": str(used_seed),
            "X-RTF": f"{rtf:.3f}",
            "X-Reference-Cache": cache_status,
        },
    )


class VoiceMeta(BaseModel):
    voice_id: str
    latent_steps: int = Field(..., description="Reference latent length (timesteps).")
    latent_dim: int
    created: float | None = Field(None, description="Unix epoch the voice was registered.")


_VOICE_ID_HELP = (
    "voice_id must be 1–128 chars, no '/', '\\', control characters, "
    "no leading dot, and no leading/trailing whitespace."
)


async def _register_voice_latent(
    request: Request, voice_id: str, reference_audio: UploadFile
) -> dict:
    """Read the upload, encode its reference latent on a pooled worker, and
    persist it under ``voice_id`` (overwriting any existing). Returns metadata.

    Shared by POST (create) and PUT (upsert). Goes through the same pool /
    backpressure / timeout machinery as /synthesize.
    """
    pool: RuntimePool = request.app.state.pool
    request_timeout: float = request.app.state.request_timeout
    store: VoiceStore = request.app.state.voice_store

    audio_bytes = await reference_audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="reference_audio is empty.")
    ref_suffix = Path(reference_audio.filename or "ref.wav").suffix or ".wav"

    if not pool.try_reserve():
        raise HTTPException(
            status_code=503,
            detail="server overloaded; retry shortly.",
            headers={"Retry-After": "1"},
        )

    def _encode_and_store(runtime: InferenceRuntime) -> dict:
        with tempfile.NamedTemporaryFile(suffix=ref_suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            ref_wav_path = tmp.name
        try:
            latent = runtime.encode_reference_latent(
                ref_wav_path, normalize_db=-16.0, ensure_max=True, max_ref_seconds=30.0
            )
        finally:
            Path(ref_wav_path).unlink(missing_ok=True)
        return store.register(voice_id, latent)

    try:
        runtime = await pool.acquire_raw()
    except BaseException:
        pool.release_reservation()
        raise

    afut = pool.submit(runtime, _encode_and_store)
    try:
        return await asyncio.wait_for(asyncio.shield(afut), timeout=request_timeout)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504, detail=f"voice registration timed out after {request_timeout:.0f}s."
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("voice registration failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/voices", response_model=VoiceMeta, status_code=201)
async def register_voice(
    request: Request,
    reference_audio: Annotated[UploadFile, File()],
    voice_id: Annotated[str | None, Form()] = None,
) -> VoiceMeta:
    """Register a NEW voice from a reference clip; reuse it by id in /synthesize
    (no re-upload). Fails 409 if the id exists — use PUT /voices/{id} to update.
    The encoded latent is persisted to disk."""
    store: VoiceStore = request.app.state.voice_store

    if voice_id is not None:
        voice_id = voice_id.strip() or None
    if voice_id is not None:
        if not _valid_voice_id(voice_id):
            raise HTTPException(status_code=422, detail=_VOICE_ID_HELP)
        if store.exists(voice_id):
            raise HTTPException(
                status_code=409,
                detail=f"voice_id '{voice_id}' already exists; use PUT /voices/{voice_id} to update.",
            )
    else:
        voice_id = uuid.uuid4().hex[:16]

    meta = await _register_voice_latent(request, voice_id, reference_audio)
    logger.info("registered voice '%s' (%d latent steps)", meta["voice_id"], meta["latent_steps"])
    return VoiceMeta(**meta)


@app.put("/voices/{voice_id}", response_model=VoiceMeta)
async def upsert_voice(
    voice_id: str,
    request: Request,
    reference_audio: Annotated[UploadFile, File()],
) -> VoiceMeta:
    """Create or replace the voice at ``voice_id`` (upsert). Use this to update
    an existing voice's reference audio in place — the old latent (memory + disk)
    is overwritten atomically by re-encoding the new clip."""
    store: VoiceStore = request.app.state.voice_store
    if not _valid_voice_id(voice_id):
        raise HTTPException(status_code=422, detail=_VOICE_ID_HELP)

    existed = store.exists(voice_id)
    meta = await _register_voice_latent(request, voice_id, reference_audio)
    logger.info(
        "%s voice '%s' (%d latent steps)",
        "updated" if existed else "created", meta["voice_id"], meta["latent_steps"],
    )
    return VoiceMeta(**meta)


@app.get("/voices")
def list_voices(request: Request) -> dict:
    store: VoiceStore = request.app.state.voice_store
    return {"voices": store.list()}


@app.get("/voices/{voice_id}", response_model=VoiceMeta)
def get_voice(voice_id: str, request: Request) -> VoiceMeta:
    store: VoiceStore = request.app.state.voice_store
    meta = store.get_meta(voice_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"voice '{voice_id}' not found.")
    return VoiceMeta(**meta)


@app.delete("/voices/{voice_id}")
def delete_voice(voice_id: str, request: Request) -> dict:
    store: VoiceStore = request.app.state.voice_store
    if not store.delete(voice_id):
        raise HTTPException(status_code=404, detail=f"voice '{voice_id}' not found.")
    logger.info("deleted voice '%s'", voice_id)
    return {"deleted": voice_id}


@app.get("/health")
def health(request: Request) -> dict:
    pool: RuntimePool | None = getattr(request.app.state, "pool", None)
    if pool is None:
        return {"status": "starting", "pool_size": 0, "available": 0, "inflight": 0}
    ref_cache: ReferenceLatentCache | None = getattr(request.app.state, "ref_cache", None)
    voice_store: VoiceStore | None = getattr(request.app.state, "voice_store", None)
    return {
        "status": "ok",
        "pool_size": pool.size,
        "available": pool.available,
        "inflight": pool.inflight,
        "ref_cache_size": len(ref_cache) if ref_cache is not None else 0,
        "voices": len(voice_store) if voice_store is not None else 0,
    }


class EosRequest(BaseModel):
    text: str = Field(..., description="Raw input text to segment into sentences.")


class EosResponse(BaseModel):
    positions: list[int] = Field(default_factory=list)
    sentences: list[str] = Field(default_factory=list)


@app.post("/eos", response_model=EosResponse)
async def eos(req: EosRequest) -> EosResponse:
    positions, sentences = find_eos_and_split(req.text)
    return EosResponse(positions=positions, sentences=sentences)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Irodori-TTS FastAPI server.")
    parser.add_argument(
        "--checkpoint", required=True,
        help="Local .pt/.safetensors path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--devices", default="cuda",
        help=(
            "Comma-separated list of devices for workers "
            "(e.g. 'cuda:0,cuda:1'). Cycled if fewer than --num-workers. "
            "Default: cuda"
        ),
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of parallel inference workers. Defaults to number of devices.",
    )
    parser.add_argument(
        "--precision", default="fp32", choices=["fp32", "bf16"],
        help="Model precision (default: fp32).",
    )
    parser.add_argument(
        "--codec-precision", default=None, choices=["fp32", "bf16"],
        help="Codec precision (defaults to --precision).",
    )
    parser.add_argument("--enable-watermark", action="store_true")
    parser.add_argument(
        "--request-timeout", type=float, default=60.0,
        help=(
            "Per-request synthesis timeout in seconds; clients get 504 on "
            "expiry while the worker is reclaimed once its thread finishes "
            "(default: 60)."
        ),
    )
    parser.add_argument(
        "--max-queue", type=int, default=None,
        help=(
            "Max requests allowed to wait beyond the busy workers before new "
            "requests get 503. Total in-flight cap = num_workers + max_queue. "
            "Defaults to num_workers."
        ),
    )
    parser.add_argument(
        "--ref-cache-size", type=int, default=64,
        help=(
            "LRU cache size for precomputed reference latents keyed by "
            "`reference_id`. Repeat requests with the same id skip the codec "
            "encode. 0 disables caching (default: 64)."
        ),
    )
    parser.add_argument(
        "--voices-dir", default="./voices",
        help=(
            "Directory for persisted registered voices (POST /voices). Loaded "
            "on startup so voices survive restarts (default: ./voices)."
        ),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--log-level", default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if not devices:
        raise SystemExit("--devices must not be empty.")
    for dev in devices:
        resolve_runtime_device(dev)

    if args.num_workers is None:
        args.num_workers = len(devices)
    if args.num_workers < 1:
        raise SystemExit("--num-workers must be >= 1.")

    if args.request_timeout <= 0:
        raise SystemExit("--request-timeout must be > 0.")
    if args.max_queue is not None and args.max_queue < 0:
        raise SystemExit("--max-queue must be >= 0.")
    if args.ref_cache_size < 0:
        raise SystemExit("--ref-cache-size must be >= 0.")

    # Bind the lifespan with startup args via a wrapper.
    @contextlib.asynccontextmanager
    async def lifespan(a: FastAPI) -> AsyncIterator[None]:
        async with _lifespan(a, args):
            yield

    app.router.lifespan_context = lifespan

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
