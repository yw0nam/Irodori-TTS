# POST /eos — sentence boundary detection

In-tree addition for streaming-TTS clients (e.g. the Unity desktop companion) that need to detect Japanese sentence boundaries on a partial buffer before requesting synthesis.

This endpoint is **not part of upstream Aratako/Irodori-TTS** — it lives alongside `/synthesize` in `api_server.py` and is implemented in `irodori_tts/eos.py` using [`fast-bunkai`](https://github.com/hotchpotch/fast-bunkai).

## Request

```http
POST /eos HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "text": "今日はいい天気ですね。明日はどうかな？"
}
```

## Response

```json
{
  "positions": [11, 19],
  "sentences": ["今日はいい天気ですね。", "明日はどうかな？"]
}
```

- `positions[i]` is the exclusive upper bound such that `text[:positions[i]]` is the i-th sentence-terminated prefix.
- `sentences[i]` is the raw substring fast-bunkai yielded for sentence `i`.
- Empty / whitespace-only input returns `{"positions": [], "sentences": []}` without raising.
- Trailing text without a terminator is **not** included (mirrors fast-bunkai semantics).

## Latency

Localhost p50 ≈ 1-5 ms after warm-up. The first call after process start pays the Janome dictionary load (~200-500 ms); `api_server.py` calls `warmup()` during lifespan so end-users do not see it.

## Concurrency

`fast_bunkai.FastBunkai` is constructed once per process via `@lru_cache(maxsize=1)` in `irodori_tts/eos.py` and is read-only after init, so concurrent FastAPI requests share the same splitter without coordination.

## Smoke test

After `uv sync` and `python api_server.py --checkpoint <id>`:

```bash
curl -s -X POST http://localhost:8000/eos \
  -H "Content-Type: application/json" \
  -d '{"text":"今日はいい天気ですね。明日はどうかな？"}'
# → {"positions":[11,19],"sentences":["今日はいい天気ですね。","明日はどうかな？"]}
```

## Unity client contract (`FastBunkaiSidecarClient.cs`)

The Unity-side wrapper in `Mate-EnginePlus/Assets/MATE ENGINE - Scripts/Hermes/FastBunkaiSidecarClient.cs` will:

1. POST `{"text": "<buffer>"}` to `${IRODORI_BASE}/eos`.
2. Read `positions` from the response.
3. Apply the same secondary filter the nanobot reference uses (`real_positions = [p for p in positions if p>0 and buffer[:p].TrimEnd() ends with a member of "。！？.!?\\n"]`).
4. Emit sentences whose length ≥ `minChunkLength` (default 50).

See `D:\codes\waifu\.sisyphus\plans\hermes-migration.md` Phase B for the full chunker contract.
