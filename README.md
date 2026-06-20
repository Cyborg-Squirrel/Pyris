# Pyris

A Python vision and speech-to-text library powered by vision LLMs and Whisper.

Pyris fetches media from a pluggable provider, processes it with ffmpeg (frame
sampling, cropping, audio extraction), and routes it to a vision LLM and/or a
Whisper-compatible STT endpoint to answer a prompt about it.

- **Zero Python runtime dependencies** — only `ffmpeg`/`ffprobe` on your `PATH`.
- **Sync and async** APIs (`analyze` / `analyze_async`).
- **Deterministic routing** by media type
- **Pluggable providers** — bundled local-file provider; implement `MediaProvider`
  for YouTube, remote URLs, S3, etc.

## Requirements

- Python 3.11+
- `ffmpeg` and `ffprobe` on `PATH`

## Install

```bash
pip install -e .
```

## Quickstart

```python
from pyris import Pyris, Config, VisionConfig, SttConfig, FileProvider, RequestMode
from pyris import SubprocessFfmpeg, TimeRange

config = Config(
    vision=VisionConfig(
        base_url="https://api.openai.com/v1",
        api_key="...",
        model="gpt-4o",          # the driver/reasoning model (must support vision)
        # vision_model="...",    # optional dedicated vision model
    ),
    stt=SttConfig(
        base_url="http://localhost:8000/v1",  # e.g. a faster-whisper server
        api_key="...",
        model="whisper-1",
    ),
)

pyris = Pyris.from_config(config)          # validates config + checks ffmpeg

# ffmpeg layer is reused as the provider's prober
probe = SubprocessFfmpeg(config)
provider = FileProvider("clip.mp4", probe=probe)

result = pyris.analyze(provider, "What is the person assembling, and in what order?")
print(result.text)
print(result.transcript.full_text if result.transcript else "")
print(result.usage)            # tokens, frames_sent, audio_seconds
```

### Modes

Routing is automatic from the media type, or force it with `mode=`:

| Media            | AUTO resolves to        |
|------------------|-------------------------|
| image            | `VISION`                |
| audio            | `STT`                   |
| video (+ audio)  | `VISION_AND_STT`        |
| video (silent)   | `VISION`                |

An explicit mode that conflicts with the media (e.g. `STT` on an image) raises
`UnsupportedMediaError`.

### Cropping, sampling, async

```python
import asyncio
from pyris import TimeRange

# only analyze 30s–90s, sampled at 0.5 fps
result = pyris.analyze(
    provider, "Summarize this segment",
    time_range=TimeRange(30, 90), fps=0.5,
)

# async (offloads ffmpeg to a thread, awaits the LLM calls)
result = asyncio.run(pyris.analyze_async(provider, "Describe the scene"))
```

### Pure transcription

Pass an empty prompt with audio to get just the transcript (skips the LLM step):

```python
result = pyris.analyze(FileProvider("podcast.mp3", probe=probe), "")
print(result.transcript.full_text)
```

### Inspecting the result

`analyze` returns a structured `AnalysisResult` so you can trace what backed the
answer, not just the answer text:

```python
result = pyris.analyze(provider, "When does the speaker mention pricing?")

print(result.mode)          # RequestMode.VISION_AND_STT
print(result.text)          # the synthesized answer
print(result.frames_used)   # how many sampled frames were sent

if result.transcript:
    for seg in result.transcript.segments:
        print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text}")

u = result.usage
print(u.input_tokens, u.output_tokens, u.frames_sent, u.audio_seconds)
```

## Examples

Runnable scripts live in [`examples/`](examples/). They read endpoint/key
settings from environment variables (see [`examples/_shared.py`](examples/_shared.py)):

```bash
export PYRIS_VISION_BASE_URL="https://api.openai.com/v1"
export PYRIS_VISION_API_KEY="sk-..."
export PYRIS_VISION_MODEL="gpt-4o"
# optional, for audio / video-with-audio:
export PYRIS_STT_BASE_URL="http://localhost:8000/v1"
export PYRIS_STT_API_KEY="sk-..."
export PYRIS_STT_MODEL="whisper-1"

python examples/analyze_video.py clip.mp4 "What is being demonstrated?"
python examples/transcribe_audio.py podcast.mp3            # pure transcription
python examples/transcribe_audio.py podcast.mp3 "Summarize"  # Q&A over audio
python examples/async_batch.py "Describe this" a.mp4 b.jpg c.mp3
python examples/custom_provider.py https://example.com/clip.mp4 "What is this?"
```

| Example | Shows |
|---------|-------|
| [`analyze_video.py`](examples/analyze_video.py) | End-to-end video analysis + inspecting the full result |
| [`transcribe_audio.py`](examples/transcribe_audio.py) | Pure transcription vs. asking a question about audio |
| [`async_batch.py`](examples/async_batch.py) | Concurrent fan-out over many files with `analyze_async` |
| [`custom_provider.py`](examples/custom_provider.py) | Implementing `MediaProvider` for a new source (remote URL) |

## Custom providers

Implement `MediaProvider` to add a source. A provider only *fetches* and
*probes* — frame sampling and audio extraction stay in the core, so you never
reimplement ffmpeg:

```python
from pyris import MediaProvider, RawMedia

class MyProvider(MediaProvider):
    @property
    def supports_range_fetch(self) -> bool:
        return True  # if you can fetch only a time range remotely

    def probe(self) -> MediaInfo: ...
    def fetch(self, time_range=None) -> RawMedia: ...
```

## Development

```bash
pip install -e '.[test]'
pytest          # ffmpeg integration tests auto-skip if ffmpeg is missing
```

## Configuration reference

| Field | Required | Notes |
|-------|----------|-------|
| `vision.base_url` / `api_key` / `model` | yes | OpenAI-compatible chat endpoint; `model` must support vision unless `vision_model` is set |
| `vision.max_images_per_request` | no (20) | over this, the pipeline batches + map-reduces |
| `vision.image_max_dim` | no (1024) | frames downscaled to this before sending |
| `stt.*` | optional at startup, required for STT | OpenAI `/audio/transcriptions`-compatible (faster-whisper, whisper.cpp) |
| `sampling.default_fps` / `max_frames` | no | frame budget |
| `sampling.scene_detection` / `scene_threshold` | no | prefer scene-change frames |
| `system_prompt_path` | no | overrides the bundled default |
