import asyncio
import io
import wave

import pytest

from pyris import (
    CallbackTranscriptSink,
    Pyris,
    TranscriptEvent,
    WavFileAudioStreamProvider,
)
from pyris import llm
from pyris.config import SttConfig
from pyris.errors import ConfigError, ProviderError
from pyris.provider import AudioStreamProvider
from pyris.types import TranscriptSegment
from tests.conftest import FakeFfmpeg, FakeVision, make_config


# -- fakes -------------------------------------------------------------------


class ListSource(AudioStreamProvider):
    """Yields a fixed list of blobs — stands in for a live capture provider."""

    def __init__(self, blobs):
        self._blobs = list(blobs)

    async def stream(self):
        for blob in self._blobs:
            yield blob


class FakeStreamingStt:
    """Emits one final segment per received blob and records what it received."""

    def __init__(self):
        self.received: list[bytes] = []
        self.model: str | None = None
        self.language: str | None = None

    async def stream_transcribe(self, audio, *, model, language=None):
        self.model = model
        self.language = language
        i = 0
        async for blob in audio:
            self.received.append(blob)
            yield TranscriptEvent(TranscriptSegment(float(i), float(i) + 1, f"seg{i}"))
            i += 1


def make_pyris(*, streaming_stt):
    return Pyris(
        make_config(),
        ffmpeg=FakeFfmpeg(),
        vision=FakeVision(),
        streaming_stt=streaming_stt,
    )


def collect(pyris, source, sink, **kw):
    return asyncio.run(pyris.stream_stt(source, sink, **kw))


# -- pipeline wiring ---------------------------------------------------------


def test_stream_stt_pipes_blobs_and_returns_transcript():
    sstt = FakeStreamingStt()
    pyris = make_pyris(streaming_stt=sstt)
    seen: list[TranscriptEvent] = []
    sink = CallbackTranscriptSink(seen.append)

    transcript = collect(pyris, ListSource([b"a", b"b", b"c"]), sink)

    # source blobs reached the STT host in order...
    assert sstt.received == [b"a", b"b", b"c"]
    assert sstt.model == "whisper"  # from make_config's SttConfig
    # ...each event reached the sink...
    assert [e.segment.text for e in seen] == ["seg0", "seg1", "seg2"]
    # ...and the accumulated transcript is returned.
    assert transcript.full_text == "seg0 seg1 seg2"
    assert len(transcript.segments) == 3


def test_stream_stt_language_override():
    sstt = FakeStreamingStt()
    pyris = make_pyris(streaming_stt=sstt)
    transcript = collect(
        pyris, ListSource([b"a"]), CallbackTranscriptSink(lambda e: None), language="fr"
    )
    assert sstt.language == "fr"
    assert transcript.language == "fr"


def test_stream_stt_forwards_partials_to_sink_but_only_finals_to_transcript():
    class PartialThenFinal:
        async def stream_transcribe(self, audio, *, model, language=None):
            async for _ in audio:
                yield TranscriptEvent(TranscriptSegment(0, 1, "partial"), is_final=False)
                yield TranscriptEvent(TranscriptSegment(0, 1, "final"), is_final=True)

    pyris = make_pyris(streaming_stt=PartialThenFinal())
    seen: list[str] = []
    transcript = collect(
        pyris,
        ListSource([b"x"]),
        CallbackTranscriptSink(lambda e: seen.append(e.segment.text)),
    )
    assert seen == ["partial", "final"]  # sink sees both
    assert transcript.full_text == "final"  # transcript keeps only finalized ones


def test_stream_stt_without_client_raises():
    pyris = make_pyris(streaming_stt=None)
    with pytest.raises(ConfigError):
        collect(pyris, ListSource([b"a"]), CallbackTranscriptSink(lambda e: None))


def test_stream_stt_closes_sink_even_on_error():
    class BoomStt:
        async def stream_transcribe(self, audio, *, model, language=None):
            async for _ in audio:
                yield TranscriptEvent(TranscriptSegment(0, 1, "ok"))
                raise RuntimeError("boom")

    closed: list[bool] = []
    sink = CallbackTranscriptSink(lambda e: None, on_close=lambda: closed.append(True))
    pyris = make_pyris(streaming_stt=BoomStt())
    with pytest.raises(RuntimeError):
        collect(pyris, ListSource([b"a"]), sink)
    assert closed == [True]


# -- bundled sink ------------------------------------------------------------


def test_callback_sink_awaits_async_callbacks():
    seen: list[str] = []
    closed: list[bool] = []

    async def on_event(event):
        seen.append(event.segment.text)

    async def on_close():
        closed.append(True)

    sink = CallbackTranscriptSink(on_event, on_close=on_close)

    async def run():
        await sink.emit(TranscriptEvent(TranscriptSegment(0, 1, "hi")))
        await sink.aclose()

    asyncio.run(run())
    assert seen == ["hi"]
    assert closed == [True]


# -- concrete OpenAI-compatible streaming client -----------------------------


def test_streaming_client_offsets_timeline_across_blobs(monkeypatch):
    # Each blob is POSTed on its own with 0-based segment times; the client must
    # shift them onto a running timeline so timestamps stay absolute.
    responses = iter(
        [
            {"segments": [
                {"start": 0.0, "end": 1.0, "text": "a"},
                {"start": 1.0, "end": 2.0, "text": "b"},
            ]},
            {"segments": [{"start": 0.0, "end": 1.5, "text": "c"}]},
        ]
    )
    monkeypatch.setattr(llm, "_send", lambda req, timeout: next(responses))
    client = llm.OpenAICompatibleSttClient(
        SttConfig(base_url="http://x/v1", api_key="k", model="whisper")
    )

    async def blobs():
        yield b"blob-one"
        yield b"blob-two"

    async def run():
        return [e async for e in client.stream_transcribe(blobs(), model="whisper")]

    events = asyncio.run(run())
    got = [(e.segment.text, e.segment.start, e.segment.end) for e in events]
    assert got == [("a", 0.0, 1.0), ("b", 1.0, 2.0), ("c", 2.0, 3.5)]
    # every event is final -- this transport has no partial hypotheses...
    assert all(e.is_final for e in events)
    # ...but end_of_batch marks only the last segment of each blob's response.
    assert [e.end_of_batch for e in events] == [False, True, True]


def test_streaming_client_marks_end_of_batch_on_single_segment_blobs(monkeypatch):
    # A blob whose response has exactly one segment is end_of_batch immediately.
    responses = iter(
        [
            {"segments": [{"start": 0.0, "end": 1.0, "text": "a"}]},
            {"segments": [{"start": 0.0, "end": 1.0, "text": "b"}]},
        ]
    )
    monkeypatch.setattr(llm, "_send", lambda req, timeout: next(responses))
    client = llm.OpenAICompatibleSttClient(
        SttConfig(base_url="http://x/v1", api_key="k", model="whisper")
    )

    async def blobs():
        yield b"blob-one"
        yield b"blob-two"

    async def run():
        return [e async for e in client.stream_transcribe(blobs(), model="whisper")]

    events = asyncio.run(run())
    assert [e.end_of_batch for e in events] == [True, True]


def test_streaming_client_no_events_for_empty_blob_response(monkeypatch):
    # A blob whose response has no segments yields nothing (and doesn't crash
    # trying to compute end_of_batch against an empty list).
    responses = iter([{"segments": []}])
    monkeypatch.setattr(llm, "_send", lambda req, timeout: next(responses))
    client = llm.OpenAICompatibleSttClient(
        SttConfig(base_url="http://x/v1", api_key="k", model="whisper")
    )

    async def blobs():
        yield b"blob-one"

    async def run():
        return [e async for e in client.stream_transcribe(blobs(), model="whisper")]

    assert asyncio.run(run()) == []


# -- bundled WAV streaming provider ------------------------------------------


def _write_wav(path, *, framerate=8000, nframes=20000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(framerate)
        w.writeframes(b"\x00\x00" * nframes)


def test_wav_provider_yields_decodable_windows_covering_the_file(tmp_path):
    path = tmp_path / "a.wav"
    _write_wav(path, framerate=8000, nframes=20000)  # 2.5s
    provider = WavFileAudioStreamProvider(path, window_seconds=1.0)

    async def run():
        return [blob async for blob in provider.stream()]

    windows = asyncio.run(run())
    assert len(windows) == 3  # 8000 + 8000 + 4000 frames

    total = 0
    for blob in windows:
        with wave.open(io.BytesIO(blob), "rb") as r:  # each window is a real WAV
            assert (r.getframerate(), r.getnchannels(), r.getsampwidth()) == (8000, 1, 2)
            total += r.getnframes()
    assert total == 20000


def test_wav_provider_missing_file_raises_provider_error(tmp_path):
    provider = WavFileAudioStreamProvider(tmp_path / "nope.wav")

    async def run():
        return [blob async for blob in provider.stream()]

    with pytest.raises(ProviderError):
        asyncio.run(run())
