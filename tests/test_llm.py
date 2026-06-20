import base64

from pyris.llm import (
    ImagePart,
    TextPart,
    _encode_multipart,
    _parse_chat_response,
    _parse_transcription,
    _render_part,
)
from pyris.pipeline import _batch_parts
from tests.conftest import make_frame


def test_render_text_and_image_parts():
    assert _render_part(TextPart("hi")) == {"type": "text", "text": "hi"}
    frame = make_frame(0.0)
    rendered = _render_part(ImagePart(frame))
    assert rendered["type"] == "image_url"
    expected_b64 = base64.b64encode(frame.image).decode()
    assert rendered["image_url"]["url"] == f"data:image/jpeg;base64,{expected_b64}"


def test_parse_chat_response_and_usage():
    data = {
        "choices": [{"message": {"content": "the answer"}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4},
    }
    resp = _parse_chat_response(data)
    assert resp.text == "the answer"
    assert resp.usage.input_tokens == 12
    assert resp.usage.output_tokens == 4


def test_parse_transcription_segments():
    data = {
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "a"},
            {"start": 1.0, "end": 2.0, "text": "b"},
        ],
    }
    t = _parse_transcription(data)
    assert t.language == "en"
    assert t.full_text == "a b"


def test_parse_transcription_plaintext_fallback():
    t = _parse_transcription({"text": "just words"})
    assert len(t.segments) == 1
    assert t.full_text == "just words"


def test_batch_parts_respects_image_limit():
    parts = []
    for i in range(7):
        parts.append(TextPart(f"[{i}]"))
        parts.append(ImagePart(make_frame(float(i))))
    batches = _batch_parts(parts, max_images=3)
    counts = [sum(isinstance(p, ImagePart) for p in b) for b in batches]
    assert counts == [3, 3, 1]


def test_batch_parts_single_when_under_limit():
    parts = [ImagePart(make_frame(0.0)), ImagePart(make_frame(1.0))]
    assert len(_batch_parts(parts, max_images=20)) == 1


def test_encode_multipart_shape(tmp_path):
    f = tmp_path / "a.wav"
    f.write_bytes(b"RIFFdata")
    body, content_type = _encode_multipart({"model": "whisper"}, f)
    assert content_type.startswith("multipart/form-data; boundary=")
    assert b'name="model"' in body
    assert b'name="file"; filename="a.wav"' in body
    assert b"RIFFdata" in body
