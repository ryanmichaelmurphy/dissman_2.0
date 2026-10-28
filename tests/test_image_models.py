import base64
from types import SimpleNamespace

import httpx
import openai
import pytest

from image_models import (
    DEFAULT_MODELS, EDIT_PARAMS, ModelRoster, classify_error, describe_error,
    edit_image, load_models,
)

REQ = httpx.Request("POST", "https://api.openai.com/v1/images/edits")


def _status_error(cls, status, body, request_id="req_123"):
    resp = httpx.Response(status, request=REQ, headers={"x-request-id": request_id})
    return cls("boom", response=resp, body=body)


# --- load_models -------------------------------------------------------------

def test_load_models_reads_order_skipping_comments_and_blanks(tmp_path):
    f = tmp_path / "models.txt"
    f.write_text("# header\n\nmodel-a\n  model-b  \n# model-c\nmodel-a  # inline\n")
    assert load_models(f) == ["model-a", "model-b"]


def test_load_models_missing_file_returns_defaults(tmp_path):
    assert load_models(tmp_path / "nope.txt") == DEFAULT_MODELS


def test_load_models_empty_file_returns_defaults(tmp_path):
    f = tmp_path / "models.txt"
    f.write_text("# only comments\n\n")
    assert load_models(f) == DEFAULT_MODELS


def test_load_models_returns_a_copy_of_defaults(tmp_path):
    out = load_models(tmp_path / "nope.txt")
    out.append("x")
    assert "x" not in DEFAULT_MODELS


def test_quality_is_low():
    assert EDIT_PARAMS["quality"] == "low"


# --- classify_error ----------------------------------------------------------

def test_moderation_blocked_is_content():
    e = _status_error(openai.BadRequestError, 400,
                      {"message": "blocked", "code": "moderation_blocked"})
    assert classify_error(e) == "content"


def test_moderation_blocked_nested_body_is_content():
    e = _status_error(openai.BadRequestError, 400,
                      {"error": {"message": "x", "code": "moderation_blocked"}})
    assert classify_error(e) == "content"


def test_safety_message_is_content():
    e = _status_error(openai.BadRequestError, 400,
                      {"message": "rejected by the safety system", "code": None})
    assert classify_error(e) == "content"


def test_other_bad_request_is_model():
    e = _status_error(openai.BadRequestError, 400,
                      {"message": "Invalid value: 'low'", "param": "quality",
                       "code": "invalid_value"})
    assert classify_error(e) == "model"


def test_not_found_is_model():
    e = _status_error(openai.NotFoundError, 404,
                      {"message": "The model `gpt-image-1` does not exist",
                       "code": "model_not_found"})
    assert classify_error(e) == "model"


def test_permission_denied_is_model():
    e = _status_error(openai.PermissionDeniedError, 403,
                      {"message": "organization must be verified"})
    assert classify_error(e) == "model"


@pytest.mark.parametrize("cls,status", [
    (openai.AuthenticationError, 401),
    (openai.RateLimitError, 429),
    (openai.InternalServerError, 500),
])
def test_account_and_server_errors_are_fatal(cls, status):
    assert classify_error(_status_error(cls, status, {"message": "x"})) == "fatal"


def test_connection_and_timeout_are_fatal():
    assert classify_error(openai.APIConnectionError(request=REQ)) == "fatal"
    assert classify_error(openai.APITimeoutError(request=REQ)) == "fatal"


def test_non_openai_exception_is_fatal():
    assert classify_error(ValueError("no image data")) == "fatal"


# --- describe_error ----------------------------------------------------------

def test_describe_status_error_has_type_status_request_id_code():
    e = _status_error(openai.NotFoundError, 404,
                      {"message": "gone", "code": "model_not_found"}, request_id="req_abc")
    s = describe_error(e)
    assert "NotFoundError" in s
    assert "status=404" in s
    assert "request_id=req_abc" in s
    assert "code=model_not_found" in s
    assert "gone" in s


def test_describe_connection_error_includes_cause():
    try:
        try:
            raise OSError("Name or service not known")
        except OSError as inner:
            raise openai.APIConnectionError(request=REQ) from inner
    except openai.APIConnectionError as e:
        s = describe_error(e)
    assert "APIConnectionError" in s
    assert "Name or service not known" in s


def test_describe_plain_exception():
    assert "ValueError: bad" in describe_error(ValueError("bad"))


# --- ModelRoster -------------------------------------------------------------

def test_roster_alive_in_order():
    assert ModelRoster(["a", "b", "c"]).alive() == ["a", "b", "c"]


def test_roster_mark_dead_removes():
    r = ModelRoster(["a", "b", "c"])
    r.mark_dead("a")
    assert r.alive() == ["b", "c"]


def test_roster_mark_good_moves_to_front():
    r = ModelRoster(["a", "b", "c"])
    r.mark_good("c")
    assert r.alive() == ["c", "a", "b"]


def test_roster_good_model_can_be_marked_dead_later():
    r = ModelRoster(["a", "b"])
    r.mark_good("b")
    r.mark_dead("b")
    assert r.alive() == ["a"]


def test_roster_all_dead_is_empty():
    r = ModelRoster(["a"])
    r.mark_dead("a")
    assert r.alive() == []


def test_roster_dedupes_and_ignores_unknown():
    r = ModelRoster(["a", "a", "b"])
    r.mark_good("zzz")
    r.mark_dead("zzz")
    assert r.alive() == ["a", "b"]


# --- edit_image --------------------------------------------------------------

class FakeImages:
    def __init__(self, b64):
        self.b64 = b64
        self.calls = []

    def edit(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(b64_json=self.b64)],
                               usage={"total_tokens": 42})


def _fake_client(b64):
    return SimpleNamespace(images=FakeImages(b64))


def test_edit_image_passes_params_and_decodes(tmp_path):
    src = tmp_path / "in.png"
    src.write_bytes(b"photo")
    client = _fake_client(base64.b64encode(b"PNGDATA").decode())
    data, usage = edit_image(client, "gpt-image-2.5-flare", src, "draw it")
    assert data == b"PNGDATA"
    assert usage == {"total_tokens": 42}
    call = client.images.calls[0]
    assert call["model"] == "gpt-image-2.5-flare"
    assert call["prompt"] == "draw it"
    for k, v in EDIT_PARAMS.items():
        assert call[k] == v


def test_edit_image_missing_data_raises_value_error(tmp_path):
    src = tmp_path / "in.png"
    src.write_bytes(b"photo")
    with pytest.raises(ValueError):
        edit_image(_fake_client(None), "m", src, "p")
