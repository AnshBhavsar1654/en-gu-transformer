import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from fastapi.testclient import TestClient

from en_gu.api import create_app
from en_gu.model import EOS_IDX, beam_search, greedy_decode
from en_gu.serving import TranslationResult, Translator


def test_request_body_limits_cover_declared_and_streaming_requests(serve_cfg):
    serve_cfg["serve"]["guardrails"].update(max_translate_body_bytes=24, max_feedback_body_bytes=64)
    with TestClient(create_app(serve_cfg, background=False)) as client:
        declared = client.post("/translate", json={"text": "x" * 30})
        assert declared.status_code == 413
        streamed = client.post("/translate", content=iter([b"x" * 15, b"y" * 15]))
        assert streamed.status_code == 413
        assert "24-byte" in streamed.json()["detail"]
        assert client.post("/translate", json={"text": "Hi."}).status_code == 200
        assert client.post("/feedback", json={"request_id": "valid", "rating": "up"}).status_code == 201
        assert 'path="/translate",status="413"} 2.0' in client.get("/metrics").text


def test_rate_limit_has_a_retry_hint_and_separate_budgets(serve_cfg):
    serve_cfg["serve"]["guardrails"].update(translate_per_minute=1, translate_burst=1)
    with TestClient(create_app(serve_cfg, background=False)) as client:
        assert client.post("/translate", json={"text": "Hi.", "beam_size": 1}).status_code == 200
        throttled = client.post("/translate", json={"text": "Hello."})
        assert throttled.status_code == 429
        assert int(throttled.headers["Retry-After"]) > 0
        assert client.post("/feedback", json={"request_id": "valid", "rating": "up"}).status_code == 201


def test_feedback_accepts_escaped_unicode_at_the_character_limit(serve_cfg):
    text = "\U0001f600" * 4000
    body = json.dumps(
        {
            "request_id": "escaped",
            "rating": "down",
            "source_text": text,
            "translation": text,
            "corrected_translation": text,
        },
        ensure_ascii=True,
    )
    with TestClient(create_app(serve_cfg, background=False)) as client:
        response = client.post("/feedback", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 201
    assert response.json()["stored_text"] is True


def test_timeout_keeps_running_work_admitted_until_the_worker_finishes(serve_cfg, monkeypatch):
    serve_cfg["serve"]["guardrails"].update(max_queued_translations=0, translation_timeout_seconds=0.25)
    translator = Translator.load(serve_cfg)
    started, finish = threading.Event(), threading.Event()

    def slow_translate(*args, **kwargs):
        started.set()
        if not finish.wait(5):
            raise RuntimeError("test did not release the worker")
        return TranslationResult("Hello", 1, 1.0)

    monkeypatch.setattr(translator, "translate_text", slow_translate)
    with TestClient(create_app(serve_cfg, translator=translator)) as client, ThreadPoolExecutor(1) as actors:
        try:
            request = actors.submit(client.post, "/translate", json={"text": "This request waits."})
            assert started.wait(5)
            assert client.get("/health").status_code == 200
            assert client.post("/translate", json={"text": "The queue is full."}).status_code == 503
            assert request.result(timeout=5).status_code == 504
            overloaded = client.post("/translate", json={"text": "The worker is still running."})
            assert overloaded.status_code == 503
            assert overloaded.headers["Retry-After"] == "1"
            assert client.get("/ready").json()["ready"]
        finally:
            finish.set()


def test_queued_work_is_cancelled_when_its_deadline_passes(serve_cfg, monkeypatch):
    serve_cfg["serve"]["guardrails"].update(max_queued_translations=1, translation_timeout_seconds=0.25)
    translator = Translator.load(serve_cfg)
    started, finish = threading.Event(), threading.Event()
    calls = []

    def translate(text, **kwargs):
        calls.append(text)
        if text == "first":
            started.set()
            if not finish.wait(5):
                raise RuntimeError("test did not release the worker")
        return TranslationResult("Hello", 1, 1.0)

    monkeypatch.setattr(translator, "translate_text", translate)
    with TestClient(create_app(serve_cfg, translator=translator)) as client, ThreadPoolExecutor(1) as actors:
        try:
            request = actors.submit(client.post, "/translate", json={"text": "first"})
            assert started.wait(5)
            assert client.post("/translate", json={"text": "queued"}).status_code == 504
            assert request.result(timeout=5).status_code == 504
        finally:
            finish.set()
        assert client.post("/translate", json={"text": "next"}).status_code == 200
    assert calls == ["first", "next"]


def test_corrupt_checkpoint_keeps_health_available(serve_cfg, tmp_path, monkeypatch):
    broken = tmp_path / "bad.pt"
    broken.write_bytes(b"broken checkpoint")
    monkeypatch.setenv("EN_GU_MODEL_PATH", str(broken))
    with TestClient(create_app(serve_cfg, background=False)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").json()["ready"] is False
        assert client.post("/translate", json={"text": "Hello."}).status_code == 503


@pytest.mark.parametrize("cached", [False, True])
def test_single_beam_is_greedy_even_when_detailed(tiny_model, cached):
    source = torch.tensor([[7, 6, EOS_IDX], [6, 9, EOS_IDX]])
    tiny_model.use_kv_cache = cached
    assert beam_search(tiny_model, source, beam_size=1, detailed=True) == greedy_decode(
        tiny_model, source, detailed=True
    )


@pytest.mark.parametrize(("source", "target"), [("Hello.", "નમસ્તે."), ("Thanks.", "આભાર.")])
def test_exact_golden_api_translation(serve_cfg, monkeypatch, source, target):
    translator = Translator.load(serve_cfg)
    output = translator.sp.encode(target) + [EOS_IDX]

    def decode(prefix, memory, mask):
        logits = torch.full((prefix.size(0), prefix.size(1), translator.model.embedding.num_embeddings), -10.0)
        logits[:, -1, output[min(prefix.size(1) - 1, len(output) - 1)]] = 10.0
        return logits

    monkeypatch.setattr(translator.model, "decode", decode)
    with TestClient(create_app(serve_cfg, translator=translator)) as client:
        response = client.post("/translate", json={"text": f"  {source}  ", "beam_size": 1})
        assert response.status_code == 200
        assert response.json()["translation"] == target
        assert response.headers["X-Model-Version"] == translator.version
