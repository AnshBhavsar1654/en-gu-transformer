import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from en_gu.api import create_app  # noqa: E402
from en_gu.serving import Translator, serve_settings  # noqa: E402


@pytest.fixture
def translator(serve_cfg):
    return Translator.load(serve_cfg, serve_settings(serve_cfg, {}))


@pytest.fixture
def client(serve_cfg, translator):
    with TestClient(create_app(serve_cfg, translator=translator)) as c:
        yield c


def test_translate_returns_the_translation_and_its_model_version(client, translator):
    r = client.post("/translate", json={"text": "The train is late. A big house.", "beam_size": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["n_sentences"] == 2 and isinstance(body["translation"], str)
    assert body["model_version"] == translator.version == r.headers["X-Model-Version"]
    assert body["request_id"] == r.headers["X-Request-ID"] and body["latency_ms"] >= 0


def test_default_beam_and_incoming_request_id(client):
    r = client.post("/translate", json={"text": "Hello."}, headers={"X-Request-ID": "abc-123"})
    assert r.status_code == 200 and r.json()["request_id"] == "abc-123"
    r = client.post("/translate", json={"text": "Hello."}, headers={"X-Request-ID": "bad id with spaces"})
    assert r.json()["request_id"] != "bad id with spaces"


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"text": "   "}, "empty"),
        ({"text": "a" * 2001}, "2001 characters"),
        ({"text": "Hi. " * 41}, "41 sentences"),
        ({"text": "Hi.", "beam_size": 0}, "beam_size"),
        ({"text": "Hi.", "beam_size": 9}, "beam_size"),
    ],
)
def test_invalid_input_is_rejected_with_a_readable_reason(client, payload, message):
    r = client.post("/translate", json=payload)
    assert r.status_code == 422 and message in r.json()["detail"]


def test_unknown_fields_and_wrong_types_are_rejected(client):
    assert client.post("/translate", json={"text": "Hi.", "beam": 3}).status_code == 422
    assert client.post("/translate", json={"text": 5}).status_code == 422


def test_health_ready_and_info(client, translator):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"ready": True, "model_version": translator.version}
    info = client.get("/info").json()
    assert info["limits"]["max_chars"] == 2000 and info["parameters"] > 0


def test_not_ready_until_the_model_has_loaded(serve_cfg, monkeypatch):
    release = __import__("threading").Event()
    real_load = Translator.load

    def slow_load(cfg, settings=None):
        release.wait(10)
        return real_load(cfg, settings)

    monkeypatch.setattr(Translator, "load", staticmethod(slow_load))
    with TestClient(create_app(serve_cfg)) as c:
        assert c.get("/health").status_code == 200
        r = c.get("/ready")
        assert r.status_code == 503 and r.json()["loading"] is True
        r = c.post("/translate", json={"text": "Hi."})
        assert r.status_code == 503 and r.headers["Retry-After"] == "5"
        release.set()
        for _ in range(100):
            if c.get("/ready").status_code == 200:
                break
            time.sleep(0.05)
        assert c.get("/ready").status_code == 200
        assert "X-Model-Version" in c.post("/translate", json={"text": "Hi."}).headers


def test_a_model_that_fails_to_load_reports_not_ready_instead_of_crashing(serve_cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("EN_GU_MODEL_PATH", str(tmp_path / "missing.pt"))
    with TestClient(create_app(serve_cfg, background=False)) as c:
        r = c.get("/ready")
        assert r.status_code == 503 and "missing.pt" in r.json()["error"]
        assert c.get("/health").status_code == 200
        assert "missing.pt" in c.post("/translate", json={"text": "Hi."}).json()["detail"]


def test_thumbs_feedback_stores_no_text(client, serve_cfg):
    from en_gu.feedback import FeedbackStore

    tr = client.post("/translate", json={"text": "The train is late."}).json()
    r = client.post(
        "/feedback",
        json={"request_id": tr["request_id"], "rating": "up", "source_text": "The train is late.", "translation": "x"},
    )
    assert r.status_code == 201 and r.json()["stored_text"] is False
    (row,) = FeedbackStore(serve_cfg["serve"]["feedback_db"]).rows()
    assert row["rating"] == "up" and row["model_version"] == tr["model_version"]
    assert row["source_text"] is None and row["translation"] is None and row["corrected_translation"] is None


def test_a_correction_stores_the_text_it_corrects(client, serve_cfg):
    from en_gu.feedback import FeedbackStore

    payload = {
        "request_id": "r1",
        "rating": "down",
        "model_version": "v9",
        "source_text": "Good morning.",
        "translation": "ખરાબ",
        "corrected_translation": " સુપ્રભાત. ",
    }
    assert client.post("/feedback", json=payload).json()["stored_text"] is True
    store = FeedbackStore(serve_cfg["serve"]["feedback_db"])
    (row,) = store.rows()
    assert row["corrected_translation"] == "સુપ્રભાત." and row["source_text"] == "Good morning."
    assert store.summary() == {"v9": {"up": 0, "down": 1, "corrections": 1}}


@pytest.mark.parametrize(
    "payload",
    [
        {"request_id": "r1", "rating": "meh"},
        {"request_id": "has spaces", "rating": "up"},
        {"request_id": "r1", "rating": "up", "corrected_translation": "x" * 4001},
        {"request_id": "r1", "rating": "up", "extra": 1},
    ],
)
def test_invalid_feedback_is_rejected(client, payload):
    assert client.post("/feedback", json=payload).status_code == 422


def test_old_feedback_is_deleted_after_the_retention_period(tmp_path):
    import sqlite3

    from en_gu.feedback import FeedbackStore

    store = FeedbackStore(tmp_path / "f.db", retention_days=30)
    store.add("old", "up")
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE feedback SET created_at = '2000-01-01T00:00:00+00:00'")
    store.add("new", "down")
    assert [r["request_id"] for r in store.rows()] == ["new"]


def test_metrics_count_requests_translations_and_feedback(client, translator):
    client.post("/translate", json={"text": "The train is late. A big house.", "beam_size": 2})
    client.post("/translate", json={"text": ""})
    client.post("/feedback", json={"request_id": "r1", "rating": "down", "corrected_translation": "x"})
    client.get("/no-such-page")
    r = client.get("/metrics")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    text = r.text
    assert 'en_gu_http_requests_total{method="POST",path="/translate",status="200"} 1.0' in text
    assert 'en_gu_http_requests_total{method="POST",path="/translate",status="422"} 1.0' in text
    assert 'path="unmatched",status="404"' in text and "/no-such-page" not in text
    assert 'en_gu_translate_seconds_count{beam_size="2"} 1.0' in text
    assert "en_gu_translate_sentences_sum 2.0" in text
    assert 'en_gu_feedback_total{rating="down",with_correction="true"} 1.0' in text
    assert f'en_gu_model_info{{device="cpu",source="path",version="{translator.version}"}} 1.0' in text
    assert "en_gu_model_ready 1.0" in text
