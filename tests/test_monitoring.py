import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from en_gu.api import create_app
from en_gu.drift import TESTS, read_events, run_report
from en_gu.model import beam_search, greedy_decode
from en_gu.quality_canary import golden_pairs
from en_gu.serving import Translator

EVAL_DIR = Path(__file__).resolve().parents[1] / "data/eval"


def test_translation_telemetry_omits_text_and_uses_stable_keyed_hash(serve_cfg, monkeypatch, capsys):
    monkeypatch.setenv("EN_GU_TELEMETRY_HASH_KEY", "test-key")
    app = create_app(serve_cfg, background=False)
    source = "This private example has 172 characters? Certainly not."
    with TestClient(app) as client:
        for beam in (1, 2):
            response = client.post("/translate", json={"text": source, "beam_size": beam})
            assert response.status_code == 200
            assert response.headers["X-Model-Revision"] == "local"
        assert (
            client.post(
                "/feedback",
                json={
                    "request_id": "has-private-content",
                    "rating": "down",
                    "corrected_translation": "this is a private correction",
                },
            ).status_code
            == 201
        )
    records = [line for line in capsys.readouterr().out.splitlines() if line.startswith('{"event":')]
    assert all(source not in line and "private" not in line for line in records)
    first, second = [event for line in records if (event := json.loads(line))["event"] == "translation"]
    assert first["input_hash"] == second["input_hash"]
    assert len(first["input_hash"]) == 64
    assert first["n_sentences"] == 2
    assert first["input_chars"] == len(source)
    assert first["input_tokens"] > 0
    assert first["mean_log_probability"] <= 0


@pytest.mark.parametrize("decode", [greedy_decode, beam_search])
def test_observed_decoding_preserves_tokens_and_detects_length_limit(tiny_model, decode, monkeypatch):
    import torch

    tiny_model.eval()
    original = tiny_model.decode

    def without_eos(*args):
        logits = original(*args)
        logits[:, :, 3] = -torch.inf
        return logits

    monkeypatch.setattr(tiny_model, "decode", without_eos)
    source = torch.tensor([[10, 11, 3]])
    actual = decode(tiny_model, source, max_len_a=0, max_len_b=3, detailed=True)[0]
    assert actual.ids == decode(tiny_model, source, max_len_a=0, max_len_b=3)[0]
    assert actual.hit_length_limit
    assert len(actual.ids) == 3
    assert actual.mean_log_probability < 0


def test_observing_translation_preserves_text_and_paragraphs(serve_cfg):
    translator = Translator.load(serve_cfg)
    text = "The train is late. What is your name?\n\nA big house."
    for beam in (1, 3):
        baseline = translator.translate_text(text, beam_size=beam)
        observed = translator.translate_text(text, beam_size=beam, observe=True)
        assert (observed.translation, observed.n_sentences) == (baseline.translation, baseline.n_sentences)
        assert observed.features["n_sentences"] == 3


def test_drift_excludes_canary_and_requires_two_shifted_windows(tmp_path, monkeypatch):
    pytest.importorskip("evidently")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    now = datetime(2026, 1, 3, 12, tzinfo=timezone.utc)
    reference = tmp_path / "reference.json"
    event_fields = dict.fromkeys(TESTS, 0)
    rows = [{**event_fields, "input_chars": i + 25, "input_tokens": i + 5, "n_sentences": 1} for i in range(40)]
    reference.write_text(json.dumps({"schema": 1, "rows": rows}), encoding="utf-8")

    def shifted(hours_ago, traffic):
        return [
            {
                "schema": 1,
                "event": "translation",
                "timestamp": (now - timedelta(hours=hours_ago)).isoformat(),
                "traffic": traffic,
                **row,
                "input_chars": row["input_chars"] + 800,
                "input_tokens": row["input_tokens"] + 80,
            }
            for row in rows
        ]

    logs = tmp_path / "logs.json"
    logs.write_text(json.dumps([{"jsonPayload": row} for row in shifted(1, "canary")]), encoding="utf-8")
    out = tmp_path / "drift"
    result = run_report(logs, reference, out, minimum=20, now=now)
    assert result["windows"]["24h"]["samples"] == 0
    assert not result["alert"]

    current, previous = shifted(1, "simulated"), shifted(25, "simulated")
    logs.write_text("\n".join(map(json.dumps, current)), encoding="utf-8")
    assert not run_report(logs, reference, out, minimum=20, now=now, include_simulated=True)["alert"]

    logs.write_text("\n".join(map(json.dumps, [*current, *previous])), encoding="utf-8")
    result = run_report(logs, reference, out, minimum=20, now=now, include_simulated=True)
    assert result["alert"]
    assert result["windows"]["24h"]["features"]["input_chars"]["drifted"]
    assert "en_gu_drift_persistent 1.0" in out.with_suffix(".prom").read_text(encoding="utf-8")
    assert (tmp_path / "drift-24h.html").stat().st_size > 1000


def test_golden_sample_is_fixed_and_parallel():
    pairs = golden_pairs(EVAL_DIR)
    assert len(pairs) == 200
    assert len({source for source, _ in pairs}) == len(pairs)
    assert all(source and target for source, target in pairs)


def test_cloud_log_timestamp_can_live_outside_json_payload(tmp_path):
    logs = tmp_path / "cloud.json"
    payload = {"event": "translation", "schema": 1, "traffic": "regular"}
    logs.write_text(
        json.dumps([{"timestamp": "2026-01-03T12:00:00.123456789Z", "jsonPayload": payload}]),
        encoding="utf-8",
    )
    assert list(read_events(logs)) == [(datetime(2026, 1, 3, 12, 0, 0, 123456, tzinfo=timezone.utc), payload)]
