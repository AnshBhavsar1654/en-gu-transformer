import json

import pytest
import torch

from en_gu.model import EOS_IDX, PAD_IDX, beam_search, greedy_decode
from en_gu.release import shadow_report, traffic_report
from en_gu.reviewed import validate_pairs


@pytest.mark.parametrize("decode", [greedy_decode, beam_search])
def test_kv_cache_preserves_decoding_and_scores(tiny_model, decode):
    source = torch.tensor([[7, 17, 13, EOS_IDX], [8, 15, EOS_IDX, PAD_IDX]])
    full = decode(tiny_model, source, detailed=True)
    tiny_model.use_kv_cache = True
    cached = decode(tiny_model, source, detailed=True)
    assert [result.ids for result in cached] == [result.ids for result in full]
    assert [result.hit_length_limit for result in cached] == [result.hit_length_limit for result in full]
    assert [result.mean_log_probability for result in cached] == pytest.approx(
        [result.mean_log_probability for result in full], abs=1e-5
    )


def test_reviewed_pairs_are_normalized_and_kept_out_of_eval(tmp_path):
    pairs = tmp_path / "reviewed.jsonl"
    rows = [
        {"source": "  Tell   me more. ", "target": "મને વધુ કહો."},
        {"source": "tell me more.", "target": "મને વધુ જણાવો."},
        {"source": "A frozen source.", "target": "હું ફસાઈ ગયો છું."},
        {"source": "Another source.", "target": "આ પરીક્ષા છે."},
        {"source": "This is valid English.", "target": "This is still English."},
    ]
    pairs.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    frozen = ((["a frozen source."], ["આ પરીક્ષા છે."]), ([], []))
    accepted, rejected = validate_pairs(pairs, frozen, 80)
    assert accepted == [{"source": "Tell me more.", "target": "મને વધુ કહો."}]
    assert [row["line"] for row in rejected] == [2, 3, 4, 5]


def test_shadow_reports_reject_wrong_versions_dropped_samples_and_low_similarity():
    event = {
        "event": "shadow_comparison",
        "stable_version": "v1",
        "candidate_version": "v2",
        "stable_latency_ms": 10,
        "candidate_latency_ms": 8,
        "similarity": 0.85,
        "ok": True,
    }
    records = [{"jsonPayload": event}] * 3
    assert shadow_report(records, "v1", "v2", 3, 0.5)["ok"]
    assert not shadow_report(records, "v1", "v3", 3, 0.5)["ok"]
    assert not shadow_report(records, "v1", "v2", 3, 0.9)["ok"]
    assert not shadow_report([*records, {**event, "event": "shadow_dropped", "ok": False}], "v1", "v2", 3, 0.5)["ok"]


def test_cloud_traffic_report_uses_both_revisions_and_excludes_probes():
    def request(revision, status=200, agent="client/1.0"):
        return {
            "resource": {"labels": {"revision_name": revision}},
            "httpRequest": {"status": status, "latency": "0.1s", "userAgent": agent},
        }

    entries = [request("stable"), request("new"), request("new", 500, "en-gu-canary")]
    assert traffic_report(entries, "stable", "new", 1)["ok"]
    assert not traffic_report(entries, "stable", "new", 2)["ok"]
    assert not traffic_report([*entries, request("new", 429)], "stable", "new", 1)["ok"]
