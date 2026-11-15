import copy

import pytest

from en_gu.config import load_config
from en_gu.gate import LIKE_FOR_LIKE, REFERENCE, decide, format_report, latency_stats, slice_metrics, slice_of

GATE = load_config(["configs/base.yaml"])["gate"]
SLICES = GATE["slices"]


def card(chrf=40.0, bleu=16.0, *, slices=None, size=120.0, p95=100.0, isolated=True, mode=LIKE_FOR_LIKE):
    slices = {"short": 40.0, "medium": 40.0, "long": 40.0} if slices is None else slices
    return {
        "overall": {"bleu": bleu, "chrf": chrf, "ter": 75.0},
        "slices": {k: {"n": 300, "bleu": 10.0, "chrf": v, "enforced": True} for k, v in slices.items()},
        "size_mb": size,
        "latency_p95_ms": p95,
        "eval_isolated": isolated,
        "sentences": 2000,
        "mode": mode,
    }


def failed(verdict):
    return {c["name"] for c in verdict["checks"] if not c["passed"]}


def test_slice_of_buckets_by_word_count():
    assert slice_of("one two three", SLICES) == "short"
    assert slice_of(" ".join(["w"] * 10), SLICES) == "short"
    assert slice_of(" ".join(["w"] * 11), SLICES) == "medium"
    assert slice_of(" ".join(["w"] * 25), SLICES) == "medium"
    assert slice_of(" ".join(["w"] * 26), SLICES) == "long"
    assert slice_of(" ".join(["w"] * 500), SLICES) == "long"
    assert slice_of("", SLICES) is None


def test_slice_metrics_scores_each_bucket_separately():
    pytest.importorskip("sacrebleu")
    short, long_ = "a b c", " ".join(["word"] * 30)
    sources = [short] * 40 + [long_] * 5
    refs = ["ક ખ ગ"] * 40 + ["ઘ"] * 5
    hyps = ["ક ખ ગ"] * 40 + ["ચ"] * 5  # short perfect, long wrong
    result = slice_metrics(sources, refs, hyps, SLICES, min_sentences=30)
    assert result["short"]["chrf"] == 100.0 and result["short"]["enforced"]
    assert result["long"]["chrf"] < 10 and not result["long"]["enforced"]  # reported, but too small to enforce
    assert "medium" not in result  # empty buckets are left out


def test_latency_stats():
    stats = latency_stats(list(range(1, 101)))
    assert stats["p50_ms"] == 50.5 and stats["p95_ms"] == 95 and stats["n"] == 100


def test_first_model_only_needs_to_be_isolated_and_within_budget():
    assert decide(card(), None, GATE) == {
        "passed": True,
        "mode": "no_champion",
        "checks": decide(card(), None, GATE)["checks"],
    }
    assert failed(decide(card(isolated=False), None, GATE)) == {"eval_isolated"}
    assert failed(decide(card(size=400), None, GATE)) == {"size"}


def test_equal_or_slightly_worse_candidate_passes_but_a_real_drop_fails():
    champion = card(chrf=40.0, bleu=16.0)
    assert decide(card(chrf=40.0, bleu=16.0), champion, GATE)["passed"]
    assert decide(card(chrf=39.5, bleu=15.5), champion, GATE)["passed"]  # exactly at the tolerance
    assert failed(decide(card(chrf=39.4, bleu=16.0), champion, GATE)) == {"chrf"}
    assert failed(decide(card(chrf=40.0, bleu=15.4), champion, GATE)) == {"bleu"}
    assert decide(card(chrf=45.0, bleu=20.0), champion, GATE)["passed"]


def test_a_regression_in_one_length_bucket_blocks_the_candidate_even_if_overall_is_better():
    champion = card(chrf=40.0)
    candidate = card(chrf=41.0, slices={"short": 43.0, "medium": 41.0, "long": 37.0})
    verdict = decide(candidate, champion, GATE)
    assert not verdict["passed"] and failed(verdict) == {"slice:long"}


def test_slices_too_small_to_trust_are_not_enforced():
    champion, candidate = card(), card(slices={"short": 40.0, "medium": 40.0, "long": 10.0})
    candidate["slices"]["long"]["enforced"] = False
    assert decide(candidate, champion, GATE)["passed"]
    del champion["slices"]["long"]  # a bucket the champion has no score for is skipped too
    candidate["slices"]["long"]["enforced"] = True
    assert decide(candidate, champion, GATE)["passed"]


def test_reference_champion_uses_wider_tolerance_and_skips_slices():
    champion = {**card(chrf=40.48, bleu=16.68, mode=REFERENCE), "slices": {}}
    assert decide(card(chrf=39.2, bleu=15.8), champion, GATE)["passed"]  # within 1.5 / 1.0
    verdict = decide(card(chrf=38.9, bleu=16.0, slices={"long": 1.0}), champion, GATE)
    assert failed(verdict) == {"chrf"}
    assert verdict["mode"] == REFERENCE


def test_latency_must_stay_within_the_ratio_and_the_optional_budget():
    champion = card(p95=100.0)
    assert decide(card(p95=125.0), champion, GATE)["passed"]
    assert failed(decide(card(p95=126.0), champion, GATE)) == {"latency_ratio"}
    assert decide(card(p95=None), champion, GATE)["passed"]  # not measured: not enforced

    gcfg = copy.deepcopy(GATE)
    gcfg["max_latency_ms"] = 90
    assert "latency_budget" in failed(decide(card(p95=95.0), None, gcfg))


def test_isolation_requirement_can_be_switched_off():
    gcfg = copy.deepcopy(GATE)
    gcfg["require_isolated_candidate"] = False
    assert decide(card(isolated=False), None, gcfg)["passed"]


def test_report_lists_every_check():
    text = format_report({**decide(card(chrf=30.0), card(), GATE)})
    assert "REJECTED" in text and "[FAIL] chrf" in text and "[ok] size" in text
