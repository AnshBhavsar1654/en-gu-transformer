"""Phase 4 rollout helpers: the canary decision, cold-start detection, and both tools against a real server."""

import json

import pytest

from en_gu.canary import Probe, Thresholds, evaluate, percentile, run_canary
from en_gu.cli import main as cli_main
from en_gu.coldstart import is_cold, measure, report

pytest.importorskip("fastapi")
from tests.test_smoke import free_port, start_server, stop, wait_until_ready  # noqa: E402


def probes(latencies, errors=0):
    return [Probe(True, ms, 200) for ms in latencies] + [Probe(False, 30000, 503, "down")] * errors


def test_percentile_is_nearest_rank():
    assert percentile([], 95) is None
    assert percentile([5], 95) == 5
    assert percentile(list(range(1, 101)), 95) == 95
    assert percentile([300, 100, 200], 50) == 200


def test_healthy_canary_passes():
    verdict = evaluate(probes([200] * 30), probes([210] * 30))
    assert verdict.ok and verdict.reasons == []
    assert verdict.candidate.requests == 30 and verdict.candidate.p95_ms == 200


def test_errors_above_the_limit_fail_the_canary():
    verdict = evaluate(probes([200] * 28, errors=2), probes([200] * 30), Thresholds(max_error_rate=0.05))
    assert not verdict.ok and "error rate 6.7%" in verdict.reasons[0]
    # failed probes don't count towards latency
    assert verdict.candidate.p95_ms == 200


def test_latency_regression_against_the_stable_revision_fails_the_canary():
    verdict = evaluate(probes([1000] * 30), probes([300] * 30))
    assert not verdict.ok and "p95 1000 ms is above 550 ms" in verdict.reasons[0]


def test_small_absolute_differences_are_tolerated():
    # 2x slower but only +100 ms: within the slack, so noise on fast requests doesn't block a rollout
    assert evaluate(probes([200] * 30), probes([100] * 30)).ok


def test_too_few_probes_is_not_evidence():
    verdict = evaluate(probes([200] * 3))
    assert not verdict.ok and "only 3 probes" in verdict.reasons[0]


def test_absolute_ceiling_without_a_baseline():
    assert evaluate(probes([400] * 20), thresholds=Thresholds(max_p95_ms=500)).ok
    assert not evaluate(probes([600] * 20), thresholds=Thresholds(max_p95_ms=500)).ok


def test_cold_start_detection():
    seen = set()
    assert is_cold(12.0, 11.5, "a", seen)  # started while we waited
    assert not is_cold(0.3, 900.0, "a", seen)  # long-running instance
    assert not is_cold(12.0, 11.5, "a", {"a"})  # already seen: warm
    assert not is_cold(12.0, None, None, seen)  # no /info: can't tell


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    from en_gu.tinymodel import make_tiny_checkpoint

    tmp = tmp_path_factory.mktemp("rollout")
    proc, url = start_server(make_tiny_checkpoint(tmp / "best.pt"), tmp / "feedback.db")
    try:
        wait_until_ready(url, timeout=90, interval=0.5)
    except AssertionError:
        stop(proc)
        pytest.fail(f"server did not start: {proc.stderr.read().decode(errors='replace')[-2000:]}")
    yield url
    stop(proc)


@pytest.mark.smoke
def test_canary_holds_against_a_healthy_server(server, monkeypatch, tmp_path):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    lines = []
    few = Thresholds(min_requests=3)  # a 3-second window: how many probes fit depends on the machine
    verdict = run_canary(server, server, minutes=0.05, interval=0.2, beam_size=2, thresholds=few, out=lines.append)
    assert verdict.ok, "\n".join(lines)
    assert verdict.candidate.requests >= 3 and verdict.baseline.requests == verdict.candidate.requests
    assert lines[-1] == "PASS  canary held"
    assert "Canary (0.05 min): held" in summary.read_text(encoding="utf-8")


@pytest.mark.smoke
def test_canary_fails_when_the_new_revision_is_down(server, capsys):
    dead = f"http://127.0.0.1:{free_port()}"
    args = ["canary", "--url", dead, "--baseline-url", server, "--minutes", "0.02", "--interval", "0.1"]
    assert cli_main(args + ["--beam-size", "2", "--min-requests", "1"]) == 1
    assert "FAIL  error rate 100.0%" in capsys.readouterr().out


@pytest.mark.smoke
def test_cold_start_against_a_fresh_server(tmp_path):
    from en_gu.tinymodel import make_tiny_checkpoint

    proc, url = start_server(make_tiny_checkpoint(tmp_path / "best.pt"), tmp_path / "feedback.db")
    try:
        lines = []
        # no waiting: the first request goes to a process that is still starting, so it is a cold start
        samples = measure(url, samples=2, wait_first=False, timeout=90, out=lines.append, sleep=lambda s: None)
        assert [s.cold for s in samples] == [True, False], "\n".join(lines)
        assert samples[0].status == 200 and samples[0].model_load_s is not None
        assert lines[-1].startswith("cold starts: 1/2")
    finally:
        stop(proc)


def test_cold_start_report_without_a_cold_sample():
    from en_gu.coldstart import Sample

    warm = Sample(0.2, 200, "a", 900.0, 3.1, "v1", "rev-1", False)
    assert report([warm])[-1].startswith("no cold start observed")
    assert json.dumps(warm.__dict__)  # serialisable for --json
