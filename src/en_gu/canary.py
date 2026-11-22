"""Canary check for a progressive rollout: probe the new revision (and the one it replaces) for a fixed window.

While a new Cloud Run revision gets a small share of production traffic, this sends a steady trickle of real
translations to it through its tag URL (and, if given, to the stable revision's tag URL for comparison). The canary
passes when the new revision's error rate stays under a limit and its p95 latency is not clearly worse than the
stable revision's. Uses only the standard library, so it runs on a bare CI runner:

    PYTHONPATH=src python -m en_gu.canary --url https://candidate---en-gu-api-xyz.run.app \\
        --baseline-url https://stable---en-gu-api-xyz.run.app --minutes 5

Exits 0 when the canary holds and 1 otherwise. The probes are synthetic traffic, labelled as such by their
User-Agent and request ids (`canary-...`), so they can be filtered out of real-traffic statistics.
"""

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import uuid
from dataclasses import dataclass, field

from en_gu.smoke import request

# Short everyday sentences of different lengths: enough to exercise beam search without slowing the window down
CANARY_SENTENCES = [
    "The train is late.",
    "What is your name?",
    "I will call you tomorrow morning.",
    "The children are playing in the garden.",
    "Please close the door when you leave.",
    "Farmers in the village are waiting for the rain to start.",
    "She bought vegetables and fruit at the market yesterday.",
    "The government announced a new scheme for students today.",
]


@dataclass
class Probe:
    ok: bool
    latency_ms: float
    status: int | None = None  # None = no HTTP response (timeout, connection error)
    error: str = ""


@dataclass
class Stats:
    requests: int
    errors: int
    error_rate: float
    p50_ms: float | None
    p95_ms: float | None

    def line(self):
        p50 = "n/a" if self.p50_ms is None else f"{self.p50_ms:.0f} ms"
        p95 = "n/a" if self.p95_ms is None else f"{self.p95_ms:.0f} ms"
        return f"{self.requests} requests, {self.errors} errors ({self.error_rate:.1%}), p50 {p50}, p95 {p95}"


@dataclass
class Thresholds:
    max_error_rate: float = 0.02  # share of failed probes allowed on the new revision
    max_p95_ratio: float = 1.5  # new p95 may be up to this many times the stable p95 ...
    p95_slack_ms: float = 250.0  # ... or this much slower, whichever is more lenient (small numbers are noisy)
    max_p95_ms: float | None = None  # absolute ceiling, when there is no baseline to compare with
    min_requests: int = 10  # fewer probes than this is not evidence of anything


@dataclass
class Verdict:
    ok: bool
    candidate: Stats
    baseline: Stats | None
    reasons: list = field(default_factory=list)


def percentile(values, pct):
    """Nearest-rank percentile (no interpolation, so it is always a latency that was actually observed)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def summarize(probes):
    """Error rate over every probe; latency percentiles over the successful ones only."""
    latencies = [p.latency_ms for p in probes if p.ok]
    errors = sum(not p.ok for p in probes)
    return Stats(
        requests=len(probes),
        errors=errors,
        error_rate=errors / len(probes) if probes else 0.0,
        p50_ms=percentile(latencies, 50),
        p95_ms=percentile(latencies, 95),
    )


def evaluate(candidate, baseline=None, thresholds=None):
    """Decide whether the new revision may take all traffic. `candidate` and `baseline` are lists of Probe."""
    t = thresholds or Thresholds()
    cand = summarize(candidate)
    base = summarize(baseline) if baseline else None
    reasons = []
    if cand.requests < t.min_requests:
        reasons.append(f"only {cand.requests} probes reached the new revision (need {t.min_requests})")
    if cand.error_rate > t.max_error_rate:
        reasons.append(f"error rate {cand.error_rate:.1%} is above {t.max_error_rate:.1%}")
    if cand.p95_ms is not None:
        if base is not None and base.p95_ms is not None:
            limit = max(base.p95_ms * t.max_p95_ratio, base.p95_ms + t.p95_slack_ms)
            if cand.p95_ms > limit:
                reasons.append(
                    f"p95 {cand.p95_ms:.0f} ms is above {limit:.0f} ms "
                    f"(stable p95 {base.p95_ms:.0f} ms x{t.max_p95_ratio} or +{t.p95_slack_ms:.0f} ms)"
                )
        if t.max_p95_ms is not None and cand.p95_ms > t.max_p95_ms:
            reasons.append(f"p95 {cand.p95_ms:.0f} ms is above the ceiling of {t.max_p95_ms:.0f} ms")
    return Verdict(not reasons, cand, base, reasons)


def probe(base_url, sentence, beam_size=5, timeout=30):
    """One translation request; a non-200 answer, a body without a translation or a network error is a failure."""
    start = time.perf_counter()
    rid = f"canary-{uuid.uuid4().hex[:12]}"
    try:
        status, _, body = request(
            base_url,
            "POST",
            "/translate",
            {"text": sentence, "beam_size": beam_size},
            {"X-Request-ID": rid, "User-Agent": "en-gu-canary"},
            timeout=timeout,
        )
    except (urllib.error.URLError, OSError) as exc:
        return Probe(False, (time.perf_counter() - start) * 1000, None, f"{type(exc).__name__}: {exc}")
    latency_ms = (time.perf_counter() - start) * 1000
    if status != 200:
        return Probe(False, latency_ms, status, body[:200])
    try:
        translation = json.loads(body).get("translation")
    except (ValueError, AttributeError):
        translation = None
    if not isinstance(translation, str):
        return Probe(False, latency_ms, status, f"no translation in {body[:200]}")
    return Probe(True, latency_ms, status)


def run_canary(url, baseline_url=None, minutes=5.0, interval=5.0, beam_size=5, thresholds=None, out=print):
    """Probe for `minutes`, then evaluate. Requests to the two revisions alternate so both see the same load."""
    candidate, baseline = [], []
    deadline = time.monotonic() + minutes * 60
    i = 0
    out(f"Canary: probing {url}" + (f" against {baseline_url}" if baseline_url else "") + f" for {minutes:g} min")
    while True:
        sentence = CANARY_SENTENCES[i % len(CANARY_SENTENCES)]
        result = probe(url, sentence, beam_size)
        candidate.append(result)
        if not result.ok:
            out(f"  new revision failed: status {result.status} {result.error}")
        if baseline_url:
            baseline.append(probe(baseline_url, sentence, beam_size))
        i += 1
        if time.monotonic() + interval > deadline:
            break
        time.sleep(interval)
    verdict = evaluate(candidate, baseline or None, thresholds)
    out(f"  new revision:    {verdict.candidate.line()}")
    if verdict.baseline:
        out(f"  stable revision: {verdict.baseline.line()}")
    out("PASS  canary held" if verdict.ok else "FAIL  " + "; ".join(verdict.reasons))
    write_step_summary(verdict, url, baseline_url, minutes)
    return verdict


def write_step_summary(verdict, url, baseline_url, minutes):
    """Add a small table to the GitHub Actions job summary when running there."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    rows = [("New revision", verdict.candidate)] + ([("Stable revision", verdict.baseline)] if verdict.baseline else [])
    lines = [
        f"#### Canary ({minutes:g} min): {'held' if verdict.ok else 'failed'}",
        "",
        "| Revision | Requests | Errors | p50 | p95 |",
        "|---|---|---|---|---|",
    ]
    for name, s in rows:
        p50 = "–" if s.p50_ms is None else f"{s.p50_ms:.0f} ms"
        p95 = "–" if s.p95_ms is None else f"{s.p95_ms:.0f} ms"
        lines.append(f"| {name} | {s.requests} | {s.errors} ({s.error_rate:.1%}) | {p50} | {p95} |")
    lines += [f"- {r}" for r in verdict.reasons] + [""]
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def build_parser(parser=None):
    parser = parser or argparse.ArgumentParser(description=__doc__.splitlines()[0])
    t = Thresholds()
    parser.add_argument("--url", required=True, help="tag URL of the new revision")
    parser.add_argument("--baseline-url", default=None, help="tag URL of the revision being replaced")
    parser.add_argument("--minutes", type=float, default=5.0, help="length of the canary window")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between probes")
    parser.add_argument("--beam-size", type=int, default=5, help="beam size of the probes (production default: 5)")
    parser.add_argument("--max-error-rate", type=float, default=t.max_error_rate)
    parser.add_argument("--max-p95-ratio", type=float, default=t.max_p95_ratio)
    parser.add_argument("--p95-slack-ms", type=float, default=t.p95_slack_ms)
    parser.add_argument("--max-p95-ms", type=float, default=None, help="absolute p95 ceiling (optional)")
    parser.add_argument("--min-requests", type=int, default=t.min_requests)
    return parser


def run_from_args(args):
    thresholds = Thresholds(
        args.max_error_rate, args.max_p95_ratio, args.p95_slack_ms, args.max_p95_ms, args.min_requests
    )
    verdict = run_canary(args.url, args.baseline_url, args.minutes, args.interval, args.beam_size, thresholds)
    return 0 if verdict.ok else 1


def main(argv=None):
    return run_from_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
