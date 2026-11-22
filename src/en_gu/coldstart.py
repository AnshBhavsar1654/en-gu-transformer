"""Measure cold starts of a deployed (scale-to-zero) API: how long the first request after idling takes.

Cloud Run stops idle instances after a while (typically ~15 minutes without requests), so each sample waits
`--idle-minutes`, then times one translation from the moment it is sent until the answer arrives (Cloud Run holds
the request while the new instance starts and loads the model). `/info` then tells whether a fresh instance really
answered (new instance id, uptime shorter than the request) and how long the model load took inside it.

    PYTHONPATH=src python -m en_gu.coldstart --url https://en-gu-api-staging-xyz.run.app --samples 3

Use it to compare the model-delivery options (hub download, Cloud Storage volume, baked into the image): deploy
each to staging, run this, and record the cold samples. Only the standard library is used.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
from dataclasses import dataclass

from en_gu.smoke import request

SENTENCE = "What is your name?"


@dataclass
class Sample:
    first_response_s: float  # client side: request sent -> translation received
    status: int | None
    instance: str | None
    uptime_s: float | None  # instance uptime reported right after the request
    model_load_s: float | None  # time the model took to load inside the instance
    model_version: str | None
    revision: str | None
    cold: bool  # a new instance started while the request waited


def is_cold(first_response_s, uptime_s, instance, seen):
    """A sample is cold when an instance we had not seen started during (or just before) our request."""
    if instance is None or uptime_s is None or instance in seen:
        return False
    return uptime_s <= first_response_s + 10  # allow for the /info round trip and clock granularity


def first_translation(base_url, timeout):
    """Send one translation and keep retrying through 429/503 or connection errors until it succeeds."""
    start = time.perf_counter()
    deadline = time.monotonic() + timeout
    status = None
    while time.monotonic() < deadline:
        try:
            status, _, _ = request(base_url, "POST", "/translate", {"text": SENTENCE}, timeout=timeout)
            if status == 200:
                break
        except (urllib.error.URLError, OSError):
            status = None
        time.sleep(1)
    return time.perf_counter() - start, status


def take_sample(base_url, seen, timeout=300):
    elapsed, status = first_translation(base_url, timeout)
    info = {}
    if status == 200:
        code, _, body = request(base_url, "GET", "/info", timeout=30)
        info = json.loads(body) if code == 200 else {}
    instance = info.get("instance")
    sample = Sample(
        first_response_s=round(elapsed, 2),
        status=status,
        instance=instance,
        uptime_s=info.get("uptime_seconds"),
        model_load_s=info.get("load_seconds"),
        model_version=info.get("model_version"),
        revision=info.get("revision"),
        cold=is_cold(elapsed, info.get("uptime_seconds"), instance, seen),
    )
    if instance:
        seen.add(instance)
    return sample


def report(samples):
    """Summary lines: every sample, then the median over the cold ones."""
    lines = []
    for i, s in enumerate(samples, 1):
        kind = "cold" if s.cold else "warm"
        lines.append(
            f"sample {i}: {kind}  first response {s.first_response_s:.1f} s  model load {s.model_load_s} s  "
            f"status {s.status}  instance {s.instance}  revision {s.revision}"
        )
    cold = [s for s in samples if s.cold]
    if cold:
        first = statistics.median(s.first_response_s for s in cold)
        loads = [s.model_load_s for s in cold if s.model_load_s is not None]
        load = f", model load median {statistics.median(loads):.1f} s" if loads else ""
        lines.append(f"cold starts: {len(cold)}/{len(samples)}, first response median {first:.1f} s{load}")
    else:
        lines.append("no cold start observed: the service was still warm (wait longer between samples)")
    return lines


def measure(base_url, samples=3, idle_minutes=16.0, wait_first=True, timeout=300, out=print, sleep=time.sleep):
    seen, results = set(), []
    for i in range(samples):
        if i > 0 or wait_first:
            out(f"waiting {idle_minutes:g} min for the service to scale to zero ...")
            sleep(idle_minutes * 60)
        results.append(take_sample(base_url, seen, timeout))
        out(report(results[-1:])[0].replace("sample 1", f"sample {i + 1}"))
    for line in report(results)[-1:]:
        out(line)
    return results


def build_parser(parser=None):
    parser = parser or argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True, help="base URL of the deployed API")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--idle-minutes", type=float, default=16.0, help="wait before each sample")
    parser.add_argument("--no-initial-wait", action="store_true", help="the service is already idle: start now")
    parser.add_argument("--timeout", type=float, default=300, help="seconds to wait for one cold request")
    parser.add_argument("--json", default=None, help="also write the samples to this JSON file")
    return parser


def run_from_args(args):
    results = measure(args.url, args.samples, args.idle_minutes, not args.no_initial_wait, args.timeout)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump([s.__dict__ for s in results], fh, indent=2)
    return 0 if all(s.status == 200 for s in results) else 1


def main(argv=None):
    return run_from_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
