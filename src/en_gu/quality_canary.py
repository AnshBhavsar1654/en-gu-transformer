"""Score a fixed validation subset through the public translation endpoint."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF

from en_gu.monitor_export import Sample, write_results
from en_gu.smoke import request, wait_until_ready

GOLDEN_BEAM = 5


def golden_pairs(folder):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    indices = json.loads((folder / "golden_indices.json").read_text(encoding="utf-8"))
    sides = []
    for name in ("val.en", "val.gu"):
        # Read as text to normalize line endings on Windows, as in the committed evaluation snapshot.
        text = (folder / name).read_text(encoding="utf-8")
        if hashlib.sha256(text.encode()).hexdigest() != manifest["files"][name]["sha256"]:
            raise ValueError(f"{name} does not match the frozen evaluation manifest")
        sides.append(text.splitlines())
    if len(set(indices)) != len(indices) or not all(0 <= idx < len(sides[0]) for idx in indices):
        raise ValueError("Invalid golden-set indices")
    return [(sides[0][i], sides[1][i]) for i in indices]


def score_endpoint(url, pairs, min_interval=0.12):
    wait_until_ready(url, 180)
    hypotheses = []
    versions = set()
    next_request = time.monotonic()
    for source, _ in pairs:
        delay = next_request - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        next_request = time.monotonic() + min_interval
        status, headers, content = request(
            url,
            "POST",
            "/translate",
            {"text": source, "beam_size": GOLDEN_BEAM},
            headers={"User-Agent": "en-gu-quality-canary"},
            timeout=60,
        )
        if status != 200:
            raise RuntimeError(f"Golden request {len(hypotheses) + 1} returned HTTP {status}")
        body = json.loads(content)
        version = body["model_version"]
        if version != headers.get("X-Model-Version"):
            raise RuntimeError("Model version in response did not match its header")
        versions.add((version, headers.get("X-Model-Revision", "local")))
        if len(versions) > 1:
            raise RuntimeError("Run received multiple model versions or revisions; retry after rollout")
        hypotheses.append(body["translation"])
    references = [[target for _, target in pairs]]
    version, revision = versions.pop()
    return {
        "bleu": BLEU().corpus_score(hypotheses, references).score,
        "chrf": CHRF(word_order=2).corpus_score(hypotheses, references).score,
        "pairs": len(pairs),
        "beam_size": GOLDEN_BEAM,
        "model_version": version,
        "revision": revision,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--eval-dir", type=Path, default=Path("data/eval"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/monitoring/golden"))
    parser.add_argument("--min-chrf", required=True, type=float)
    parser.add_argument("--expect-version")
    parser.add_argument("--min-interval", type=float, default=0.12, help="minimum seconds between request starts")
    args = parser.parse_args()
    if not 0 <= args.min_chrf <= 100:
        parser.error("--min-chrf must be between 0 and 100")
    if not 0 <= args.min_interval < 60:
        parser.error("--min-interval must be at least zero and less than 60 seconds")
    outcome = {"min_chrf": args.min_chrf}
    try:
        outcome.update(score_endpoint(args.url, golden_pairs(args.eval_dir), args.min_interval))
        outcome["passed"] = outcome["chrf"] >= args.min_chrf and (
            args.expect_version is None or outcome["model_version"] == args.expect_version
        )
    except Exception as exc:
        outcome.update(passed=False, error=str(exc))

    samples = [
        Sample("golden", "timestamp_seconds", time.time()),
        Sample("golden", "passed", int(outcome["passed"])),
        Sample("golden", "probe_error", int("error" in outcome)),
    ]
    if "chrf" in outcome:
        tags = {"version": outcome["model_version"], "revision": outcome["revision"]}
        samples.extend(
            [
                Sample("golden", "chrf", outcome["chrf"], tags),
                Sample("golden", "bleu", outcome["bleu"], tags),
                Sample("golden", "chrf_floor", args.min_chrf, tags),
            ]
        )
    write_results(args.out, outcome, samples)
    print(json.dumps(outcome, indent=2))
    return int(not outcome["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
