"""Build a frozen feature reference and compare it with recent translation logs."""

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean

from en_gu.files import atomic_write
from en_gu.monitor_export import Sample, write_results

# Each threshold is in the units of its test: p-value for KS, population stability index for PSI.
TESTS = {
    "input_chars": ("ks", 0.01),
    "input_tokens": ("ks", 0.01),
    "n_sentences": ("psi", 0.2),
    "byte_or_unknown_share": ("psi", 0.2),
    "digit_or_punctuation_share": ("psi", 0.2),
    "non_latin_share": ("psi", 0.2),
    "non_english": ("psi", 0.2),
}
MIN_DRIFT_SCORE = 0.25


def build_reference(checkpoint, validation, size):
    import sentencepiece as spm
    import torch

    from en_gu.telemetry import input_features

    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    tokenizer = bytes(data["spm_model"].tolist())
    sp = spm.SentencePieceProcessor(model_proto=tokenizer)
    text = validation.read_text(encoding="utf-8")
    sources = [line.strip() for line in text.splitlines() if line.strip()]
    if not 1 <= size <= len(sources):
        raise ValueError(f"Reference size must be between 1 and {len(sources)}")
    return {
        "schema": 1,
        "validation_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(tokenizer).hexdigest(),
        "rows": [input_features(source, sp) for source in random.Random(42).sample(sources, size)],
    }


def read_events(path):
    """Read local mixed stdout or a gcloud logging read JSON array."""
    content = path.read_text(encoding="utf-8-sig").lstrip()
    if content.startswith("["):
        records = json.loads(content)
        entries = ((record.get("jsonPayload", {}), record.get("timestamp")) for record in records)
    else:
        entries = ((json.loads(line), None) for line in content.splitlines() if line.lstrip().startswith("{"))
    for event, log_timestamp in entries:
        if event.get("schema") == 1 and event.get("event") in {"translation", "feedback"}:
            timestamp = event.get("timestamp") or log_timestamp
            yield datetime.fromisoformat(timestamp.replace("Z", "+00:00")), event


def compare(reference, current, minimum, html=None):
    # Feature analysis stays in the runner. Disable the library's usage reporting.
    os.environ.setdefault("DO_NOT_TRACK", "1")
    from evidently import Report
    from evidently.metrics import ValueDrift
    from pandas import DataFrame

    if len(current) < minimum:
        return {"samples": len(current), "score": None, "features": {}}

    ref, cur = (DataFrame(rows).reindex(columns=TESTS).astype(float) for rows in (reference, current))
    tests = {key: settings for key, settings in TESTS.items() if min(ref[key].count(), cur[key].count()) >= minimum}
    if not tests:
        return {"samples": len(current), "score": None, "features": {}}
    metrics = [ValueDrift(column=key, method=method, threshold=threshold) for key, (method, threshold) in tests.items()]
    snapshot = Report(metrics).run(cur[list(tests)], ref[list(tests)])
    if html is not None:
        atomic_write(html, snapshot.get_html_str(as_iframe=False).encode())
    details = {}
    for (key, (method, threshold)), result in zip(tests.items(), snapshot.dict()["metrics"], strict=True):
        value = float(result["value"])
        if math.isfinite(value):
            detected = value <= threshold if method == "ks" else value >= threshold
            details[key] = {"method": method, "value": value, "threshold": threshold, "drifted": detected}
    score = mean(feature["drifted"] for feature in details.values()) if details else None
    return {"samples": len(current), "score": score, "features": details}


def summaries(events):
    features = defaultdict(list)
    feedback = Counter()
    feedback_versions = set()
    for event in events:
        labels = (event.get("model_version", "unknown"), event.get("revision", "local"))
        if event["event"] == "feedback":
            feedback_versions.add(labels)
            feedback[(*labels, event["rating"], str(event["corrected"]).lower())] += 1
        else:
            for feature in (
                *TESTS,
                "latency_ms",
                "output_tokens",
                "output_input_ratio",
                "output_latin_share",
                "repetition_rate",
                "hit_length_limit",
                "mean_log_probability",
            ):
                value = event.get(feature)
                if value is not None:
                    features[(*labels, feature)].append(value)
    samples = [
        Sample("daily", "feature_mean", mean(values), {"version": model, "revision": revision, "feature": feature})
        for (model, revision, feature), values in features.items()
    ]
    samples.extend(
        Sample(
            "daily",
            "feedback_count",
            feedback[model, revision, rating, corrected],
            {"version": model, "revision": revision, "rating": rating, "with_correction": corrected},
        )
        for model, revision in sorted(feedback_versions)
        for rating in ("up", "down")
        for corrected in ("true", "false")
    )
    return samples


def run_report(logs, reference, out, minimum=100, now=None, include_simulated=False):
    now = now or datetime.now(timezone.utc)
    traffic = {"regular", "simulated"} if include_simulated else {"regular"}
    rows = [
        (stamp, event)
        for stamp, event in read_events(logs)
        if stamp <= now and event.get("traffic", "regular") in traffic
    ]
    data = json.loads(reference.read_text(encoding="utf-8"))
    if data["schema"] != 1:
        raise ValueError("Unsupported reference schema")

    out.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    samples = [Sample("drift", "timestamp_seconds", now.timestamp())]
    for window, start, end in (("24h", 1, 0), ("7d", 7, 0), ("previous_24h", 2, 1)):
        since, until = now - timedelta(days=start), now - timedelta(days=end)
        events = [event for stamp, event in rows if since < stamp <= until]
        current = [event for event in events if event["event"] == "translation"]
        html = out.with_name(f"{out.name}-{window}.html")
        html.unlink(missing_ok=True)
        result = compare(data["rows"], current, minimum, html if window != "previous_24h" else None)
        results[window] = result
        samples.append(Sample("drift", "samples", len(current), {"window": window}))
        if result["score"] is not None:
            samples.append(Sample("drift", "score", result["score"], {"window": window}))
        for key, feature in result["features"].items():
            samples.append(
                Sample("drift", "feature_drifted", int(feature["drifted"]), {"window": window, "feature": key})
            )
        if window == "24h":
            samples.extend(summaries(events))
    alert = all(
        results[window]["score"] is not None and results[window]["score"] >= MIN_DRIFT_SCORE
        for window in ("24h", "previous_24h")
    )
    samples.append(Sample("drift", "persistent", int(alert)))
    outcome = {"as_of": now.isoformat(), "alert": alert, "threshold": MIN_DRIFT_SCORE, "windows": results}
    write_results(out, outcome, samples)
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    jobs = parser.add_subparsers(dest="job", required=True)
    reference = jobs.add_parser("reference")
    reference.add_argument("--checkpoint", type=Path, required=True)
    reference.add_argument("--validation", type=Path, default=Path("data/eval/val.en"))
    reference.add_argument("--size", type=int, default=500)
    reference.add_argument("--out", type=Path, default=Path("monitoring/reference.json"))
    report = jobs.add_parser("report")
    report.add_argument("--logs", type=Path, required=True)
    report.add_argument("--reference", type=Path, default=Path("monitoring/reference.json"))
    report.add_argument("--out", type=Path, default=Path("artifacts/monitoring/drift"))
    report.add_argument("--minimum", type=int, default=100)
    report.add_argument("--include-simulated", action="store_true")
    args = parser.parse_args()
    if args.job == "reference":
        data = build_reference(args.checkpoint, args.validation, args.size)
        atomic_write(args.out, (json.dumps(data, indent=2) + "\n").encode())
    elif args.minimum < 2:
        parser.error("--minimum must be at least 2")
    else:
        outcome = run_report(
            args.logs, args.reference, args.out, args.minimum, include_simulated=args.include_simulated
        )
        print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
