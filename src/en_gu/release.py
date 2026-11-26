"""Check shadow comparisons and update the registry after a verified rollout."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from en_gu.canary import Probe, Thresholds, evaluate
from en_gu.files import atomic_write


def shadow_report(records, stable, candidate, minimum, min_similarity):
    live, shadow, similarity = [], [], []
    for entry in records:
        event = entry.get("jsonPayload", entry)
        if event.get("stable_version") != stable or event.get("event") not in ("shadow_comparison", "shadow_dropped"):
            continue
        ok = event.get("ok") is True and event.get("candidate_version") == candidate
        shadow.append(Probe(ok, float(event["candidate_latency_ms"]) if ok else 0))
        if ok:
            live.append(Probe(True, float(event["stable_latency_ms"])))
            similarity.append(float(event["similarity"]))
    verdict = evaluate(shadow, live, Thresholds(min_requests=minimum))
    average = mean(similarity) if similarity else None
    if average is None or average < min_similarity:
        verdict.reasons.append("mean shadow text similarity missed the floor")
    return {
        **asdict(verdict),
        "ok": not verdict.reasons,
        "mean_similarity": average,
        "stable_version": stable,
        "candidate_version": candidate,
    }


def traffic_report(records, stable_revision, candidate_revision, minimum):
    probes = {stable_revision: [], candidate_revision: []}
    for entry in records:
        revision = entry["resource"]["labels"]["revision_name"]
        http = entry["httpRequest"]
        if revision not in probes or http.get("userAgent", "").startswith("en-gu-"):
            continue
        status = int(http["status"])
        latency_ms = float(http["latency"].removesuffix("s")) * 1000
        probes[revision].append(Probe(status < 500 and status != 429, latency_ms, status))
    settings = Thresholds(min_requests=minimum)
    verdict = evaluate(probes[candidate_revision], probes[stable_revision], settings)
    if (
        verdict.baseline is None
        or verdict.baseline.requests < minimum
        or verdict.baseline.error_rate > settings.max_error_rate
    ):
        verdict.reasons.append("the stable revision had too little healthy production traffic")
    return {
        **asdict(verdict),
        "ok": not verdict.reasons,
        "candidate_revision": candidate_revision,
        "stable_revision": stable_revision,
    }


def finalize(cfg, report, shadow, canary, golden, traffic):
    from en_gu import registry

    version = report.get("version")
    if report.get("passed") is not True or report.get("dry_run") or not version:
        raise ValueError("The gate report must describe an approved, registered candidate")
    expected = (report["champion"] or {}).get("version")
    for label, evidence in (("shadow", shadow), ("canary", canary)):
        if not (
            evidence.get("ok") is True
            and evidence.get("candidate_version") == f"v{version}"
            and evidence.get("stable_version") == f"v{expected}"
        ):
            raise ValueError(f"The {label} report does not approve this candidate against its champion")
    if golden.get("passed") is not True or golden.get("model_version") != f"v{version}":
        raise ValueError("The golden report does not approve this candidate")
    if traffic.get("ok") is not True or traffic.get("candidate_revision") != golden.get("revision"):
        raise ValueError("Production traffic did not verify the golden-checked revision")
    staged = registry.get_by_alias(cfg, registry.CANDIDATE)
    if (
        staged is None
        or int(staged.version) != version
        or staged.tags.get("model_sha256") != report["candidate_sha256"]
    ):
        raise ValueError("The candidate alias or checkpoint hash changed after the gate")
    current = registry.get_by_alias(cfg, registry.CHAMPION)
    if (int(current.version) if current else None) != expected:
        raise ValueError("The champion changed after the gate")
    registry.make_champion(cfg, version)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    shadow = commands.add_parser("shadow")
    shadow.add_argument("--logs", type=Path, required=True)
    shadow.add_argument("--stable", required=True)
    shadow.add_argument("--candidate", required=True)
    shadow.add_argument("--minimum", type=int, default=20)
    shadow.add_argument("--min-similarity", type=float, default=0.5)
    shadow.add_argument("--out", type=Path, required=True)
    traffic = commands.add_parser("traffic")
    traffic.add_argument("--logs", type=Path, required=True)
    traffic.add_argument("--stable", required=True)
    traffic.add_argument("--candidate", required=True)
    traffic.add_argument("--minimum", type=int, default=10)
    traffic.add_argument("--out", type=Path, required=True)
    promote = commands.add_parser("finalize")
    promote.add_argument("--gate", type=Path, required=True)
    promote.add_argument("--shadow", type=Path, required=True)
    promote.add_argument("--canary", type=Path, required=True)
    promote.add_argument("--golden", type=Path, required=True)
    promote.add_argument("--traffic", type=Path, required=True)
    promote.add_argument("--tracking-uri", required=True)
    args = parser.parse_args()
    if args.command in ("shadow", "traffic"):
        if args.minimum < 1:
            parser.error("minimum must be positive")
        rows = json.loads(args.logs.read_text(encoding="utf-8"))
        if args.command == "shadow":
            if not 0 <= args.min_similarity <= 1:
                parser.error("similarity must be between 0 and 1")
            report = shadow_report(rows, args.stable, args.candidate, args.minimum, args.min_similarity)
        else:
            report = traffic_report(rows, args.stable, args.candidate, args.minimum)
        atomic_write(args.out, json.dumps(report, indent=2).encode())
        print(json.dumps(report))
        return int(not report["ok"])

    from en_gu.config import load_config

    cfg = load_config(["configs/base.yaml"])
    cfg["tracking"]["uri"] = args.tracking_uri
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (args.gate, args.shadow, args.canary, args.golden, args.traffic)
    ]
    finalize(cfg, *reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
