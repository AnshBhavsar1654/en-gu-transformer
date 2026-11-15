"""The promotion gate: may this candidate replace the champion?

`decide` is a pure function over two score cards, so the rules are easy to test and to read. A score card is:

    {"overall": {"bleu", "chrf", "ter"}, "slices": {name: {"n", "bleu", "chrf"}},
     "size_mb", "latency_p95_ms" (or None), "eval_isolated", "sentences"}

The champion card has an extra "mode":
  - "like_for_like": scored on the same frozen eval set as the candidate, so slices are compared too.
  - "reference": the champion was scored on some other test set (run 1 on Kaggle). Its eval-set scores would be
    contaminated, so only the overall scores are compared, with a wider tolerance.
"""

import statistics
import time

LIKE_FOR_LIKE, REFERENCE = "like_for_like", "reference"


def slice_of(sentence, slices):
    """Name of the length bucket (by English word count) a sentence belongs to, or None."""
    words = len(sentence.split())
    for name, (low, high) in slices.items():
        if words >= low and (high is None or words <= high):
            return name
    return None


def slice_metrics(sources, references, hypotheses, slices, min_sentences=1):
    """Corpus BLEU / chrF++ per length bucket. Buckets with fewer than `min_sentences` sentences are still reported."""
    import sacrebleu

    members = {name: [] for name in slices}
    for i, src in enumerate(sources):
        name = slice_of(src, slices)
        if name is not None:
            members[name].append(i)
    result = {}
    for name, idx in members.items():
        if not idx:
            continue
        hyp, ref = [hypotheses[i] for i in idx], [references[i] for i in idx]
        result[name] = {
            "n": len(idx),
            "bleu": round(sacrebleu.corpus_bleu(hyp, [ref]).score, 2),
            "chrf": round(sacrebleu.corpus_chrf(hyp, [ref], word_order=2).score, 2),
            "enforced": len(idx) >= min_sentences,
        }
    return result


def latency_stats(timings_ms):
    ordered = sorted(timings_ms)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return {"p50_ms": round(statistics.median(ordered), 1), "p95_ms": round(p95, 1), "n": len(ordered)}


def measure_latency(model, sp, sentences, beam_size, translate_fn, warmup=3):
    """Per-sentence wall time (batch of one, as in serving) after a few warm-up calls."""
    for sentence in sentences[:warmup]:
        translate_fn(model, sp, [sentence], beam_size=beam_size, batch_size=1)
    timings = []
    for sentence in sentences:
        start = time.perf_counter()
        translate_fn(model, sp, [sentence], beam_size=beam_size, batch_size=1)
        timings.append((time.perf_counter() - start) * 1000)
    return latency_stats(timings)


def _check(name, passed, value, threshold, detail):
    return {"name": name, "passed": bool(passed), "value": value, "threshold": threshold, "detail": detail}


def decide(candidate, champion, gcfg):
    """Apply every gate rule. Returns {"passed", "mode", "checks": [...]}; a candidate passes only if all checks do."""
    checks = []
    if gcfg["require_isolated_candidate"]:
        checks.append(
            _check(
                "eval_isolated",
                candidate["eval_isolated"],
                candidate["eval_isolated"],
                True,
                "the candidate must have been trained without the frozen eval sentences",
            )
        )
    checks.append(
        _check(
            "size",
            candidate["size_mb"] <= gcfg["max_model_mb"],
            round(candidate["size_mb"], 1),
            gcfg["max_model_mb"],
            "checkpoint size in MB",
        )
    )
    p95 = candidate.get("latency_p95_ms")
    if gcfg.get("max_latency_ms") and p95 is not None:
        checks.append(
            _check("latency_budget", p95 <= gcfg["max_latency_ms"], p95, gcfg["max_latency_ms"], "p95 ms per sentence")
        )

    mode = "no_champion"
    if champion is not None:
        mode = champion["mode"]
        tol = gcfg["tolerance"] if mode == LIKE_FOR_LIKE else gcfg["reference_tolerance"]
        note = "" if mode == LIKE_FOR_LIKE else " (champion's scores are from a different test set)"
        for metric in ("chrf", "bleu"):
            floor = champion["overall"][metric] - tol[metric]
            value = candidate["overall"][metric]
            checks.append(
                _check(
                    metric,
                    value >= floor,
                    round(value, 2),
                    round(floor, 2),
                    f"champion {champion['overall'][metric]:.2f} minus tolerance {tol[metric]}{note}",
                )
            )
        if mode == LIKE_FOR_LIKE:
            for name, cand in candidate["slices"].items():
                champ = champion["slices"].get(name)
                if champ is None or not (cand.get("enforced", True) and champ.get("enforced", True)):
                    continue
                floor = champ["chrf"] - gcfg["tolerance"]["slice_chrf"]
                checks.append(
                    _check(
                        f"slice:{name}",
                        cand["chrf"] >= floor,
                        cand["chrf"],
                        round(floor, 2),
                        f"{cand['n']} sentences; champion {champ['chrf']:.2f} minus tolerance "
                        f"{gcfg['tolerance']['slice_chrf']}",
                    )
                )
        champ_p95 = champion.get("latency_p95_ms")
        if p95 is not None and champ_p95:
            ceiling = round(champ_p95 * gcfg["max_latency_ratio"], 1)
            checks.append(
                _check(
                    "latency_ratio",
                    p95 <= ceiling,
                    p95,
                    ceiling,
                    f"champion p95 {champ_p95} ms x {gcfg['max_latency_ratio']}",
                )
            )
    return {"passed": all(c["passed"] for c in checks), "mode": mode, "checks": checks}


def format_report(report):
    """Human-readable verdict."""
    lines = [f"Gate verdict: {'PASSED' if report['passed'] else 'REJECTED'}  (comparison: {report['mode']})"]
    for c in report["checks"]:
        status = "ok" if c["passed"] else "FAIL"
        lines.append(f"  [{status}] {c['name']:<16} {c['value']!s:<8} vs {c['threshold']!s:<8} {c['detail']}")
    return "\n".join(lines)
