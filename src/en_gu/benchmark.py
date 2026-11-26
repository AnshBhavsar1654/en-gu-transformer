"""Measure a serving candidate on a frozen test subset using one CPU process per variant."""

import argparse
import json
from pathlib import Path

import psutil
import sacrebleu
import torch

from en_gu.checkpoint import atomic_save
from en_gu.data import load_eval_set, sha256_file
from en_gu.gate import measure_latency
from en_gu.model import load_for_inference, translate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--eval-dir", default="data/eval")
    parser.add_argument("--variant", choices=("baseline", "cached", "cached-int8"), required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--latency-sentences", type=int, default=40)
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--candidate", type=Path, help="write a self-contained checkpoint with caching enabled")
    args = parser.parse_args()
    if min(args.threads, args.limit, args.latency_sentences, args.beam) < 1:
        parser.error("measurement counts must be positive")
    if args.candidate and (args.variant != "cached" or args.candidate.resolve() == args.checkpoint.resolve()):
        parser.error("--candidate needs the cached variant and a separate output path")
    torch.set_num_threads(args.threads)
    process = psutil.Process()
    model, tokenizer, _ = load_for_inference(args.checkpoint, "cpu")
    model.use_kv_cache = args.variant != "baseline"
    if args.variant == "cached-int8":
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    rss = process.memory_info().rss
    _, (en, gu) = load_eval_set(args.eval_dir)
    sources, refs = en[: args.limit], gu[: args.limit]
    budget = measure_latency(model, tokenizer, sources[: args.latency_sentences], args.beam, translate)
    hypotheses = translate(model, tokenizer, sources, beam_size=args.beam, batch_size=4)
    print(
        json.dumps(
            {
                **budget,
                "variant": args.variant,
                "sentences": len(sources),
                "threads": args.threads,
                "beam": args.beam,
                "checkpoint_sha256": sha256_file(args.checkpoint),
                "rss_after_load_bytes": rss,
                "bleu": sacrebleu.corpus_bleu(hypotheses, [refs]).score,
                "chrf": sacrebleu.corpus_chrf(hypotheses, [refs], word_order=2).score,
            },
            indent=2,
        )
    )
    if args.candidate:
        data = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        data["inference"] = {**data.get("inference", {}), "kv_cache": True}
        atomic_save(data, args.candidate)


if __name__ == "__main__":
    main()
