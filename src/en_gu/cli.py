"""`python -m en_gu <stage>`: run one pipeline stage, or all of them.

Stages: prepare -> tokenizer -> train -> evaluate -> export. `score` evaluates any exported checkpoint;
`promote`, `backfill-run1` and `registry` manage the model registry.
"""

import argparse
import sys

from en_gu.config import load_config

STAGES = ("prepare", "tokenizer", "train", "evaluate", "export")


def run_stage(name, cfg):
    # Imported lazily so e.g. `prepare` doesn't pay for the training imports
    if name == "prepare":
        from en_gu.prepare import prepare

        prepare(cfg)
    elif name == "tokenizer":
        from en_gu.tokenizer import train_tokenizer

        train_tokenizer(cfg)
    elif name == "train":
        from en_gu.train import train

        train(cfg)
    elif name == "evaluate":
        from en_gu.evaluate import evaluate

        evaluate(cfg)
    elif name == "export":
        from en_gu.export import export

        export(cfg)
    else:
        raise ValueError(name)


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m en_gu", description=__doc__.splitlines()[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config", action="append", default=None, help="YAML file; repeat to layer (default: configs/base.yaml)"
    )
    common.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")

    sub = parser.add_subparsers(dest="command", required=True)
    for name in STAGES:
        sub.add_parser(name, parents=[common], help=f"run the {name} stage")
    sub.add_parser("all", parents=[common], help="run every stage in order")

    score = sub.add_parser("score", parents=[common], help="score an exported checkpoint on the frozen eval set")
    score.add_argument("checkpoint")
    score.add_argument("--split", choices=("val", "test"), default="test")
    score.add_argument("--beam", type=int, default=5)
    score.add_argument("--limit", type=int, default=None, help="only score the first N sentences")
    score.add_argument("--spm", default=None, help="tokenizer for weights-only checkpoints")

    backfill = sub.add_parser(
        "backfill-run1", parents=[common], help="register the Kaggle run 1 model as version 1 of the registry"
    )
    backfill.add_argument("--checkpoint", default="best.pt")
    backfill.add_argument("--notebook", default="notebooks/archive/run1-gujarati-english-translation.ipynb")

    promote = sub.add_parser(
        "promote", parents=[common], help="run the promotion gate on a candidate; promote it if it passes"
    )
    promote.add_argument("--candidate", default=None, help="checkpoint to judge (default: <artifacts>/best.pt)")
    promote.add_argument("--beam", type=int, default=None, help="beam size to score with (default: evaluate.beam_size)")
    promote.add_argument("--limit", type=int, default=None, help="score only the first N test sentences (dry run)")
    promote.add_argument("--dry-run", action="store_true", help="show the verdict without touching the registry")
    promote.add_argument("--no-latency", action="store_true", help="skip the latency comparison")
    promote.add_argument(
        "--assume-isolated",
        action="store_true",
        help="treat the candidate as trained without the eval sentences even though its checkpoint doesn't say so",
    )

    registry = sub.add_parser("registry", parents=[common], help="show registry versions, or roll back")
    registry.add_argument("--rollback", action="store_true", help="make the previous version the champion again")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config or ["configs/base.yaml"], args.set)
    if args.command == "score":
        from en_gu.evaluate import score_checkpoint

        score_checkpoint(args.checkpoint, cfg, args.split, args.beam, args.limit, args.spm)
    elif args.command == "backfill-run1":
        from en_gu.backfill import backfill

        backfill(cfg, args.checkpoint, args.notebook)
    elif args.command == "promote":
        from en_gu.promote import promote

        report = promote(
            cfg,
            args.candidate,
            beam_size=args.beam,
            limit=args.limit,
            dry_run=args.dry_run,
            with_latency=not args.no_latency,
            assume_isolated=args.assume_isolated,
        )
        return 0 if report["passed"] else 1
    elif args.command == "registry":
        from en_gu.registry_cli import show_registry

        show_registry(cfg, rollback=args.rollback)
    else:
        for name in STAGES if args.command == "all" else (args.command,):
            print(f"\n=== {name} ===")
            run_stage(name, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
