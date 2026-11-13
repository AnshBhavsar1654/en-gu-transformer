"""`python -m en_gu <stage>`: run one pipeline stage, or all of them.

Stages: prepare -> tokenizer -> train -> evaluate -> export. `score` evaluates any exported checkpoint.
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
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config or ["configs/base.yaml"], args.set)
    if args.command == "score":
        from en_gu.evaluate import score_checkpoint

        score_checkpoint(args.checkpoint, cfg, args.split, args.beam, args.limit, args.spm)
    else:
        for name in STAGES if args.command == "all" else (args.command,):
            print(f"\n=== {name} ===")
            run_stage(name, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
