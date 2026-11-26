"""Validate explicitly reviewed corrections before including them in training."""

import argparse
import json
from pathlib import Path

from en_gu.data import keep_pair, load_eval_set
from en_gu.files import atomic_write
from en_gu.textutils import normalize_text


def identity(text):
    return normalize_text(text).casefold()


def read_pairs(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not all(isinstance(row.get(key), str) for key in ("source", "target")):
                raise ValueError(f"{path} line {number}: source and target must be strings")
            yield number, normalize_text(row["source"]), normalize_text(row["target"])


def validate_pairs(path, frozen, max_words, seen=None):
    held_out = [{identity(sentence) for split in frozen for sentence in split[side]} for side in (0, 1)]
    seen = set() if seen is None else seen
    accepted, rejected = [], []
    for number, source, target in read_pairs(path):
        key = identity(source)
        if not keep_pair(source, target, max_words):
            reason = "invalid script or length"
        elif key in held_out[0] or identity(target) in held_out[1]:
            reason = "overlaps the frozen evaluation data"
        elif key in seen:
            reason = "duplicate source"
        else:
            accepted.append({"source": source, "target": target})
            seen.add(key)
            continue
        rejected.append({"line": number, "reason": reason})
    return accepted, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviewed", type=Path, help="private JSONL of human-approved source/target pairs")
    parser.add_argument("--eval-dir", default="data/eval")
    parser.add_argument("--out", type=Path, default=Path("data/reviewed/accepted.jsonl"))
    parser.add_argument("--max-words", type=int, default=80)
    args = parser.parse_args()
    frozen = load_eval_set(args.eval_dir)
    existing = []
    if args.out.exists():
        existing, invalid = validate_pairs(args.out, frozen, args.max_words)
        if invalid:
            parser.error("the existing accepted data failed validation")
    seen = {identity(row["source"]) for row in existing}
    new, rejected = validate_pairs(args.reviewed, frozen, args.max_words, seen)
    # A rerun never deletes the previous batch, including when all submitted rows are duplicates.
    if new:
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in [*existing, *new])
        atomic_write(args.out, content.encode(), mode=0o600)
    print(json.dumps({"new_pairs": len(new), "total_pairs": len(existing) + len(new), "rejected": rejected}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
