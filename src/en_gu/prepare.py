"""Stage 1: clean the corpus and write the training set plus the frozen validation/test sets."""

from pathlib import Path

from en_gu.data import (
    has_frozen_eval,
    load_clean_pairs,
    load_eval_set,
    split_pairs,
    write_eval_set,
    write_lines,
)


def prepare(cfg):
    dcfg, paths = cfg["data"], cfg["paths"]
    en, gu = load_clean_pairs(paths["corpus_dir"], dcfg["max_sentences"], dcfg["max_words"])
    if not en:
        raise RuntimeError(f"No usable sentence pairs found in {paths['corpus_dir']}")

    eval_dir = Path(paths["eval_dir"])
    frozen = None
    if has_frozen_eval(eval_dir) and not dcfg.get("refreeze_eval", False):
        # Keep scoring every model on exactly the same sentences, whatever happens to the training data
        frozen = load_eval_set(eval_dir)
        print(f"Reusing frozen eval set in {eval_dir} ({len(frozen[0][0])} val / {len(frozen[1][0])} test)")

    splits = split_pairs(en, gu, cfg["seed"], dcfg["val_size"], dcfg["test_size"], frozen=frozen)
    if frozen is None:
        meta = {"seed": cfg["seed"], "max_sentences": dcfg["max_sentences"], "max_words": dcfg["max_words"]}
        write_eval_set(eval_dir, splits["val"], splits["test"], meta)
        print(f"Froze a new eval set in {eval_dir}")

    processed = Path(paths["processed_dir"])
    write_lines(processed / "train.en", splits["train"][0])
    write_lines(processed / "train.gu", splits["train"][1])
    print(f"train: {len(splits['train'][0]):,}  val: {len(splits['val'][0]):,}  test: {len(splits['test'][0]):,}")
    return splits
