import numpy as np
import pytest

from en_gu.data import (
    TokenBatchSampler,
    collate,
    keep_pair,
    load_clean_pairs,
    load_eval_set,
    split_pairs,
    write_eval_set,
)
from en_gu.model import BOS_IDX, EOS_IDX, PAD_IDX

from .conftest import make_pairs

GOOD_EN, GOOD_GU = "I am hungry.", "મને ભૂખ લાગી છે"


def test_keep_pair_accepts_a_normal_pair():
    assert keep_pair(GOOD_EN, GOOD_GU)


@pytest.mark.parametrize(
    ("en", "gu"),
    [
        ("", GOOD_GU),
        (GOOD_EN, ""),
        (GOOD_EN, GOOD_EN),  # untranslated
        ("મને ભૂખ", GOOD_GU),  # Gujarati on the English side
        ("12345", GOOD_GU),  # no Latin letters at all
        (GOOD_EN, "this is not gujarati at all"),
        (GOOD_EN, " ".join(["ભૂખ"] * 30)),  # badly misaligned lengths
    ],
)
def test_keep_pair_rejects_bad_pairs(en, gu):
    assert not keep_pair(en, gu)


def test_keep_pair_rejects_overlong_sentences():
    long_en = " ".join(["word"] * 81)
    long_gu = " ".join(["ભૂખ"] * 81)
    assert not keep_pair(long_en, long_gu)
    assert keep_pair(long_en, long_gu, max_words=100)


def test_load_clean_pairs_deduplicates_and_respects_limit(tmp_path):
    en = [GOOD_EN, GOOD_EN, "Another one."]
    gu = [GOOD_GU, GOOD_GU, "બીજું એક"]
    (tmp_path / "train.en").write_text("\n".join(en) + "\n", encoding="utf-8")
    (tmp_path / "train.gu").write_text("\n".join(gu) + "\n", encoding="utf-8")
    assert load_clean_pairs(tmp_path)[0] == [GOOD_EN, "Another one."]
    assert load_clean_pairs(tmp_path, max_sentences=1)[0] == [GOOD_EN]


def test_split_is_deterministic_and_leak_free():
    en, gu = make_pairs(500)
    a = split_pairs(en, gu, seed=1, val_size=30, test_size=40)
    b = split_pairs(en, gu, seed=1, val_size=30, test_size=40)
    assert a == b
    assert len(a["val"][0]) == 30 and len(a["test"][0]) == 40
    heldout = {s.lower() for s in a["val"][0] + a["test"][0]}
    assert not heldout & {s.lower() for s in a["train"][0]}
    assert all(len(x[0]) == len(x[1]) for x in a.values())


def test_frozen_eval_survives_changes_to_the_training_data(tmp_path):
    en, gu = make_pairs(500)
    first = split_pairs(en, gu, seed=1, val_size=30, test_size=40)
    write_eval_set(tmp_path, first["val"], first["test"], {"seed": 1})

    en2, gu2 = make_pairs(700, seed=5)  # a different, larger corpus
    frozen = load_eval_set(tmp_path)
    second = split_pairs(
        en2 + first["test"][0], gu2 + first["test"][1], seed=1, val_size=30, test_size=40, frozen=frozen
    )
    assert second["val"] == first["val"] and second["test"] == first["test"]
    assert not {s.lower() for s in first["test"][0]} & {s.lower() for s in second["train"][0]}


def test_modified_eval_set_is_rejected(tmp_path):
    en, gu = make_pairs(200)
    splits = split_pairs(en, gu, seed=1, val_size=10, test_size=10)
    write_eval_set(tmp_path, splits["val"], splits["test"], {})
    load_eval_set(tmp_path)  # fine
    with open(tmp_path / "test.en", "a", encoding="utf-8") as f:
        f.write("tampered\n")
    with pytest.raises(RuntimeError, match="modified after it was frozen"):
        load_eval_set(tmp_path)


def test_missing_eval_set_has_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare"):
        load_eval_set(tmp_path)


def test_token_batch_sampler_covers_every_item_within_the_token_budget():
    rng = np.random.default_rng(0)
    src, tgt = rng.integers(1, 40, 500), rng.integers(1, 40, 500)
    sampler = TokenBatchSampler(src, tgt, max_tokens=256, seed=3)
    seen = [i for batch in sampler for i in batch]
    assert sorted(seen) == list(range(500))
    widths = np.maximum(src, tgt) + 1
    assert all(len(batch) * widths[batch].max() <= 256 for batch in sampler)


def test_token_batch_sampler_reshuffles_per_epoch_but_is_reproducible():
    lens = np.arange(1, 200) % 30 + 1
    sampler = TokenBatchSampler(lens, lens, max_tokens=128, seed=3)
    epoch0 = [list(b) for b in sampler]
    sampler.set_epoch(1)
    epoch1 = [list(b) for b in sampler]
    sampler.set_epoch(0)
    assert [list(b) for b in sampler] == epoch0
    assert epoch0 != epoch1


def test_collate_adds_special_tokens_and_padding():
    batch = [
        (np.array([5, 6, 7], dtype=np.int32), np.array([8, 9], dtype=np.int32)),
        (np.array([5], dtype=np.int32), np.array([8, 9, 10, 11], dtype=np.int32)),
    ]
    src, tgt_in, tgt_out = collate(batch)
    assert src.tolist() == [[5, 6, 7, EOS_IDX], [5, EOS_IDX, PAD_IDX, PAD_IDX]]
    assert tgt_in.tolist() == [[BOS_IDX, 8, 9, PAD_IDX, PAD_IDX], [BOS_IDX, 8, 9, 10, 11]]
    assert tgt_out.tolist() == [[8, 9, EOS_IDX, PAD_IDX, PAD_IDX], [8, 9, 10, 11, EOS_IDX]]
