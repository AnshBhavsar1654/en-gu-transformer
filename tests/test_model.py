import pytest
import torch

from en_gu.checkpoint import save_for_app
from en_gu.model import (
    BANNED_TOKENS,
    EOS_IDX,
    PAD_IDX,
    Transformer,
    beam_search,
    config_from_state_dict,
    greedy_decode,
    load_for_inference,
    translate,
)
from en_gu.tokenizer import load_tokenizer

from .conftest import TINY_MODEL


def _src(batch=2, length=6):
    torch.manual_seed(1)
    return torch.randint(4, 300, (batch, length))


def test_forward_shape(tiny_model):
    src, tgt = _src(), _src(length=5)
    assert tiny_model(src, tgt).shape == (2, 5, 400)


def test_output_projection_is_tied_to_the_embedding(tiny_model):
    assert tiny_model.output_proj.weight is tiny_model.embedding.weight


@pytest.mark.parametrize("decode", [greedy_decode, lambda m, s: beam_search(m, s, beam_size=3)])
def test_decoding_never_emits_banned_tokens_and_respects_the_length_limit(tiny_model, decode):
    src = _src()
    for hyp in decode(tiny_model, src):
        assert not set(hyp) & set(BANNED_TOKENS)
        assert EOS_IDX not in hyp
        assert len(hyp) <= int(1.5 * src.size(1) + 10)


def test_beam_search_handles_a_batch_with_padding(tiny_model):
    src = _src(3, 7)
    src[1, 4:] = PAD_IDX
    assert len(beam_search(tiny_model, src, beam_size=2)) == 3


def test_translate_returns_one_string_per_input_including_blanks(tiny_model, spm_path):
    sp = load_tokenizer(spm_path)
    out = translate(tiny_model, sp, ["the train", "", "a big house"], beam_size=2)
    assert len(out) == 3 and out[1] == "" and all(isinstance(x, str) for x in out)


def test_config_is_recovered_from_a_bare_state_dict(tiny_model):
    config = config_from_state_dict(tiny_model.state_dict())
    assert config["vocab_size"] == 400
    assert config["d_model"] == TINY_MODEL["d_model"]
    assert config["num_encoder_layers"] == 1 and config["num_decoder_layers"] == 1
    assert config["d_ff"] == TINY_MODEL["d_ff"]


def test_load_self_contained_checkpoint(tmp_path, tiny_model, spm_path):
    sp = load_tokenizer(spm_path)
    path = tmp_path / "best.pt"
    state = {k: v.half() for k, v in tiny_model.state_dict().items() if k != "output_proj.weight"}
    save_for_app(path, {"vocab_size": 400, **TINY_MODEL}, state, sp, {"test_bleu": 1.0})
    model, loaded_sp, metrics = load_for_inference(path)
    assert metrics == {"test_bleu": 1.0} and loaded_sp.get_piece_size() == 400
    assert isinstance(model, Transformer) and next(model.parameters()).dtype == torch.float32


def test_weights_only_checkpoint_needs_a_tokenizer(tmp_path, tiny_model, spm_path):
    path = tmp_path / "weights.pt"
    torch.save(tiny_model.state_dict(), path)
    with pytest.raises(FileNotFoundError, match="spm_en_gu.model"):
        load_for_inference(path)
    model, sp, _ = load_for_inference(path, spm_path=spm_path)
    assert sp.get_piece_size() == 400 and model.d_model == TINY_MODEL["d_model"]


def test_tokenizer_from_a_different_run_is_rejected(tmp_path, spm_path):
    other = Transformer(vocab_size=300, **TINY_MODEL)
    path = tmp_path / "weights.pt"
    torch.save(other.state_dict(), path)
    with pytest.raises(RuntimeError, match="different training runs"):
        load_for_inference(path, spm_path=spm_path)


def test_mismatched_weights_are_rejected(tmp_path, tiny_model, spm_path):
    state = tiny_model.state_dict()
    state["unexpected.weight"] = torch.zeros(1)
    path = tmp_path / "bad.pt"
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="Checkpoint mismatch"):
        load_for_inference(path, spm_path=spm_path)
