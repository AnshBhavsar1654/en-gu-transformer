"""Reading and writing the checkpoint formats used by training, evaluation and the app."""

import os

import torch


def atomic_save(obj, path):
    """Write to a temp file and rename, so a reader never sees a half-written checkpoint."""
    path = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def spm_tensor(sp):
    """The SentencePiece model as a uint8 tensor, so it can live inside a checkpoint."""
    return torch.frombuffer(bytearray(sp.serialized_model_proto()), dtype=torch.uint8)


def save_for_app(path, config, state_dict, sp, metrics):
    """Self-contained checkpoint (config + weights + tokenizer + metrics): all `app.py` needs."""
    atomic_save(
        {
            "config": config,
            "model_state_dict": state_dict,
            "spm_model": spm_tensor(sp),
            "metrics": metrics,
        },
        path,
    )


def average_state_dicts(paths):
    """Element-wise mean of the weights stored in several bare state_dict files."""
    avg = None
    for path in paths:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if avg is None:
            avg = {k: v.float().clone() for k, v in state.items()}
        else:
            for k in avg:
                avg[k] += state[k].float()
    return {k: v / len(paths) for k, v in avg.items()}
