"""YAML configuration with layering (`--config a.yaml --config b.yaml`) and `--set key.sub=value` overrides."""

import copy
from pathlib import Path

import yaml


class Config(dict):
    """A dict with attribute access for reading: `cfg.train.max_epochs`.

    Nested dicts are wrapped on access, so assign through item syntax (`cfg["train"]["x"] = 1`).
    """

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError:
            raise AttributeError(name) from None
        return Config(value) if isinstance(value, dict) else value


def deep_merge(base, override):
    """Return `base` updated recursively with `override` (neither input is modified)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_override(text):
    """'train.max_epochs=3' -> {'train': {'max_epochs': 3}}. Values are parsed as YAML (ints, floats, null, ...)."""
    key, sep, raw = text.partition("=")
    if not sep or not key.strip():
        raise ValueError(f"Override must look like key.sub=value, got {text!r}")
    nested = yaml.safe_load(raw)
    if isinstance(nested, str):  # YAML reads `7e-4` as text; accept it as the float it obviously is
        try:
            nested = float(nested)
        except ValueError:
            pass
    for part in reversed(key.strip().split(".")):
        nested = {part: nested}
    return nested


def load_config(paths, overrides=()):
    """Merge YAML files in order, then apply `key=value` overrides."""
    cfg = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    for item in overrides:
        cfg = deep_merge(cfg, parse_override(item))
    return Config(cfg)


def model_config(cfg):
    """Constructor arguments for `Transformer`; the vocabulary size always follows the tokenizer."""
    return {**cfg["model"], "vocab_size": cfg["tokenizer"]["vocab_size"]}


def artifact_path(cfg, *parts):
    return Path(cfg["paths"]["artifacts_dir"]).joinpath(*parts)
