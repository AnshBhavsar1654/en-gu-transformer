"""Model loading and translation for the inference API (and the Streamlit app in standalone mode).

The model can come from three places (`serve.model.source`):

- `path`: a checkpoint file, e.g. mounted into the container or baked into the image;
- `registry`: the version that holds an alias (default `champion`) in the MLflow registry;
- `hub`: `best.pt` downloaded from the Hugging Face repo.

Every loaded model gets a version label that the API returns with each translation, so a response can always be
traced back to the exact model that produced it.
"""

import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from en_gu.data import sha256_file
from en_gu.model import load_for_inference, translate
from en_gu.textutils import join_sentences, split_sentences

log = logging.getLogger(__name__)

SERVE_CONFIGS = ["configs/base.yaml", "configs/serve.yaml"]
VERSION_SUFFIX = ".version.json"  # sidecar next to a fetched checkpoint: which registry version it is

# Environment variables that override `serve:` settings (handy in containers and on Cloud Run)
ENV_OVERRIDES = {
    "EN_GU_MODEL_SOURCE": ("model", "source"),
    "EN_GU_MODEL_PATH": ("model", "path"),
    "EN_GU_SPM_PATH": ("model", "spm_path"),
    "EN_GU_MODEL_ALIAS": ("model", "alias"),
    "EN_GU_FEEDBACK_DB": ("feedback_db",),
    "EN_GU_DEVICE": ("device",),
}


def serve_settings(cfg, environ=None):
    """The `serve:` section of `cfg` with environment overrides applied (a new dict; cfg is not changed)."""
    environ = os.environ if environ is None else environ
    settings = json.loads(json.dumps(cfg["serve"]))  # cheap deep copy of plain YAML data
    for var, keys in ENV_OVERRIDES.items():
        if environ.get(var):
            target = settings
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = environ[var]
    return settings


def resolve_device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def file_version(path):
    """Version label of a checkpoint file: from the sidecar written by `fetch-model`, else its content hash."""
    sidecar = Path(str(path) + VERSION_SUFFIX)
    if sidecar.exists():
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if meta.get("sha256") in (None, sha256_file(path)):
            return meta["version"]
        log.warning("%s does not match %s; ignoring it", sidecar.name, Path(path).name)
    return f"sha-{sha256_file(path)[:12]}"


def locate_model(cfg, settings):
    """Return (checkpoint path, version label) for the configured source."""
    mcfg = settings["model"]
    source = mcfg["source"]
    if source == "path":
        path = Path(mcfg["path"])
        if not path.exists():
            raise FileNotFoundError(f"Model file {path} not found (serve.model.path / EN_GU_MODEL_PATH).")
        return path, file_version(path)
    if source == "registry":
        from en_gu.registry import download_checkpoint, get_by_alias, model_name

        version = get_by_alias(cfg, mcfg["alias"])
        if version is None:
            raise LookupError(f"No version of '{model_name(cfg)}' holds the alias '{mcfg['alias']}'.")
        return download_checkpoint(cfg, version), f"v{version.version}"
    if source == "hub":
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(mcfg["hub_repo"], mcfg["hub_filename"]))
        return path, f"hf-{sha256_file(path)[:12]}"
    raise ValueError(f"Unknown serve.model.source {source!r}: use path, registry or hub")


def fetch_model(cfg, out, alias="champion"):
    """Copy a registry version's checkpoint to `out` (e.g. for a Docker build) with a version sidecar."""
    from en_gu.registry import download_checkpoint, get_by_alias, model_name

    version = get_by_alias(cfg, alias)
    if version is None:
        raise LookupError(f"No version of '{model_name(cfg)}' holds the alias '{alias}'.")
    src = download_checkpoint(cfg, version)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, out)
    meta = {
        "version": f"v{version.version}",
        "registered_model": model_name(cfg),
        "alias": alias,
        "sha256": sha256_file(out),
    }
    Path(str(out) + VERSION_SUFFIX).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out} (version v{version.version} of {model_name(cfg)}, alias {alias})")
    return out, meta


@dataclass
class TranslationResult:
    translation: str
    n_sentences: int
    latency_ms: float
    features: dict = field(default_factory=dict)


@dataclass
class Translator:
    """A loaded model plus everything needed to translate free text with it."""

    model: torch.nn.Module
    sp: object
    version: str
    metrics: dict = field(default_factory=dict)
    device: str = "cpu"
    load_seconds: float = 0.0
    source: str = "path"
    # One translation at a time: concurrent CPU inference only competes for the same cores
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def load(cls, cfg, settings=None):
        settings = settings or serve_settings(cfg)
        start = time.perf_counter()
        if settings.get("threads"):
            torch.set_num_threads(int(settings["threads"]))
        path, version = locate_model(cfg, settings)
        device = resolve_device(settings.get("device", "cpu"))
        model, sp, metrics = load_for_inference(path, device, spm_path=settings["model"].get("spm_path"))
        elapsed = time.perf_counter() - start
        log.info("Loaded model %s from %s on %s in %.1fs", version, path, device, elapsed)
        return cls(model, sp, version, metrics, device, elapsed, settings["model"]["source"])

    def translate_text(self, text, beam_size=5, *, observe=False):
        """Translate free text: split into sentences, translate them as one batch, keep the line layout."""
        start = time.perf_counter()
        layout, sentences = split_sentences(text)
        traces = {} if observe else None
        with self._lock:
            outputs = translate(self.model, self.sp, sentences, beam_size=beam_size, traces=traces) if sentences else []
        latency_ms = (time.perf_counter() - start) * 1000
        translation = join_sentences(layout, outputs)
        features = {}
        if observe:
            from en_gu.telemetry import input_features, output_features

            ordered = [traces[i] for i in sorted(traces)]
            features = input_features(text, self.sp, [trace.source_ids for trace in ordered])
            features.update(output_features(translation, ordered, features["input_tokens"]))
        return TranslationResult(translation, len(sentences), latency_ms, features)

    def info(self):
        m = self.model
        return {
            "model_version": self.version,
            "source": self.source,
            "device": self.device,
            "parameters": sum(p.numel() for p in m.parameters()),
            "architecture": {
                "encoder_layers": len(m.encoder_layers),
                "decoder_layers": len(m.decoder_layers),
                "d_model": m.d_model,
                "vocab_size": m.embedding.num_embeddings,
                "max_len": m.max_len,
            },
            "metrics": {k: v for k, v in self.metrics.items() if isinstance(v, int | float)},
            "load_seconds": round(self.load_seconds, 2),
        }
