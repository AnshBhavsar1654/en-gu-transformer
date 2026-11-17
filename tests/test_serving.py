import json

import pytest

from en_gu.serving import VERSION_SUFFIX, Translator, locate_model, serve_settings


def test_environment_overrides_serve_settings(serve_cfg):
    settings = serve_settings(serve_cfg, {"EN_GU_MODEL_PATH": "/models/x.pt", "EN_GU_DEVICE": "cuda"})
    assert settings["model"]["path"] == "/models/x.pt" and settings["device"] == "cuda"
    assert serve_cfg["serve"]["device"] == "cpu"  # the config itself is untouched


def test_file_version_is_the_content_hash(serve_cfg, app_checkpoint):
    _, version = locate_model(serve_cfg, serve_settings(serve_cfg, {}))
    assert version.startswith("sha-") and len(version) == 16


def test_file_version_comes_from_a_matching_sidecar(serve_cfg, app_checkpoint):
    from en_gu.data import sha256_file

    sidecar = app_checkpoint.with_name(app_checkpoint.name + VERSION_SUFFIX)
    sidecar.write_text(json.dumps({"version": "v7", "sha256": sha256_file(app_checkpoint)}))
    assert locate_model(serve_cfg, serve_settings(serve_cfg, {}))[1] == "v7"
    sidecar.write_text(json.dumps({"version": "v7", "sha256": "0" * 64}))  # stale sidecar: ignored
    assert locate_model(serve_cfg, serve_settings(serve_cfg, {}))[1].startswith("sha-")


def test_missing_model_file_is_a_clear_error(serve_cfg, tmp_path):
    settings = serve_settings(serve_cfg, {"EN_GU_MODEL_PATH": str(tmp_path / "nope.pt")})
    with pytest.raises(FileNotFoundError, match="nope.pt"):
        locate_model(serve_cfg, settings)


def test_unknown_source_is_rejected(serve_cfg):
    with pytest.raises(ValueError, match="source"):
        locate_model(serve_cfg, serve_settings(serve_cfg, {"EN_GU_MODEL_SOURCE": "ftp"}))


def test_translator_keeps_the_paragraph_layout(serve_cfg):
    translator = Translator.load(serve_cfg, serve_settings(serve_cfg, {}))
    result = translator.translate_text("The train is late. The bus is early.\n\nA big house.", beam_size=2)
    assert result.n_sentences == 3
    assert result.translation.count("\n") == 2 and result.latency_ms > 0
    info = translator.info()
    assert info["model_version"] == translator.version and info["architecture"]["vocab_size"] == 400
    assert info["metrics"] == {"test_bleu": 1.5}


def test_registry_source_serves_the_champion(serve_cfg, tracking_cfg, app_checkpoint, tmp_path):
    pytest.importorskip("mlflow")
    from en_gu.config import deep_merge
    from en_gu.registry import import_run, make_champion, register_version
    from en_gu.serving import fetch_model

    cfg = type(serve_cfg)(deep_merge(serve_cfg, {"tracking": tracking_cfg["tracking"]}))
    run_id = import_run(cfg, app_checkpoint, run_name="r")
    version = register_version(cfg, run_id)
    make_champion(cfg, version.version)

    path, label = locate_model(cfg, serve_settings(cfg, {"EN_GU_MODEL_SOURCE": "registry"}))
    assert label == f"v{version.version}" and path.exists()

    out, meta = fetch_model(cfg, tmp_path / "models" / "best.pt")
    assert meta["version"] == label
    fetched = serve_settings(cfg, {"EN_GU_MODEL_PATH": str(out)})
    assert locate_model(cfg, fetched)[1] == label  # the sidecar carries the registry version along
