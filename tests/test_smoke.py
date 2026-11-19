"""Model smoke test: the real `python -m en_gu serve` process on a tiny random model, checked over HTTP.

This is what CI runs before building the image, and the same checks (`en_gu.smoke`) later run against the
container and the deployed staging/production services.
"""

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from en_gu.cli import main as cli_main
from en_gu.smoke import run_smoke, wait_until_ready
from en_gu.tinymodel import make_tiny_checkpoint

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

ROOT = Path(__file__).resolve().parent.parent


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(model_path, feedback_db, build="abc123"):
    port = free_port()
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "EN_GU_MODEL_PATH": str(model_path),
        "EN_GU_FEEDBACK_DB": str(feedback_db),
        "EN_GU_BUILD_SHA": build,
    }
    cmd = [sys.executable, "-m", "en_gu", "serve", "--port", str(port), "--log-level", "warning"]
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return proc, f"http://127.0.0.1:{port}"


def stop(proc):
    proc.terminate()
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    return make_tiny_checkpoint(tmp_path_factory.mktemp("model") / "best.pt")


@pytest.fixture(scope="module")
def server(tiny_checkpoint, tmp_path_factory):
    proc, url = start_server(tiny_checkpoint, tmp_path_factory.mktemp("fb") / "feedback.db")
    try:
        wait_until_ready(url, timeout=90, interval=0.5)
    except AssertionError:
        stop(proc)
        pytest.fail(f"server did not start: {proc.stderr.read().decode(errors='replace')[-2000:]}")
    yield url
    stop(proc)


def test_tiny_checkpoint_loads_and_translates(tiny_checkpoint):
    from en_gu.model import load_for_inference, translate

    model, sp, metrics = load_for_inference(tiny_checkpoint, "cpu")
    assert sum(p.numel() for p in model.parameters()) < 100_000
    (out,) = translate(model, sp, ["The train is late."], beam_size=2)
    assert isinstance(out, str) and metrics["note"] == "tiny random model"


def test_make_tiny_model_command(tmp_path):
    out = tmp_path / "m" / "best.pt"
    assert cli_main(["make-tiny-model", "--out", str(out)]) == 0
    assert out.exists() and out.stat().st_size < 1_000_000


@pytest.mark.smoke
def test_all_smoke_checks_pass_against_a_real_server(server):
    lines = []
    assert run_smoke(server, timeout=30, expect_build="abc123", out=lines.append) == 0, "\n".join(lines)
    assert lines[-1].startswith("7/7 checks passed")


@pytest.mark.smoke
def test_smoke_test_command_fails_on_the_wrong_build(server, capsys):
    assert cli_main(["smoke-test", "--url", server, "--expect-build", "other", "--no-feedback"]) == 1
    out = capsys.readouterr().out
    assert "FAIL  info" in out and "PASS  translate" in out and "5/6 checks passed" in out


@pytest.mark.smoke
def test_smoke_test_fails_fast_when_the_model_cannot_load(tmp_path):
    proc, url = start_server(tmp_path / "missing.pt", tmp_path / "feedback.db")
    try:
        lines = []
        assert run_smoke(url, timeout=60, out=lines.append) > 0
        ready = next(line for line in lines if "ready" in line)
        assert ready.startswith("FAIL") and "missing.pt" in ready
    finally:
        stop(proc)


def test_smoke_test_reports_an_unreachable_api():
    lines = []
    assert run_smoke(f"http://127.0.0.1:{free_port()}", timeout=1, out=lines.append) == 7
    assert lines[0].startswith("FAIL  health")
