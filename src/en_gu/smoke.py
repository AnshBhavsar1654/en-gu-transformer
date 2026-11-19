"""HTTP smoke test for a running inference API: local, in a CI container, or a deployed staging/production URL.

Uses only the standard library, so it runs on a bare CI runner without torch:

    python src/en_gu/smoke.py --url https://staging.example.run.app --expect-build "$GITHUB_SHA"
    python -m en_gu smoke-test --url http://127.0.0.1:8000

Checks, in order: /health answers (retried until the process is up); /ready turns true within --timeout;
/info reports the expected build;
/translate returns a translation with matching version headers; invalid input is rejected with 422;
/feedback stores a rating (skipped with --no-feedback, e.g. against production); /metrics counts the requests.
Exits 0 when every check passes and 1 otherwise, printing one line per check.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

SMOKE_TEXT = "The train is late.\nWhat is your name?"


class SmokeFailure(AssertionError):
    pass


def request(base_url, method, path, payload=None, headers=None, timeout=30):
    """(status, headers, body) for one request; HTTP errors are returned, not raised."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base_url.rstrip("/") + path, data=data, method=method)
    req.add_header("User-Agent", "en-gu-smoke-test")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.headers, resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        return err.code, err.headers, err.read().decode("utf-8", "replace")


def check(condition, message):
    if not condition:
        raise SmokeFailure(message)


def wait_until_up(base_url, timeout, interval=1.0):
    """Poll /health until the process answers (a new container or Cloud Run instance takes a moment)."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return request(base_url, "GET", "/health", timeout=10)
        except (urllib.error.URLError, OSError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(interval)


def wait_until_ready(base_url, timeout, interval=2.0):
    """Poll /health then /ready until both answer 200; return the /ready body."""
    deadline = time.monotonic() + timeout
    last = "no response"
    while time.monotonic() < deadline:
        try:
            status, _, body = request(base_url, "GET", "/ready", timeout=10)
            if status == 200:
                return json.loads(body)
            last = f"/ready -> {status} {body[:200]}"
            if status == 503 and '"error"' in body:  # the model failed to load: waiting won't help
                break
        except (urllib.error.URLError, OSError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(interval)
    raise SmokeFailure(f"API not ready after {timeout:.0f}s ({last})")


def run_smoke(base_url, timeout=180, expect_version=None, expect_build=None, feedback=True, out=print):
    """Run every check against `base_url`; returns the number of failed checks."""
    results = []
    context = {}

    def step(name, fn):
        start = time.perf_counter()
        try:
            detail = fn() or ""
            results.append(True)
            out(f"PASS  {name:<24} {detail} ({(time.perf_counter() - start) * 1000:.0f} ms)")
        except Exception as exc:  # report every check, not just the first failure
            results.append(False)
            out(f"FAIL  {name:<24} {type(exc).__name__}: {exc}")

    def health():
        status, _, body = wait_until_up(base_url, timeout)
        check(status == 200 and json.loads(body) == {"status": "ok"}, f"/health -> {status} {body[:200]}")

    def ready():
        body = wait_until_ready(base_url, timeout)
        context["version"] = body.get("model_version")
        if expect_version:
            check(body.get("model_version") == expect_version, f"model {body.get('model_version')} != {expect_version}")
        return f"model {body.get('model_version')}"

    def info():
        status, _, body = request(base_url, "GET", "/info")
        check(status == 200, f"/info -> {status} {body[:200]}")
        data = json.loads(body)
        check(data.get("parameters", 0) > 0, "/info reports no parameters")
        if expect_build:
            check(data.get("build") == expect_build, f"build {data.get('build')!r} != {expect_build!r}")
        return f"build {data.get('build')}"

    def translate():
        rid = f"smoke-{uuid.uuid4().hex[:12]}"
        status, headers, body = request(
            base_url, "POST", "/translate", {"text": SMOKE_TEXT, "beam_size": 2}, {"X-Request-ID": rid}
        )
        check(status == 200, f"/translate -> {status} {body[:200]}")
        data = json.loads(body)
        check(isinstance(data.get("translation"), str), "no translation in the response")
        check(data.get("n_sentences") == 2, f"n_sentences {data.get('n_sentences')} != 2")
        check(data.get("request_id") == rid == headers.get("X-Request-ID"), "request id not echoed")
        check(data.get("model_version") == headers.get("X-Model-Version"), "X-Model-Version header mismatch")
        if context.get("version"):
            check(data["model_version"] == context["version"], "model version changed since /ready")
        context["request_id"] = rid
        return f"{data['latency_ms']} ms, {ascii(data['translation'][:30])}"  # ascii: safe on any console

    def rejects_bad_input():
        for payload in ({"text": "   "}, {"text": "Hi.", "beam_size": 99}, {"txt": "Hi."}):
            status, _, body = request(base_url, "POST", "/translate", payload)
            check(status == 422, f"{payload} -> {status} {body[:200]}, expected 422")

    def store_feedback():
        payload = {"request_id": context.get("request_id", "smoke"), "rating": "up"}
        status, _, body = request(base_url, "POST", "/feedback", payload)
        check(status == 201 and json.loads(body)["stored_text"] is False, f"/feedback -> {status} {body[:200]}")

    def metrics():
        status, _, body = request(base_url, "GET", "/metrics")
        check(status == 200, f"/metrics -> {status}")
        for name in ("en_gu_http_requests_total", "en_gu_translate_seconds", "en_gu_model_ready 1.0"):
            check(name in body, f"/metrics has no {name}")

    step("health", health)
    step("ready", ready)
    step("info", info)
    step("translate", translate)
    step("rejects bad input", rejects_bad_input)
    if feedback:
        step("feedback", store_feedback)
    step("metrics", metrics)
    failed = results.count(False)
    out(f"{len(results) - failed}/{len(results)} checks passed against {base_url}")
    return failed


def build_parser(parser=None):
    parser = parser or argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="base URL of the API")
    parser.add_argument("--timeout", type=float, default=180, help="seconds to wait for /ready")
    parser.add_argument("--expect-version", default=None, help="fail unless this model version is served")
    parser.add_argument("--expect-build", default=None, help="fail unless /info reports this build (git SHA)")
    parser.add_argument("--no-feedback", action="store_true", help="don't write a feedback row (production)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    failed = run_smoke(args.url, args.timeout, args.expect_version, args.expect_build, feedback=not args.no_feedback)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
