"""Sample successful translations against a private candidate without delaying the live response."""

import json
import os
import queue
import random
import threading
import time
import urllib.parse
import urllib.request
from difflib import SequenceMatcher

from en_gu.telemetry import emit


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ShadowSampler:
    def __init__(self, url, rate):
        self.url = url.rstrip("/")
        self.rate = rate
        self.pending = queue.Queue(maxsize=8)
        self.closed = threading.Event()
        self.opener = urllib.request.build_opener(NoRedirect())
        self.token = None
        self.token_until = 0
        parsed = urllib.parse.urlsplit(self.url)
        self.cloud = parsed.scheme == "https" and bool(parsed.hostname and parsed.hostname.endswith(".run.app"))
        local = parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost", "::1")
        if not (self.cloud or local) or parsed.username or parsed.path or parsed.query or parsed.fragment:
            raise ValueError("Shadow target must be a Cloud Run HTTPS origin or a local loopback HTTP origin")
        if not 0 < rate <= 1:
            raise ValueError("Shadow sampling rate must be between 0 and 1")
        self.worker = threading.Thread(target=self.run, name="shadow-translations", daemon=True)
        self.worker.start()

    @classmethod
    def from_env(cls):
        url = os.environ.get("EN_GU_SHADOW_URL")
        return cls(url, float(os.environ.get("EN_GU_SHADOW_RATE", "0.05"))) if url else None

    def offer(self, text, beam, result, stable_version):
        if random.random() < self.rate:
            try:
                self.pending.put_nowait((text, beam, result.translation, result.latency_ms, stable_version))
            except queue.Full:
                emit("shadow_dropped", stable_version=stable_version)

    def authorization(self):
        if time.monotonic() >= self.token_until:
            audience = urllib.parse.urlencode({"audience": self.url})
            metadata = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity"
            req = urllib.request.Request(f"{metadata}?{audience}", headers={"Metadata-Flavor": "Google"})
            with self.opener.open(req, timeout=2) as response:
                self.token = response.read().decode()
            self.token_until = time.monotonic() + 600
        return f"Bearer {self.token}"

    def run(self):
        while not self.closed.is_set():
            try:
                text, beam, live, live_latency, stable_version = self.pending.get(timeout=0.5)
            except queue.Empty:
                continue
            fields = {"stable_version": stable_version, "stable_latency_ms": live_latency, "ok": False}
            try:
                headers = {"Content-Type": "application/json", "User-Agent": "en-gu-shadow"}
                if self.cloud:
                    headers["Authorization"] = self.authorization()
                data = json.dumps({"text": text, "beam_size": beam}).encode()
                req = urllib.request.Request(f"{self.url}/translate", data=data, headers=headers)
                with self.opener.open(req, timeout=8) as response:
                    candidate = json.load(response)
                similarity = SequenceMatcher(None, live[:512], candidate["translation"][:512], autojunk=False).ratio()
                fields.update(
                    candidate_version=candidate["model_version"],
                    candidate_latency_ms=candidate["latency_ms"],
                    similarity=similarity,
                    ok=True,
                )
            except (OSError, ValueError, TypeError, KeyError):
                pass
            emit("shadow_comparison", **fields)

    def close(self):
        self.closed.set()
        self.worker.join(timeout=0.6)
