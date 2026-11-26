"""Bound inference concurrency and validate request budgets before parsing JSON."""

import asyncio
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from starlette.datastructures import Headers
from starlette.responses import JSONResponse

DEFAULT_GUARDRAILS = {
    "max_translate_body_bytes": 32768,
    "max_feedback_body_bytes": 163840,
    "translate_per_minute": 600,
    "translate_burst": 24,
    "feedback_per_minute": 60,
    "feedback_burst": 12,
    "max_queued_translations": 2,
    "translation_timeout_seconds": 45,
}


def request_budgets(settings):
    config = {**DEFAULT_GUARDRAILS, **settings.get("guardrails", {})}
    for key in DEFAULT_GUARDRAILS:
        value = config[key]
        if key == "translation_timeout_seconds":
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be a positive number")
        else:
            lower_bound = 0 if key == "max_queued_translations" else 1
            if type(value) is not int or value < lower_bound:
                raise ValueError(f"{key} must be an integer greater than or equal to {lower_bound}")
    return config


class TokenBucket:
    """A shared, process-local budget for an endpoint."""

    def __init__(self, per_minute, burst):
        self.rate = per_minute / 60
        self.burst = burst
        self.tokens = burst
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def retry_after(self):
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.burst, self.tokens + (now - self.updated) * self.rate)
            self.updated = now
            if self.tokens >= 1:
                self.tokens -= 1
                return None
            return max(1, math.ceil((1 - self.tokens) / self.rate))


class RequestLimits:
    def __init__(self, app, settings):
        self.app = app
        self.max_bytes = {f"/{route}": settings[f"max_{route}_body_bytes"] for route in ("translate", "feedback")}
        self.buckets = {
            f"/{route}": TokenBucket(settings[f"{route}_per_minute"], settings[f"{route}_burst"])
            for route in ("translate", "feedback")
        }

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in self.buckets:
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        limit = self.max_bytes[scope["path"]]
        declared = (headers.get("content-length") or "").lstrip("0")
        if (
            declared
            and declared.isascii()
            and declared.isdecimal()
            and (len(declared) > len(str(limit)) or int(declared) > limit)
        ):
            await self.too_large(scope, receive, send, limit)
            return

        wait = self.buckets[scope["path"]].retry_after()
        if wait is not None:
            response = JSONResponse(
                {"detail": "Request rate limit exceeded."}, status_code=429, headers={"Retry-After": str(wait)}
            )
            await response(scope, receive, send)
            return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > limit:
                await self.too_large(scope, receive, send, limit)
                return
            body.extend(chunk)
            if not message.get("more_body"):
                break

        delivered = False

        async def replay():
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)

    async def too_large(self, scope, receive, send, limit):
        response = JSONResponse({"detail": f"Request body exceeds the {limit}-byte limit."}, status_code=413)
        await response(scope, receive, send)


class QueueFull(Exception):
    pass


class InferenceQueue:
    """A running translation occupies its slot until its worker actually exits."""

    def __init__(self, queued, timeout):
        self.timeout = timeout
        self.slots = threading.BoundedSemaphore(1 + queued)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="translation")

    async def translate(self, translator, text, beam):
        if not self.slots.acquire(blocking=False):
            raise QueueFull()
        try:
            job = self.executor.submit(translator.translate_text, text, beam_size=beam, observe=True)
        except RuntimeError:
            self.slots.release()
            raise
        job.add_done_callback(lambda _: self.slots.release())
        result = asyncio.wrap_future(job)
        try:
            return await asyncio.wait_for(asyncio.shield(result), self.timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            job.cancel()  # cancel queued work. A running worker releases its slot via the callback.
            result.add_done_callback(observe_exception)
            raise

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)


def observe_exception(result):
    if not result.cancelled():
        result.exception()
