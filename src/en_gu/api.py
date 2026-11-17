"""FastAPI inference service.

    POST /translate   {text, beam_size?} -> {request_id, translation, model_version, latency_ms, n_sentences}
    GET  /health      the process is up (liveness)
    GET  /ready       the model is loaded and requests can be served (readiness)
    GET  /info        model details and request limits

Run it with `python -m en_gu serve` or `uvicorn --factory en_gu.api:create_app`. The model loads in a background
thread at startup, so /health answers immediately and /ready turns true once the model is in memory. Every response
carries an `X-Request-ID` header, and `X-Model-Version` once a model is loaded.
"""

import logging
import os
import re
import threading
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from en_gu import __version__
from en_gu.config import load_config
from en_gu.serving import SERVE_CONFIGS, Translator, serve_settings
from en_gu.textutils import split_sentences

log = logging.getLogger(__name__)
REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class TranslateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="English text; several sentences and lines are fine")
    beam_size: int | None = Field(default=None, description="1 = greedy; default from serve.limits.default_beam")


class TranslateResponse(BaseModel):
    request_id: str
    translation: str
    model_version: str
    latency_ms: float
    n_sentences: int


class ServiceState:
    """What the endpoints share: the translator once loaded, or the reason it failed to load."""

    def __init__(self):
        self.translator = None
        self.error = None
        self.loaded = threading.Event()

    def load(self, cfg, settings):
        try:
            self.translator = Translator.load(cfg, settings)
        except Exception as exc:  # stay up and report not-ready instead of crash-looping
            log.exception("Model failed to load")
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.loaded.set()

    def require_translator(self):
        if self.translator is None:
            detail = f"Model failed to load: {self.error}" if self.error else "Model is still loading"
            raise HTTPException(503, detail=detail, headers={"Retry-After": "5"})
        return self.translator


def default_config():
    paths = os.environ.get("EN_GU_CONFIG")
    return load_config(paths.split(",") if paths else SERVE_CONFIGS)


def create_app(cfg=None, translator=None, background=True):
    """Build the app. Tests pass a ready `translator`; `background=False` loads the model before serving."""
    cfg = cfg or default_config()
    settings = serve_settings(cfg)
    limits = settings["limits"]
    state = ServiceState()

    @asynccontextmanager
    async def lifespan(app):
        if translator is not None:
            state.translator = translator
            state.loaded.set()
        elif background:
            threading.Thread(target=state.load, args=(cfg, settings), name="model-loader", daemon=True).start()
        else:
            state.load(cfg, settings)
        yield

    app = FastAPI(
        title="English → Gujarati translation API",
        version=__version__,
        description="Transformer trained from scratch; see the repository README for the model and its limits.",
        lifespan=lifespan,
    )
    app.state.service = state
    app.state.cfg = cfg
    app.state.settings = settings

    @app.middleware("http")
    async def add_ids(request: Request, call_next):
        incoming = request.headers.get("X-Request-ID", "")
        request.state.request_id = incoming if REQUEST_ID.match(incoming) else uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        if state.translator is not None:
            response.headers["X-Model-Version"] = state.translator.version
        return response

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        if state.translator is None:
            body = {"ready": False, "error": state.error} if state.error else {"ready": False, "loading": True}
            return JSONResponse(body, status_code=503)
        return {"ready": True, "model_version": state.translator.version}

    @app.get("/info")
    def info():
        return {**state.require_translator().info(), "limits": limits, "api_version": __version__}

    @app.post("/translate", response_model=TranslateResponse)
    def translate(body: TranslateRequest, request: Request):
        text = body.text.strip()
        if not text:
            raise HTTPException(422, detail="text is empty")
        if len(text) > limits["max_chars"]:
            raise HTTPException(422, detail=f"text has {len(text)} characters; the limit is {limits['max_chars']}")
        n_sentences = len(split_sentences(text)[1])
        if n_sentences > limits["max_sentences"]:
            raise HTTPException(422, detail=f"text has {n_sentences} sentences; the limit is {limits['max_sentences']}")
        beam = limits["default_beam"] if body.beam_size is None else body.beam_size
        if not limits["min_beam"] <= beam <= limits["max_beam"]:
            raise HTTPException(422, detail=f"beam_size must be between {limits['min_beam']} and {limits['max_beam']}")

        translator = state.require_translator()
        result = translator.translate_text(text, beam_size=beam)
        return TranslateResponse(
            request_id=request.state.request_id,
            translation=result.translation,
            model_version=translator.version,
            latency_ms=round(result.latency_ms, 1),
            n_sentences=result.n_sentences,
        )

    return app
