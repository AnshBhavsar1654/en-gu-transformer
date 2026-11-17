"""Streamlit front end for the English -> Gujarati Transformer trained by the en_gu pipeline.

By default the app is a thin client of the inference API (`python -m en_gu serve`, or `docker compose up`), set by
EN_GU_API_URL. With STANDALONE=1 it loads the model in-process instead, for quick local use without the API.
"""

import csv
import html
import io
import os
import sys
import time
from datetime import datetime

import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    import en_gu  # noqa: F401
except ModuleNotFoundError:  # package not pip-installed: use the copy in this repo
    sys.path.insert(0, os.path.join(APP_DIR, "src"))

STANDALONE = os.environ.get("STANDALONE", "0").lower() in ("1", "true", "yes")
API_URL = os.environ.get("EN_GU_API_URL", "http://localhost:8000").rstrip("/")
API_TIMEOUT = float(os.environ.get("EN_GU_API_TIMEOUT", "120"))

# Standalone mode only
MODEL_PATH = os.path.join(APP_DIR, "best.pt")  # downloaded from the Kaggle training output
# Only for best.pt files from runs before the checkpoint became self-contained
SPM_PATH = os.path.join(APP_DIR, "spm_en_gu.model")
HF_REPO_ID = "ansh164/english-gujarati-transformer"  # used when best.pt isn't next to app.py
FEEDBACK_DB = os.path.join(APP_DIR, "data", "feedback", "feedback.db")

MAX_CHARS = 2000
HISTORY_SIZE = 20

EXAMPLES = [
    "I am feeling very hungry.",
    "Can I get your contact number?",
    "What is your name?",
    "The train will arrive at the station in ten minutes.",
    "India won the cricket match by five wickets.",
    "Please drink plenty of water during summer.",
]

st.set_page_config(page_title="English → Gujarati Translator", page_icon="🌐", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Noto+Sans+Gujarati:wght@400;600&display=swap');

    :root { --accent: #e8590c; --accent-2: #f59f00; }
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 2rem; max-width: 1200px; }

    .hero { margin-bottom: 1.5rem; }
    .hero h1 {
        font-size: 2.6rem; font-weight: 800; margin: 0; line-height: 1.15;
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .hero h1 .gu { font-family: 'Noto Sans Gujarati', sans-serif; }
    .hero p { opacity: 0.7; margin: 0.35rem 0 0; font-size: 1.05rem; }

    .pane-label {
        font-size: 0.8rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
        opacity: 0.65; margin-bottom: 0.4rem;
    }
    .output-card {
        min-height: 238px; padding: 1rem 1.15rem; border-radius: 0.6rem;
        border: 1px solid rgba(128, 128, 128, 0.28);
        background: rgba(232, 89, 12, 0.05);
        font-family: 'Noto Sans Gujarati', sans-serif; font-size: 1.3rem; line-height: 1.75;
        white-space: pre-wrap; word-wrap: break-word;
    }
    .output-card.empty { opacity: 0.45; font-family: 'Inter', sans-serif; font-size: 1rem; }
    .stats { font-size: 0.85rem; opacity: 0.65; margin-top: 0.5rem; }

    .history-item {
        padding: 0.75rem 1rem; border-radius: 0.5rem; margin-bottom: 0.6rem;
        border-left: 3px solid var(--accent); background: rgba(128, 128, 128, 0.07);
    }
    .history-item .en { opacity: 0.8; }
    .history-item .gu { font-family: 'Noto Sans Gujarati', sans-serif; font-size: 1.12rem; margin-top: 0.2rem; }
    .history-item .meta { font-size: 0.75rem; opacity: 0.5; margin-top: 0.25rem; }

    textarea { font-size: 1.05rem !important; line-height: 1.6 !important; }
    div[data-testid="stMetricValue"] { font-size: 1.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


class BackendError(Exception):
    """A translation backend problem, with a message that can be shown to the user as is."""


class ApiBackend:
    """Talks to the FastAPI service."""

    mode = "API"

    def __init__(self, url):
        import httpx

        self.url = url
        self.client = httpx.Client(base_url=url, timeout=API_TIMEOUT)

    def _call(self, method, path, **kwargs):
        import httpx

        try:
            response = self.client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise BackendError(f"Cannot reach the translation API at {self.url} ({type(exc).__name__}).") from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise BackendError(f"{detail} (HTTP {response.status_code})")
        return response.json()

    def info(self):
        return self._call("GET", "/info")

    def translate(self, text, beam_size):
        return self._call("POST", "/translate", json={"text": text, "beam_size": beam_size})

    def feedback(self, payload):
        return self._call("POST", "/feedback", json=payload)


class LocalBackend:
    """Loads the model into this process (STANDALONE=1)."""

    mode = "standalone"

    def __init__(self):
        self.translator = load_local_translator(local_model_mtime())
        from en_gu.feedback import FeedbackStore

        self.store = FeedbackStore(FEEDBACK_DB, retention_days=90)

    def info(self):
        return self.translator.info()

    def translate(self, text, beam_size):
        translation, n_sentences, latency_ms = cached_local_translation(text, beam_size, local_model_mtime())
        import uuid

        return {
            "request_id": uuid.uuid4().hex,
            "translation": translation,
            "model_version": self.translator.version,
            "latency_ms": latency_ms,
            "n_sentences": n_sentences,
        }

    def feedback(self, payload):
        corrected = payload.get("corrected_translation")
        text = (payload.get("source_text"), payload.get("translation")) if corrected else (None, None)
        self.store.add(payload["request_id"], payload["rating"], payload.get("model_version"), *text, corrected)
        return {"stored_text": corrected is not None}


def local_model_mtime():
    """Changes whenever best.pt is replaced, so a newer download is picked up without a restart."""
    return os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


@st.cache_resource(show_spinner="Loading translation model…", max_entries=1)
def load_local_translator(mtime):
    import torch

    from en_gu.model import load_for_inference
    from en_gu.serving import Translator, file_version

    start = time.perf_counter()
    path = MODEL_PATH
    if mtime is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(HF_REPO_ID, os.path.basename(MODEL_PATH))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, sp, metrics = load_for_inference(path, device, spm_path=SPM_PATH)
    source = "path" if mtime is not None else "hub"
    return Translator(model, sp, file_version(path), metrics, device, time.perf_counter() - start, source)


@st.cache_data(show_spinner=False, max_entries=512)
def cached_local_translation(text, beam_size, mtime):
    result = load_local_translator(mtime).translate_text(text, beam_size)
    return result.translation, result.n_sentences, result.latency_ms


@st.cache_resource(max_entries=1)
def get_api_backend(url):
    return ApiBackend(url)


def get_backend():
    return LocalBackend() if STANDALONE else get_api_backend(API_URL)


def init_state():
    defaults = {"src_text": "", "result": None, "history": [], "auto_translate": False}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def use_example(sentence):
    st.session_state.src_text = sentence
    st.session_state.auto_translate = True


def clear_input():
    st.session_state.src_text = ""
    st.session_state.result = None


def run_translation(backend, text, beam_size):
    start = time.perf_counter()
    try:
        response = backend.translate(text, beam_size)
    except BackendError as exc:
        st.error(f"Translation failed: {exc}")
        return
    elapsed = time.perf_counter() - start
    st.session_state.result = {
        "english": text,
        "gujarati": response["translation"],
        "seconds": elapsed,
        "sentences": response["n_sentences"],
        "beam": beam_size,
        "request_id": response["request_id"],
        "model_version": response["model_version"],
        "feedback": None,
    }
    history = st.session_state.history
    if not history or history[0]["english"] != text:
        history.insert(
            0, {"english": text, "gujarati": response["translation"], "time": datetime.now().strftime("%H:%M:%S")}
        )
        del history[HISTORY_SIZE:]


def send_feedback(backend, rating, correction=None):
    result = st.session_state.result
    payload = {"request_id": result["request_id"], "rating": rating, "model_version": result["model_version"]}
    if correction:
        payload.update(corrected_translation=correction, source_text=result["english"], translation=result["gujarati"])
    try:
        backend.feedback(payload)
    except BackendError as exc:
        st.toast(f"Feedback not sent: {exc}", icon="⚠️")
        return
    result["feedback"] = "correction" if correction else rating
    st.toast("Thanks for the feedback!", icon="🙏")


def history_csv():
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["time", "english", "gujarati"])
    writer.writeheader()
    writer.writerows(st.session_state.history)
    return buffer.getvalue().encode("utf-8-sig")  # BOM so Excel shows Gujarati correctly


def render_sidebar(backend, info):
    with st.sidebar:
        st.header("⚙️ Settings")
        limits = info.get("limits", {})
        beam_size = st.slider(
            "Beam size",
            limits.get("min_beam", 1),
            limits.get("max_beam", 8),
            limits.get("default_beam", 5),
            help="How many candidate translations are explored. 1 = greedy (fastest); 4–6 is usually best.",
        )

        st.divider()
        st.header("🧠 Model")
        metrics, arch = info.get("metrics", {}), info.get("architecture", {})
        col1, col2 = st.columns(2)
        col1.metric("Parameters", f"{info['parameters'] / 1e6:.0f}M")
        col2.metric("Device", info["device"].upper())
        if metrics.get("test_bleu") is not None:
            col1.metric("Test BLEU", f"{metrics['test_bleu']:.1f}")
            col2.metric("Test chrF++", f"{metrics['test_chrf']:.1f}")
        st.caption(
            f"Transformer encoder-decoder · {arch.get('encoder_layers')}+{arch.get('decoder_layers')} layers · "
            f"d_model {arch.get('d_model')} · SentencePiece vocab {arch.get('vocab_size', 0):,}"
        )
        where = f"API at {API_URL}" if backend.mode == "API" else "in-process (standalone)"
        st.caption(f"Model version `{info['model_version']}` · served {where}")

        st.divider()
        if st.session_state.history:
            st.download_button(
                "⬇️ Download history (CSV)", history_csv(), "translations.csv", "text/csv", use_container_width=True
            )
            if st.button("🗑️ Clear history", use_container_width=True):
                st.session_state.history = []
                st.rerun()
    return beam_size


def render_feedback(backend, result):
    if result["feedback"]:
        st.caption("✅ Feedback recorded, thank you.")
        return
    c1, c2, _ = st.columns([1, 1, 4])
    if c1.button("👍", key=f"up_{result['request_id']}", help="Good translation", use_container_width=True):
        send_feedback(backend, "up")
        st.rerun()
    if c2.button("👎", key=f"down_{result['request_id']}", help="Poor translation", use_container_width=True):
        send_feedback(backend, "down")
        st.rerun()
    with st.expander("✏️ Suggest a better translation"):
        with st.form(f"correction_{result['request_id']}", border=False):
            correction = st.text_area("Your translation", value=result["gujarati"], height=120)
            st.caption(
                "Submitting stores your English text, the model's output and your correction (for up to 90 days) "
                "so they can be reviewed and used to improve the model. Thumbs up/down store no text."
            )
            if st.form_submit_button("Submit correction"):
                correction = correction.strip()
                if correction and correction != result["gujarati"].strip():
                    send_feedback(backend, "down", correction)
                    st.rerun()
                else:
                    st.toast("Edit the translation before submitting.", icon="✍️")


def main():
    init_state()

    st.markdown(
        '<div class="hero"><h1>English → <span class="gu">ગુજરાતી</span></h1>'
        "<p>Neural machine translation with a Transformer trained from scratch</p></div>",
        unsafe_allow_html=True,
    )

    try:
        backend = get_backend()
        info = backend.info()
    except BackendError as exc:
        st.error(f"The translation service is not available: {exc}\n\nStart it with `python -m en_gu serve` or "
                 "`docker compose up`, or run the app with `STANDALONE=1` to load the model in-process.")
        if st.button("Retry"):
            st.rerun()
        st.stop()
    except Exception as exc:
        st.error(
            f"Could not load the model: {exc}\n\n"
            "Download `best.pt` from the Kaggle notebook output and put it next to `app.py`."
        )
        st.stop()

    beam_size = render_sidebar(backend, info)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="pane-label">English</div>', unsafe_allow_html=True)
        with st.form("translate_form", border=False):
            st.text_area(
                "English text",
                key="src_text",
                height=238,
                max_chars=info.get("limits", {}).get("max_chars", MAX_CHARS),
                label_visibility="collapsed",
                placeholder="Type or paste English text…  (Ctrl+Enter to translate)",
            )
            b1, b2, _ = st.columns([2, 1, 3])
            submitted = b1.form_submit_button("Translate ➜", type="primary", use_container_width=True)
            b2.form_submit_button("Clear", on_click=clear_input, use_container_width=True)

    text = st.session_state.src_text.strip()
    if submitted or st.session_state.auto_translate:
        st.session_state.auto_translate = False
        if text:
            with right, st.spinner("Translating…"):
                run_translation(backend, text, beam_size)
        elif submitted:
            st.toast("Enter some English text first.", icon="✍️")

    with right:
        st.markdown('<div class="pane-label">Gujarati</div>', unsafe_allow_html=True)
        result = st.session_state.result
        if result:
            st.markdown(f'<div class="output-card">{html.escape(result["gujarati"])}</div>', unsafe_allow_html=True)
            sentences = f"{result['sentences']} sentence{'s' if result['sentences'] != 1 else ''}"
            st.markdown(
                f'<div class="stats">⏱ {result["seconds"]:.2f}s · {sentences} · beam {result["beam"]} · '
                f"model {html.escape(result['model_version'])}</div>",
                unsafe_allow_html=True,
            )
            render_feedback(backend, result)
            with st.expander("Copy text"):
                st.code(result["gujarati"], language=None, wrap_lines=True)
        else:
            st.markdown(
                '<div class="output-card empty">Your translation will appear here.</div>', unsafe_allow_html=True
            )

    st.markdown("#### ✨ Try an example")
    cols = st.columns(3)
    for i, sentence in enumerate(EXAMPLES):
        cols[i % 3].button(
            sentence, key=f"example_{i}", on_click=use_example, args=(sentence,), use_container_width=True
        )

    if st.session_state.history:
        st.markdown("#### 🕘 Recent translations")
        for item in st.session_state.history[:5]:
            st.markdown(
                f'<div class="history-item"><div class="en">{html.escape(item["english"])}</div>'
                f'<div class="gu">{html.escape(item["gujarati"])}</div>'
                f'<div class="meta">{item["time"]}</div></div>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
