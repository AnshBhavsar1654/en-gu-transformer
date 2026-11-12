"""Streamlit front end for the English -> Gujarati Transformer trained in the notebook."""
import csv
import html
import io
import os
import re
import time
from datetime import datetime

import streamlit as st
import torch

from nmt_model import load_for_inference, translate

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "best.pt")  # downloaded from the Kaggle notebook output
# Only for best.pt files from runs before the notebook made them self-contained
SPM_PATH = os.path.join(APP_DIR, "spm_en_gu.model")
HF_REPO_ID = "anshbhavsar2674/english-gujarati-transformer"  # used when best.pt isn't next to app.py
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


def model_version():
    """Changes whenever best.pt is replaced, so a newer download is picked up without a restart."""
    return os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


@st.cache_resource(show_spinner="Loading translation model…", max_entries=1)
def load_model(version):
    path = MODEL_PATH
    if version is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(HF_REPO_ID, os.path.basename(MODEL_PATH))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, sp, metrics = load_for_inference(path, device, spm_path=SPM_PATH)
    return model, sp, metrics, device


SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@st.cache_data(show_spinner=False, max_entries=512)
def translate_text(text, beam_size, version):
    """The model is trained on single sentences, so split paragraphs into sentences first."""
    model, sp, _, _ = load_model(version)
    layout, sentences = [], []
    for paragraph in text.split("\n"):
        parts = [s for s in SENTENCE_END.split(paragraph.strip()) if s]
        layout.append(len(parts))
        sentences.extend(parts)
    outputs = iter(translate(model, sp, sentences, beam_size=beam_size)) if sentences else iter(())
    return "\n".join(" ".join(next(outputs) for _ in range(n)) for n in layout), len(sentences)


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


def run_translation(text, beam_size):
    start = time.perf_counter()
    translation, n_sentences = translate_text(text, beam_size, model_version())
    elapsed = time.perf_counter() - start
    st.session_state.result = {
        "english": text, "gujarati": translation, "seconds": elapsed,
        "sentences": n_sentences, "beam": beam_size,
    }
    history = st.session_state.history
    if not history or history[0]["english"] != text:
        history.insert(0, {"english": text, "gujarati": translation,
                           "time": datetime.now().strftime("%H:%M:%S")})
        del history[HISTORY_SIZE:]


def history_csv():
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["time", "english", "gujarati"])
    writer.writeheader()
    writer.writerows(st.session_state.history)
    return buffer.getvalue().encode("utf-8-sig")  # BOM so Excel shows Gujarati correctly


def render_sidebar(model, metrics, device):
    with st.sidebar:
        st.header("⚙️ Settings")
        beam_size = st.slider(
            "Beam size", 1, 8, 5,
            help="How many candidate translations are explored. 1 = greedy (fastest); 4–6 is usually best.",
        )

        st.divider()
        st.header("🧠 Model")
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        col1, col2 = st.columns(2)
        col1.metric("Parameters", f"{n_params:.0f}M")
        col2.metric("Device", device.upper())
        if metrics.get("test_bleu") is not None:
            col1.metric("Test BLEU", f"{metrics['test_bleu']:.1f}")
            col2.metric("Test chrF++", f"{metrics['test_chrf']:.1f}")
        st.caption(
            f"Transformer encoder-decoder · {len(model.encoder_layers)}+{len(model.decoder_layers)} layers · "
            f"d_model {model.d_model} · SentencePiece vocab {model.embedding.num_embeddings:,}"
        )

        st.divider()
        if st.session_state.history:
            st.download_button("⬇️ Download history (CSV)", history_csv(), "translations.csv",
                               "text/csv", use_container_width=True)
            if st.button("🗑️ Clear history", use_container_width=True):
                st.session_state.history = []
                st.rerun()
    return beam_size


def main():
    init_state()

    st.markdown(
        '<div class="hero"><h1>English → <span class="gu">ગુજરાતી</span></h1>'
        "<p>Neural machine translation with a Transformer trained from scratch</p></div>",
        unsafe_allow_html=True,
    )

    try:
        model, _, metrics, device = load_model(model_version())
    except Exception as exc:
        st.error(
            f"Could not load the model: {exc}\n\n"
            "Download `best.pt` from the Kaggle notebook output and put it next to `app.py`."
        )
        st.stop()

    beam_size = render_sidebar(model, metrics, device)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="pane-label">English</div>', unsafe_allow_html=True)
        with st.form("translate_form", border=False):
            st.text_area(
                "English text", key="src_text", height=238, max_chars=MAX_CHARS,
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
                run_translation(text, beam_size)
        elif submitted:
            st.toast("Enter some English text first.", icon="✍️")

    with right:
        st.markdown('<div class="pane-label">Gujarati</div>', unsafe_allow_html=True)
        result = st.session_state.result
        if result:
            st.markdown(f'<div class="output-card">{html.escape(result["gujarati"])}</div>',
                        unsafe_allow_html=True)
            sentences = f"{result['sentences']} sentence{'s' if result['sentences'] != 1 else ''}"
            st.markdown(
                f'<div class="stats">⏱ {result["seconds"]:.2f}s · {sentences} · beam {result["beam"]}</div>',
                unsafe_allow_html=True,
            )
            with st.expander("Copy text"):
                st.code(result["gujarati"], language=None, wrap_lines=True)
        else:
            st.markdown('<div class="output-card empty">Your translation will appear here.</div>',
                        unsafe_allow_html=True)

    st.markdown("#### ✨ Try an example")
    cols = st.columns(3)
    for i, sentence in enumerate(EXAMPLES):
        cols[i % 3].button(sentence, key=f"example_{i}", on_click=use_example, args=(sentence,),
                           use_container_width=True)

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
