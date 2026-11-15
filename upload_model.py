# upload_model.py
"""Publish best.pt to the Hugging Face Hub as one commit: weights, model card (README.md) and config.json."""
import json
import os
import sys

import torch
from huggingface_hub import CommitOperationAdd, EvalResult, HfApi, ModelCardData, login

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "best.pt")  # self-contained: config + weights + tokenizer + metrics
CARD_SOURCE = os.path.join(HERE, "MODEL_CARD.md")  # the repo's model card; the Hub page is built from it

# Keep in sync with HF_REPO_ID in app.py
REPO_ID = "ansh164/english-gujarati-transformer"
GITHUB_URL = "https://github.com/AnshBhavsar1654/en-gu-transformer"
DATASET_NAME = "English to Gujarati Machine Translation Dataset (Kaggle), held-out test split of 2,000 sentences"

USAGE = f"""\
## How to use

The checkpoint is a custom PyTorch format (not `transformers`). The model code is in the
[GitHub repository]({GITHUB_URL}).

```bash
pip install torch sentencepiece huggingface_hub
pip install git+{GITHUB_URL}
```

```python
from huggingface_hub import hf_hub_download
from en_gu.model import load_for_inference, translate

path = hf_hub_download("{REPO_ID}", "best.pt")
model, sp, metrics = load_for_inference(path)  # weights, config and tokenizer all come from this one file
print(translate(model, sp, ["Can I get your contact number?"], beam_size=5))
```

Example translations (beam 5):

| English | Gujarati |
|---|---|
| Can I get your contact number? | શું હું તમારો સંપર્ક નંબર મેળવી શકું છું? |
| What is your name? | તમારું નામ શું છે? |
| The government has announced new schemes for farmers. | સરકારે ખેડૂતો માટે નવી યોજનાઓની જાહેરાત કરી છે. |
| India won the cricket match by five wickets. | ભારતે આ મેચ પાંચ વિકેટથી જીતી હતી. |

A Streamlit demo app is in the same repository.

"""


def resolve_token():
    """HF_TOKEN env var, else hf_token.txt next to this script (gitignored), else an interactive login."""
    token = os.environ.get("HF_TOKEN")
    token_file = os.path.join(HERE, "hf_token.txt")
    if token is None and os.path.exists(token_file):
        with open(token_file, encoding="utf-8") as f:
            token = f.read().strip() or None
    if token is None:
        login()
    return token


def build_card(metrics):
    """Hub model card: YAML metadata (language, task, evaluation results) + usage + the repo's MODEL_CARD.md."""
    results = [
        EvalResult(
            task_type="translation",
            dataset_type="english-to-gujarati-machine-translation-dataset",
            dataset_name=DATASET_NAME,
            metric_type=metric_type,
            metric_value=metrics[key],
        )
        for metric_type, key in (("bleu", "test_bleu"), ("chrf", "test_chrf"), ("ter", "test_ter"))
        if key in metrics
    ]
    data = ModelCardData(
        language=["en", "gu"],
        library_name="pytorch",
        pipeline_tag="translation",
        tags=["translation", "machine-translation", "transformer", "english-gujarati", "from-scratch"],
        model_name="English to Gujarati Transformer",
        eval_results=results,
    )
    with open(CARD_SOURCE, encoding="utf-8") as f:
        body = f.read().split("\n", 1)[1].lstrip("\n")  # drop its own title line; the Hub shows model_name
    return f"---\n{data.to_yaml()}\n---\n\n# English to Gujarati Transformer\n\n{USAGE}{body}"


def build_config(checkpoint):
    return {
        "architecture": "Transformer encoder-decoder (pre-LayerNorm, shared embeddings)",
        "model_config": checkpoint["config"],
        "tokenizer": {
            "type": "sentencepiece-unigram",
            "vocab_size": checkpoint["config"]["vocab_size"],
            "byte_fallback": True,
            "special_ids": {"pad": 0, "unk": 1, "bos": 2, "eos": 3},
            "note": "the tokenizer model is stored inside best.pt",
        },
        "checkpoint_format": "best.pt = config + fp16 weights + tokenizer + metrics",
        "library": GITHUB_URL,
        "metrics": checkpoint.get("metrics", {}),
    }


def main():
    api = HfApi(token=resolve_token())

    # Make sure the token belongs to the account that owns the repo (e.g. not a token from an old account)
    account, owner = api.whoami()["name"], REPO_ID.split("/")[0]
    if account != owner:
        sys.exit(f"This token belongs to '{account}', but {REPO_ID} is owned by '{owner}'.")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    card = build_card(checkpoint.get("metrics", {}))
    config = json.dumps(build_config(checkpoint), indent=2, ensure_ascii=False)
    del checkpoint

    try:
        # Only create the repo if it's missing, so a token that can write to this one repo is enough
        if not api.repo_exists(REPO_ID, repo_type="model"):
            api.create_repo(repo_id=REPO_ID, repo_type="model", private=False)
        api.create_commit(
            repo_id=REPO_ID,
            repo_type="model",
            operations=[
                CommitOperationAdd("best.pt", MODEL_PATH),  # app.py downloads this name
                CommitOperationAdd("README.md", card.encode("utf-8")),
                CommitOperationAdd("config.json", config.encode("utf-8")),
            ],
            commit_message="Upload English to Gujarati Transformer (run 1)",
        )
        print(f"Model uploaded: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        sys.exit(f"Upload failed: {e}")


if __name__ == "__main__":
    main()
