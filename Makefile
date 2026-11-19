PY ?= python

.PHONY: install lint format test smoke smoke-test tiny-model prepare tokenizer train evaluate export pipeline fetch-model serve ui up down

install:
	$(PY) -m pip install -e ".[serve,train,dev]"

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check src tests

format:
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format src tests

test:
	$(PY) -m pytest

# CI helpers: a tiny random checkpoint, and the HTTP smoke test against a running API (URL=http://host:port)
URL ?= http://127.0.0.1:8000
tiny-model:
	$(PY) -m en_gu make-tiny-model --out models/tiny.pt

smoke-test:
	$(PY) -m en_gu smoke-test --url $(URL)

# Tiny end-to-end run on CPU; writes to data/smoke/ only
smoke:
	$(PY) -m en_gu all --config configs/base.yaml --config configs/smoke.yaml

prepare tokenizer train evaluate export:
	$(PY) -m en_gu $@

pipeline:
	$(PY) -m en_gu all

# Serving: copy the registry champion to models/best.pt, then run the API (and UI) locally or in Docker
fetch-model:
	$(PY) -m en_gu fetch-model

serve:
	EN_GU_MODEL_PATH=models/best.pt $(PY) -m en_gu serve

ui:
	$(PY) -m streamlit run app.py

up:
	docker compose up --build -d

down:
	docker compose down
