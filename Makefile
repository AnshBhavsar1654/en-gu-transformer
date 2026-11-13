PY ?= python

.PHONY: install lint format test smoke prepare tokenizer train evaluate export pipeline

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

# Tiny end-to-end run on CPU; writes to data/smoke/ only
smoke:
	$(PY) -m en_gu all --config configs/base.yaml --config configs/smoke.yaml

prepare tokenizer train evaluate export:
	$(PY) -m en_gu $@

pipeline:
	$(PY) -m en_gu all
