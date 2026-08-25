# Thin wrapper around `pramaan.cli` — the CLI is the single source of
# truth (see src/pramaan/cli.py). This exists for CI (ubuntu-latest has
# `make`) and for the VM. On this Windows dev machine (no `make` binary),
# use `python -m pramaan.cli <command>` directly — see README "Reproduce".

PY ?= python
SCALE ?= dev

.PHONY: setup data data-full train eval report serve all lint test

setup:
	$(PY) -m pip install -e ".[dev,docs]"
	$(PY) -m pramaan.cli setup

data:
	$(PY) -m pramaan.cli data --scale $(SCALE)

data-full:
	$(PY) -m pramaan.cli data-full

train:
	$(PY) -m pramaan.cli train --scale $(SCALE)

eval:
	$(PY) -m pramaan.cli eval --scale $(SCALE)

report:
	$(PY) -m pramaan.cli report --scale $(SCALE)

serve:
	$(PY) -m pramaan.cli serve

all:
	$(PY) -m pramaan.cli all --scale $(SCALE)

lint:
	$(PY) -m ruff check src tests eval benchmarks scripts
	$(PY) -m mypy src

test:
	$(PY) -m pytest
