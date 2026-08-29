# Thin wrapper around the `pramaan` CLI — the CLI is the single source of
# truth (see src/pramaan/cli.py). This exists for CI (ubuntu-latest has
# `make`) and for the VM. On this Windows dev machine (no `make` binary),
# use `pramaan <command>` directly — see README "Reproduce".
#
# Targets invoke the installed console script rather than
# `python -m pramaan.cli`, deliberately: `python -m` from the repo root
# puts the working directory on sys.path, which once hid a wheel that
# could not import its own modules. See docs/LIMITATIONS.md.

PRAMAAN ?= pramaan
PY ?= python
SCALE ?= dev

.PHONY: setup data data-full train certify eval report serve all full lint test

setup:
	$(PY) -m pip install -e ".[dev,docs]"
	$(PRAMAAN) setup

data:
	$(PRAMAAN) data --scale $(SCALE)

data-full:
	$(PRAMAAN) data-full

train:
	$(PRAMAAN) train --scale $(SCALE)

certify:
	$(PRAMAAN) certify --scale $(SCALE)

eval:
	$(PRAMAAN) eval --scale $(SCALE)

report:
	$(PRAMAAN) report --scale $(SCALE)

serve:
	$(PRAMAAN) serve

all:
	$(PRAMAAN) all --scale $(SCALE)

# The whole full-tier run, in order, with the scale set once.
#
# `make data-full && make train && make eval SCALE=full` was the
# documented chain and it was wrong twice over: SCALE defaults to `dev`,
# so `make train` trained a dev model that the full-scale eval then
# loaded, and `certify` — the step the entire guarantee rests on — had no
# target at all and was simply absent from the chain.
full:
	$(MAKE) data-full
	$(MAKE) train SCALE=full
	$(MAKE) certify SCALE=full
	$(MAKE) eval SCALE=full
	$(MAKE) report SCALE=full

lint:
	$(PY) -m ruff check src tests eval benchmarks scripts
	$(PY) -m mypy src eval benchmarks

test:
	$(PY) -m pytest
