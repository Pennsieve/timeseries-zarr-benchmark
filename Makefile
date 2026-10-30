.PHONY: venv test lint typecheck check smoke lock lockcheck

VENV_DIR ?= .venv
PYTHON   ?= python3.12
SOURCES  := tszbench generate upload format_benchmark resampling_benchmark plot smoke envs tests

venv:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r envs/requirements.lock
	$(VENV_DIR)/bin/pip install --no-deps -e ".[test,dev]"

test:
	$(VENV_DIR)/bin/python -m pytest -q

lint:
	$(VENV_DIR)/bin/ruff check --fix $(SOURCES)
	$(VENV_DIR)/bin/ruff format $(SOURCES)

typecheck:
	$(VENV_DIR)/bin/mypy

check:
	$(VENV_DIR)/bin/ruff check $(SOURCES)
	$(VENV_DIR)/bin/ruff format --check $(SOURCES)
	$(VENV_DIR)/bin/mypy
	$(VENV_DIR)/bin/python -m pytest -q

# Outputs go to SMOKE_DIR, or to a new temporary directory when it is empty.
SMOKE_DIR ?=

smoke:
	$(VENV_DIR)/bin/python -m smoke $(SMOKE_DIR)

# Resolves for Linux x86_64 and Python 3.12 on any host. Needs network access.
lock:
	$(VENV_DIR)/bin/python envs/lock.py

# Installs the lock into a new venv in a temporary directory and imports every
# module. Run it on Linux x86_64 before you tag the commit for repo_commit.
lockcheck:
	scratch=$$(mktemp -d) && trap 'rm -rf "$$scratch"' EXIT && \
	$(PYTHON) -m venv "$$scratch/venv" && \
	"$$scratch/venv/bin/pip" install --upgrade pip && \
	"$$scratch/venv/bin/pip" install -r envs/requirements.lock && \
	"$$scratch/venv/bin/python" envs/import_all.py
