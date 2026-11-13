.PHONY: install test lint

VENV ?= .venv

ifeq ($(origin VIRTUAL_ENV), undefined)
    PYTHON := $(VENV)/bin/python
    PIP := $(VENV)/bin/pip
    PYTEST := $(VENV)/bin/pytest
    RUFF := $(VENV)/bin/ruff
else
    PYTHON := python3
    PIP := pip
    PYTEST := pytest
    RUFF := ruff
endif

install:
	@if [ -z "$$VIRTUAL_ENV" ] && [ ! -d "$(VENV)" ]; then \
		if command -v python3.11 >/dev/null 2>&1; then \
			python3.11 -m venv $(VENV); \
		else \
			python3 -m venv $(VENV); \
		fi; \
	fi
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

test:
	$(PYTEST)

lint:
	$(RUFF) check .
