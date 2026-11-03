# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

UV_RUN := uv run

# ---------------------------------------------------------------------
# Dev Targets
# ---------------------------------------------------------------------

.PHONY: all lint test

all: lint test

lint:
	$(UV_RUN) ruff check . --fix
	$(UV_RUN) ruff format .
	$(UV_RUN) ty check .

test:
	$(UV_RUN) pytest tests

# ---------------------------------------------------------------------
# Notebook Targets
# ---------------------------------------------------------------------

.PHONY: jupyter marimo streamlit

jupyter:
	$(UV_RUN) jupyter lab

# Usage: make marimo APP={MARIMO_SCRIPT.py}
marimo:
	$(UV_RUN) marimo edit ${APP}

# Usage: make streamlit APP={STREAMLIT_SCRIPT.py}
streamlit:
	$(UV_RUN) streamlit run ${APP}
