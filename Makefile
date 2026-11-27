# Atalhos. Python 3.12 (ver .python-version). Para outro interpretador: make install PYTHON=python3.12
PYTHON ?= python3
VENV   ?= .venv
PY     := $(VENV)/bin/python

.PHONY: help install install-light reproduce test pipeline models clean

help:
	@echo "make install-light  ambiente mínimo (sem modelos) para reproduzir os números do artigo"
	@echo "make reproduce      reanálise a partir das saídas salvas + testes (~1 min)"
	@echo "make install        ambiente completo (spaCy, Stanza, UDPipe; PyTorch CPU)"
	@echo "make pipeline       roda as três ferramentas do zero e regenera outputs/ e results/"
	@echo "make clean          apaga caches"

$(PY):
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install -q --upgrade pip

install-light: $(PY)
	$(PY) -m pip install -r requirements-reanalysis.txt

install: $(PY)
	$(PY) -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements.txt
	$(PY) scripts/download_models.py

reproduce: test
	$(PY) reanalysis.py

test:
	$(PY) -m pytest -q tests/

models:
	$(PY) scripts/download_models.py

pipeline:
	$(PY) main.py

clean:
	find . -name __pycache__ -type d -prune -not -path './$(VENV)/*' -exec rm -rf {} +
	rm -rf .pytest_cache
