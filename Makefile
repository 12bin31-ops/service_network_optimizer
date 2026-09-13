.PHONY: setup setup-full demo pipeline plan scenario dashboard test lint clean

PY ?= python

setup:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[dev]"

setup-full:
	$(PY) -m pip install -e ".[dev,agents,ml]"

demo: pipeline
	@echo ""
	@echo "완료. 대시보드 실행: make dashboard"

pipeline:
	snx pipeline --source sample --p 15

plan:
	snx plan --budget 900

scenario:
	snx scenario --loyalty 6-10=0.50 --loyalty 11+=0.25

dashboard:
	streamlit run app/dashboard.py

test:
	pytest -q

lint:
	ruff check src app tests

clean:
	rm -rf data/interim/* data/processed/* outputs *.db
	find . -name __pycache__ -type d -exec rm -rf {} +
