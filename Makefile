.PHONY: help
help:
	@echo ""
	@echo "make clean     - remove .pyc + __pycache__ fragments"
	@echo "make format    - format source code"
	@echo "make lint      - lint source code"
	@echo "make lint-fix  - lint source code and fix auto-fixable problems"
	@echo ""

init:
	poetry install						#  install the dependencies specified in `pyproject.toml` 
	poetry run pre-commit install

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

format:
	poetry run yapf -i -r -p .

lint:
	poetry run ruff .

lint-fix:
	poetry run ruff . --fix