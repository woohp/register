.PHONY: format

python_files = register tests setup.py

format:
	ruff check --fix-only --ignore F401 $(python_files)
	ruff format $(python_files)

fformat:
	ruff check --fix-only $(python_files)
	ruff format $(python_files)

lint:
	ruff check $(python_files)
	mypy $(python_files)
