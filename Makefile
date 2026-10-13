.PHONY: format fformat lint

python_files = register tests

format:
	ruff check --fix-only --ignore F401 $(python_files)
	ruff format $(python_files)

fformat:
	ruff check --fix-only $(python_files)
	ruff format $(python_files)

lint:
	ruff check $(python_files)
	mypy $(python_files)
