.PHONY: format

python_files = register tests setup.py

format:
	ruff check --select I --fix $(python_files)
	ruff format --preview $(python_files)
	clang-format -i src/*.cpp

lint:
	ruff check --preview $(python_files)
	mypy $(python_files)
