.PHONY: format

python_files = register tests setup.py

format:
	ruff check --fix-only $(python_files)
	ruff format $(python_files)
	clang-format -i src/*.cpp src/*.hpp

lint:
	ruff check $(python_files)
	mypy $(python_files)
