.PHONY: format

python_files = register tests setup.py

format:
	isort $(python_files)
	black $(python_files)
	clang-format -i src/*.cpp

lint:
	flake8 $(python_files)
	mypy $(python_files)
