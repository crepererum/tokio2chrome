fmt:
	black *.py

lint:
	mypy *.py
	shellcheck *.sh

.PHONY: fmt lint
