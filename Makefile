.PHONY: core check clean demo

PYTHON ?= python3
FC ?= gfortran

core:
	$(FC) -shared -fPIC -o core/processing.so core/lib_pro.f90

check:
	$(PYTHON) -m compileall -q src scripts examples
	find src scripts examples -type d -name "__pycache__" -prune -exec rm -rf {} +

demo:
	$(PYTHON) scripts/test_matrix.py demo --trace-index 3 --save-figures

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
