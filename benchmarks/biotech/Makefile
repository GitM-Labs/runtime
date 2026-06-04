NAME := biotech
include ../Makefile.common

# Tiny synthetic proteins + trivial MSAs so the loop runs without mmseqs2/OpenFold.
.PHONY: smoke-data
smoke-data:
	python fetch.py --out "$(STAGE)" --smoke
