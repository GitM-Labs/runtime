NAME := hft
include ../Makefile.common

# Tiny local dataset (200k events/seed) so the full loop runs without a cluster.
# Full scale is the same command with --events 1_000_000_000.
.PHONY: smoke-data
smoke-data:
	python generate.py --events 200_000 --seed 42 --out "$(STAGE)/hft_smoke_seed42" --events-per-file 100_000
	python generate.py --events 200_000 --seed 43 --out "$(STAGE)/hft_smoke_seed43" --events-per-file 100_000
	python generate.py --events 200_000 --seed 44 --out "$(STAGE)/hft_smoke_seed44" --events-per-file 100_000
