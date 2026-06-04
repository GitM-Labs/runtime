NAME := edge
include ../Makefile.common

# Edge carries a second manifest beyond the sha256 freeze: the flat keyframe
# work-list the harness iterates. It lives with the data (in STAGE), so it is
# frozen by `make manifest`. `make manifest` runs it first.
.PHONY: keyframes smoke-data
keyframes:
	$(PY) edge-manifest --root "$(STAGE)" --out "$(STAGE)/manifest.jsonl"

manifest: keyframes

# Tiny nuScenes+KITTI-shaped fixture so the loop runs without the real datasets.
smoke-data:
	python fetch.py --out "$(STAGE)" --smoke
