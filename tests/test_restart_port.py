"""Restart candidate gets a fresh distributed port.

The restart A/B keeps the baseline engine alive while building the candidate, so
the candidate is a second in-process engine. _free_port gives it a distinct
distributed init port to avoid a tcp://…:PORT collision on V1. (The full
two-engine behaviour is GPU-validated; here we just pin the port helper.)
"""

from __future__ import annotations

import socket

from gitm.workloads import _free_port


def test_free_port_is_a_valid_bindable_port():
    p = _free_port()
    assert isinstance(p, int)
    assert 1 <= p <= 65535
    # it was OS-assigned free, so it should be bindable right now.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("", p))
    finally:
        s.close()
