"""ENG-79: every lever in library.yaml carries a class and a restart cost."""

from __future__ import annotations

import pytest
import yaml

from gitm.kernels.library import _library_path, load_library


def _unlabelled(specs):
    return [s.name for s in specs if s.lever_class is None or s.restart_cost is None]


def test_every_lever_has_a_class_and_a_restart_cost():
    assert _unlabelled(load_library()) == []


@pytest.mark.parametrize("field", ["lever_class", "restart_cost"])
def test_a_lever_without_the_field_fails_the_check(tmp_path, field):
    raw = yaml.safe_load(_library_path().read_text())
    del raw["interventions"][0][field]
    p = tmp_path / "library.yaml"
    p.write_text(yaml.safe_dump(raw))
    assert _unlabelled(load_library(p)) == [raw["interventions"][0]["name"]]


def test_an_unknown_value_is_refused(tmp_path):
    raw = yaml.safe_load(_library_path().read_text())
    raw["interventions"][0]["lever_class"] = 4
    p = tmp_path / "library.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_library(p)
