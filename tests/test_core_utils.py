import pytest
import os
from gym_drones.utils.utils import recursive_dict_update, fix_none_values, get_latest_run_id


def test_recursive_dict_update():
    """test recursive dictionary update."""
    d = {"a": 1, "b": {"c": 2, "d": 3}}
    u = {"b": {"c": 4, "e": 5}, "f": 6}
    updated_d = recursive_dict_update(d, u)
    assert updated_d == {"a": 1, "b": {"c": 4, "d": 3, "e": 5}, "f": 6}


def test_fix_none_values():
    """test fix None values in a dictionary."""
    d = {"a": "None", "b": {"c": "None", "d": "not_none"}, "e": None}
    fix_none_values(d)
    assert d["a"] is None
    assert d["b"]["c"] is None
    assert d["b"]["d"] == "not_none"
    assert d["e"] is None


def test_get_latest_run_id(tmp_path):
    """test getting the latest run ID from a directory."""
    log_path = tmp_path
    log_name = "test_run"

    # 1. test empty directory
    assert get_latest_run_id(log_path, log_name) == 0

    # 2. create some directories and test
    os.makedirs(log_path / f"{log_name}_1")
    os.makedirs(log_path / f"{log_name}_3")
    os.makedirs(log_path / f"{log_name}_2")
    os.makedirs(log_path / "other_run_1")  # should be ignored
    assert get_latest_run_id(log_path, log_name) == 3

    # 3. test directories with non-numeric suffix
    os.makedirs(log_path / f"{log_name}_final")
    assert get_latest_run_id(log_path, log_name) == 3
