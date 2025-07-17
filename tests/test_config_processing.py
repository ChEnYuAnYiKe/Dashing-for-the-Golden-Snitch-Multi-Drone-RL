import pytest
import yaml
from matplotlib.colors import ListedColormap, Colormap
from gym_drones.utils.rl_manager.config import _recursive_process_cmap


def test_recursive_process_cmap(tmp_path):
    """test _recursive_process_cmap function"""
    config = {
        "plot_a": {"cmap": "viridis"},
        "plot_b": {"drone_colors": {"cmap": ["red", "blue", "plasma"]}},
        "other_list": ["a", "b"],  # This should remain unchanged
    }

    _recursive_process_cmap(config)

    # Check single cmap string
    assert isinstance(config["plot_a"]["cmap"], Colormap)
    assert config["plot_a"]["cmap"].name == "viridis"

    # Check cmap list
    cmap_list = config["plot_b"]["drone_colors"]["cmap"]
    assert isinstance(cmap_list, list)
    assert isinstance(cmap_list[0], ListedColormap)
    assert isinstance(cmap_list[1], ListedColormap)
    assert isinstance(cmap_list[2], Colormap)
    assert cmap_list[2].name == "plasma"

    # Check unrelated list remains unchanged
    assert config["other_list"] == ["a", "b"]
