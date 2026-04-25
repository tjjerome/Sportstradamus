"""JSON configuration I/O for distribution and zero-inflation settings."""

import importlib.resources as pkg_resources
import json
import os

from sportstradamus import data


def load_distribution_config() -> dict:
    """Load distribution configuration from stat_dist.json."""
    filepath = pkg_resources.files(data) / "stat_dist.json"
    if os.path.isfile(filepath):
        with open(filepath) as f:
            return json.load(f)
    return {}


def save_distribution_config(config: dict) -> None:
    """Save distribution configuration to stat_dist.json."""
    filepath = pkg_resources.files(data) / "stat_dist.json"
    with open(filepath, "w") as f:
        json.dump(config, f, indent=4)


def load_zi_config() -> dict:
    """Load zero-inflation gate configuration from stat_zi.json."""
    filepath = pkg_resources.files(data) / "stat_zi.json"
    if os.path.isfile(filepath):
        with open(filepath) as f:
            return json.load(f)
    return {}


def save_zi_config(config: dict) -> None:
    """Save zero-inflation gate configuration to stat_zi.json."""
    filepath = pkg_resources.files(data) / "stat_zi.json"
    with open(filepath, "w") as f:
        json.dump(config, f, indent=4)
