"""Load the hyperparameters of FitKV and the baselines from configs/fitkv.yaml."""

from functools import lru_cache
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "fitkv.yaml"


@lru_cache(maxsize=None)
def load_config(path: str = str(DEFAULT_CONFIG)) -> dict:
    """Read the YAML configuration file.

    Args:
        path: path to a YAML file with the sections fitkv, numerics and baselines.

    Returns:
        dict: the parsed configuration, keyed by section name.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def get(section: str, key: str):
    """Return one hyperparameter from the default configuration.

    Args:
        section: one of "fitkv", "numerics" or "baselines".
        key: the name of the hyperparameter inside that section.

    Returns:
        The configured value (int, float or str, as written in the YAML file).
    """
    return load_config()[section][key]
