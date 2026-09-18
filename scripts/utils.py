"""
utils.py

Common utility functions used throughout the multilingual LLM safety benchmark.

Author: Shriya Patil
Project: Multilingual LLM Safety Benchmark
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Environment Variables
# ---------------------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_CONFIG_PATH = "config/config.json"


def load_config(config_path: str | None = None) -> dict:
    """
    Load the experiment configuration.

    Resolution order:
        1. the config_path argument, when given
        2. the BENCHMARK_CONFIG environment variable
        3. config/config.json

    The environment variable lets a separate experiment run from its
    own config file without editing the committed default, so the
    512-token and 2048-token runs stay independent and reproducible:

        BENCHMARK_CONFIG=config/config_2048_qwen.json \
            python scripts/run_experiment.py

    Parameters
    ----------
    config_path : str | None
        Explicit path to a config JSON file. Overrides the environment.

    Returns
    -------
    dict
        Configuration dictionary.
    """

    if config_path is None:
        config_path = os.environ.get(
            "BENCHMARK_CONFIG",
            DEFAULT_CONFIG_PATH,
        )

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

def load_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Load the benchmark dataset.

    Parameters
    ----------
    dataset_path : str

    Returns
    -------
    pandas.DataFrame
    """

    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    return pd.read_csv(path, encoding="utf-8")


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def ensure_directory(directory: str) -> Path:
    """
    Create a directory if it doesn't exist.

    Returns
    -------
    pathlib.Path
    """

    path = Path(directory)

    path.mkdir(parents=True, exist_ok=True)

    return path


# ---------------------------------------------------------------------
# Timestamp
# ---------------------------------------------------------------------

def get_timestamp() -> str:
    """
    Return current timestamp.

    Example
    -------
    2026-07-10_21-45-32
    """

    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def setup_logger(
    log_directory: str = "logs",
    logger_name: str = "experiment"
) -> logging.Logger:
    """
    Create experiment logger.

    Returns
    -------
    logging.Logger
    """

    ensure_directory(log_directory)

    log_file = (
        Path(log_directory)
        / f"{logger_name}_{get_timestamp()}.log"
    )

    logger = logging.getLogger(logger_name)

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8"
    )

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------

def save_dataframe(
    df: pd.DataFrame,
    output_path: str
) -> None:
    """
    Save DataFrame to CSV.
    """

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output,
        index=False,
        encoding="utf-8-sig"
    )

# ---------------------------------------------------------------------
# Environment Variables
# ---------------------------------------------------------------------

def get_environment_variable(variable_name: str) -> str:
    """
    Retrieve an environment variable.

    Parameters
    ----------
    variable_name : str

    Returns
    -------
    str
    """

    value = os.getenv(variable_name)

    if not value:
        raise ValueError(
            f"Missing environment variable: {variable_name}"
        )

    return value