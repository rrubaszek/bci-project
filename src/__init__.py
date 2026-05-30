"""
BCI Project - Brain-Computer Interface using EEG signals

A comprehensive toolkit for training and evaluating deep learning and machine learning
models on EEG data from brain-computer interfaces.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Import key modules for convenient access
from src.paths import (
    PROJECT_ROOT,
    SRC_DIR,
    DEFAULT_DATA_DIR,
    EMOTIV_RAW,
    EMOTIV_CLEANED,
    DEFAULT_BCI_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_MODEL_SAVE_DIR,
)

__all__ = [
    "PROJECT_ROOT",
    "SRC_DIR",
    "DEFAULT_DATA_DIR",
    "EMOTIV_RAW",
    "EMOTIV_CLEANED",
    "DEFAULT_BCI_DIR",
    "DEFAULT_OUT_DIR",
    "DEFAULT_MODEL_SAVE_DIR",
]
