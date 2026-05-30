# BCI Project - Brain-Computer Interface

A comprehensive toolkit for training and evaluating deep learning and machine learning models on EEG data from brain-computer interfaces.

## Overview

This project implements several state-of-the-art approaches for motor imagery EEG classification:

- **Deep ConvNet** (Schirrmeister et al., 2017) - Deep convolutional neural network
- **Shallow ConvNet** (Schirrmeister et al., 2017) - Shallow convolutional architecture
- **EEG-Conformer** - Hybrid CNN + Transformer architecture
- **Riemannian Classifiers** - MDM and TangentSpace + LR
- **ML Baselines** - LDA, SVM, Random Forest, LightGBM

## Installation

### Prerequisites
- Python 3.9+
- UV package manager (recommended) or pip

### Setup

Clone or navigate to the project directory:

```bash
cd bci-project
```

Install the package in editable mode:

```bash
# Using UV (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

This installs all dependencies and makes the `bci-*` commands available globally.

## Quick Start

### Using CLI Commands

After installation, you can run scripts as commands from anywhere:

```bash
# Convert EDF files to processed format
bci-convert

# Train Deep ConvNet model
bci-train-deep --epochs 150 --data-dir ./data

# Train Shallow ConvNet model
bci-train-shallow --epochs 150

# Train Riemannian geometry models
bci-train-riemann --data-dir ./data

# Train ML baselines (LDA, SVM, RF, LightGBM)
bci-train-ml --data-dir ./data

# Train EEG-Conformer
bci-train-conformer --epochs 150
```

### Using Python Modules

You can also import modules directly in your Python code:

```python
from src.paths import EMOTIV_RAW, DEFAULT_BCI_DIR
from src.preprocessing.convert import main as convert_data

# Run conversion
convert_data()
```

### Running from Project Root

If you prefer not to install globally, you can run from the project root:

```bash
python -m src.preprocessing.convert
python -m src.models.train_deep --epochs 100
```

## Project Structure

```
bci-project/
├── src/
│   ├── __init__.py
│   ├── paths.py                          # Path configuration
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── convert.py                    # EDF to processed format conversion
│   │   └── preprocessing.py              # Data preprocessing utilities
│   └── models/
│       ├── __init__.py
│       ├── train_deep.py                 # Deep ConvNet training
│       ├── train_shallow.py              # Shallow ConvNet training
│       ├── train_riemann.py              # Riemannian classifiers
│       ├── train_ml_baselines.py         # ML baselines
│       └── train_eeg_conformer.py        # EEG-Conformer training
├── data/
│   ├── bci_comp/                         # BCI Competition IV dataset
│   └── emotiv/                           # Emotiv EEG data
│       ├── raw/                          # Raw recordings
│       └── cleaned/                      # Processed recordings
├── models/                               # Trained model checkpoints
├── results/                              # Experimental results
│   ├── exports/                          # CSV exports
│   ├── figures/                          # Generated plots
│   └── logs/                             # Training logs
├── pyproject.toml                        # Project configuration
└── README.md                             # This file
```

## Usage Examples

### Convert EDF Files

Convert Emotiv EDF files to BCI Competition IV 2b format (C3, Cz, C4 channels):

```bash
bci-convert
```

This will:
1. Read all `.edf` files from `data/emotiv/raw/`
2. Extract FC5 and FC6 channels
3. Create proxy channels for C3, Cz, C4
4. Resample to 250 Hz
5. Save processed files to `data/emotiv/cleaned/`

### Train Deep ConvNet Model

```bash
bci-train-deep \
  --data-dir ./data \
  --epochs 150 \
  --lr 1e-3 \
  --batch-size 32 \
  --eval-mode per-subj \
  --save models/deep_best.pt
```

### Compare All Models

Train and compare all models:

```bash
bci-train-shallow --epochs 150
bci-train-deep --epochs 150
bci-train-riemann
bci-train-ml
bci-train-conformer --epochs 150
```

## Configuration

### Paths

All data paths are configured in src/paths.py:

```python
from src.paths import (
    EMOTIV_RAW,           # data/emotiv/raw
    EMOTIV_CLEANED,       # data/emotiv/cleaned
    DEFAULT_BCI_DIR,      # data/bci_comp
    DEFAULT_OUT_DIR,      # results
    DEFAULT_MODEL_SAVE_DIR,  # models
)
```

## Running from Any Directory

After installation with `uv pip install -e .`, you can run commands from anywhere:

## Dependencies

See pyproject.toml for the complete dependency list.

## Development

### Installing in Editable Mode for Development

```bash
uv pip install -e .
```

Changes to source code will be immediately reflected without reinstalling.