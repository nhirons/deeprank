# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepOrdinal is a Python library implementing deep ordinal regression for PyTorch and TensorFlow/Keras. It provides an `OrdinalOutput` layer that converts a learned logit into ordinal class probabilities via sorted thresholds, plus loss functions from Rennie & Srebro (2005).

## Architecture

The library is packaged as `deepordinal/` (flat layout) with two backend modules:

- **`deepordinal/torch.py`** — PyTorch backend: `OrdinalOutput(nn.Module)`, `ordinal_loss`, `ordistic_loss`
- **`deepordinal/tf.py`** — TensorFlow/Keras backend: `OrdinalOutput(Layer)`, `ordinal_loss`, `ordistic_loss`

Both backends expose identical APIs. The `OrdinalOutput` layer takes an input, projects to a single logit, and produces K class probabilities using K-1 learned, sorted thresholds.

Key math: `P(y=k|x) = sigmoid(t(k+1) - logit) - sigmoid(t(k) - logit)`, with `t(0) = -inf` and `t(K) = inf` fixed.

## Build System

Uses `pyproject.toml` with setuptools. Install with `pip install -e .` for development.

## Dependencies

- **Base:** `numpy`
- **Optional extras:** `pip install ".[tf]"` for TensorFlow, `pip install ".[torch]"` for PyTorch

## Usage

```python
from deepordinal.torch import OrdinalOutput, ordinal_loss  # PyTorch
from deepordinal.tf import OrdinalOutput, ordinal_loss      # TensorFlow/Keras
```

## Examples

- `examples/example_torch.ipynb` — PyTorch with `ordinal_loss` and standard training loop
- `examples/example_tf.ipynb` — TensorFlow/Keras with `ordinal_loss` and `GradientTape` training loop
