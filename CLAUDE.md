# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepRank is a Keras library implementing deep ordinal regression via an `OrdinalOutput` custom layer. It converts a 1D "ordered logit" activation into ordinal class probabilities using learned, sorted thresholds.

## Architecture

The library is packaged as `deeprank/` (flat layout), with the main code in `deeprank/__init__.py`. It exports one class:

- **`OrdinalOutput(Layer)`** — A Keras custom layer that takes a 1D input (the ordered logit) and produces K class probabilities using K+1 thresholds (first/last fixed at -inf/+inf, interior ones are learned). Thresholds are initialized sorted via `sorted_initializer` to ensure proper loss behavior.

Key math: `P(y<k|Xi) = sigmoid(t(k) - logit)`, with class probabilities derived from adjacent threshold differences.

## Build System

Uses `pyproject.toml` with setuptools. Install with `pip install -e .` for development.

## Dependencies

- tensorflow>=2.0, numpy (declared in `pyproject.toml`)
- All Keras imports use `tensorflow.keras` (not standalone `keras`)

## Usage

```python
from deeprank import OrdinalOutput
model.add(OrdinalOutput(output_dim=K))  # K = number of ordered classes
```

The layer accepts any input width — it learns its own projection to a single logit internally. Works with `categorical_crossentropy` but an ordinal-specific loss (e.g., Rennie & Srebro) is preferred.

## Notebooks

`notebooks/example.ipynb` contains a full working example with synthetic ordinal data.
