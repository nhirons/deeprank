# DeepRank

Deep ordinal regression for ranking problems, with TensorFlow/Keras and PyTorch backends.

DeepRank provides an `OrdinalOutput` layer that converts a learned logit into ordinal class probabilities via sorted thresholds, plus loss functions from [Rennie & Srebro (IJCAI 2005)](https://ttic.uchicago.edu/~nati/Publications/RennieSrebroIJCAI05.pdf) designed specifically for ordinal regression.

## Installation

```bash
pip install .
```

With a specific backend:

```bash
pip install ".[tf]"     # TensorFlow/Keras
pip install ".[torch]"  # PyTorch
```

For development:

```bash
pip install -e ".[tf,torch]"
```

## Backends

DeepRank supports two backends with identical APIs:

| | TensorFlow/Keras | PyTorch |
|---|---|---|
| Module | `deeprank.tf` | `deeprank.torch` |
| Layer | `OrdinalOutput(output_dim=K)` | `OrdinalOutput(input_dim=D, output_dim=K)` |
| Loss functions | `ordinal_loss`, `ordistic_loss` | `ordinal_loss`, `ordistic_loss` |

## OrdinalOutput Layer

The `OrdinalOutput` layer projects input features to a single logit and converts it into K class probabilities using K-1 learned, sorted thresholds:

```
P(y = k | x) = sigmoid(t(k+1) - logit) - sigmoid(t(k) - logit)
```

where `t(0) = -inf` and `t(K) = inf` are fixed, and interior thresholds are initialized sorted.

### TensorFlow/Keras

```python
from deeprank.tf import OrdinalOutput

model = Sequential([
    Dense(32, activation='relu', input_dim=20),
    Dense(32, activation='relu'),
    OrdinalOutput(output_dim=4),
])
```

### PyTorch

```python
from deeprank.torch import OrdinalOutput

model = nn.Sequential(
    nn.Linear(20, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    OrdinalOutput(input_dim=32, output_dim=4),
)
```

## Loss Functions

DeepRank implements the threshold-based ordinal loss functions from Rennie & Srebro, "Loss Functions for Preference Levels" (IJCAI 2005). These operate on raw logits and thresholds rather than probability output.

### `ordinal_loss`

```python
ordinal_loss(logits, targets, thresholds, construction='all', penalty='logistic')
```

- **logits**: `(batch,)` or `(batch, 1)` — raw predictor output
- **targets**: `(batch,)` — integer labels in `[0, K)`
- **thresholds**: `(K-1,)` — sorted interior thresholds
- **construction**: `'all'` or `'immediate'`
- **penalty**: `'hinge'`, `'smooth_hinge'`, `'modified_least_squares'`, or `'logistic'`
- **Returns**: scalar mean loss over the batch

#### Constructions

- **All-threshold** (default, eq 13): penalizes violations of every threshold, weighted by direction. Bounds mean absolute error. Best performer in the paper's experiments.
- **Immediate-threshold** (eq 12): only penalizes violations of the two thresholds bounding the correct class segment.

#### Penalty functions

| Name | Formula | Reference |
|---|---|---|
| `'hinge'` | `max(0, 1-z)` | eq 5 |
| `'smooth_hinge'` | 0 if z≥1, (1-z)²/2 if 0<z<1, 0.5-z if z≤0 | eq 6 |
| `'modified_least_squares'` | 0 if z≥1, (1-z)² if z<1 | eq 7 |
| `'logistic'` | `log(1 + exp(-z))` | eq 9 |

The paper recommends **all-threshold + logistic** as the best-performing combination.

### `ordistic_loss`

Probabilistic generalization of logistic regression to K-class ordinal problems (Section 4).

```python
ordistic_loss(logits, targets, means, log_priors=None)
```

- **logits**: `(batch,)` or `(batch, 1)` — raw predictor output
- **targets**: `(batch,)` — integer labels in `[0, K)`
- **means**: `(K,)` — class means (convention: μ₁=-1, μ_K=1; interior means learned)
- **log_priors**: `(K,)` or `None` — optional log-prior terms π_i
- **Returns**: scalar mean negative log-likelihood over the batch

### Example usage

#### PyTorch

```python
from deeprank.torch import OrdinalOutput, ordinal_loss

layer = OrdinalOutput(input_dim=32, output_dim=4)
probs = layer(x)

# Access logits and thresholds for the loss
logit = layer.linear(x)
thresholds = layer.interior_thresholds

loss = ordinal_loss(logit, targets, thresholds, construction='all', penalty='logistic')
loss.backward()
```

#### TensorFlow

```python
from deeprank.tf import OrdinalOutput, ordinal_loss

layer = OrdinalOutput(output_dim=4)
probs = layer(x)

# Access logits and thresholds for the loss
logit = tf.matmul(x, layer.kernel) + layer.bias
thresholds = tf.squeeze(layer.interior_thresholds)

loss = ordinal_loss(logit, targets, thresholds, construction='all', penalty='logistic')
```

## Running Tests

```bash
pip install -e ".[tf,torch]"
pytest -v
```

## Changelog

### 0.2.0

- Added `ordinal_loss` — Rennie & Srebro threshold-based ordinal loss with two constructions (all-threshold, immediate-threshold) and four penalty functions (hinge, smooth hinge, modified least squares, logistic)
- Added `ordistic_loss` — ordistic negative log-likelihood loss (Rennie & Srebro, Section 4)
- Both loss functions available in `deeprank.torch` and `deeprank.tf`

### 0.1.0

- Added PyTorch backend (`deeprank.torch`) with `OrdinalOutput` module
- Modernized TensorFlow backend to `tf.keras` with self-contained `OrdinalOutput` layer and `SortedInitializer`
- Dual-backend support (TensorFlow/Keras and PyTorch) with matching APIs
- `pyproject.toml` build configuration with optional `[tf]` and `[torch]` extras

### Initial

- `OrdinalOutput` Keras layer for deep ordinal regression
- Example notebook with synthetic ordinal data

## Notebooks

`notebooks/example.ipynb` contains a full working example with synthetic ordinal data using the TensorFlow backend.

