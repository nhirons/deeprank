import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer

__all__ = ["OrdinalOutput", "SortedInitializer", "ordinal_loss", "ordistic_loss"]

_INF = tf.constant(float("inf"))


def _penalty(z, name):
    if name == "hinge":
        return tf.maximum(0.0, 1.0 - z)
    elif name == "smooth_hinge":
        return tf.where(
            z >= 1.0,
            tf.zeros_like(z),
            tf.where(z > 0.0, (1.0 - z) ** 2 / 2.0, 0.5 - z),
        )
    elif name == "modified_least_squares":
        return tf.where(z >= 1.0, tf.zeros_like(z), (1.0 - z) ** 2)
    elif name == "logistic":
        return tf.math.softplus(-z)
    else:
        raise ValueError(f"Unknown penalty: {name}")


def ordinal_loss(logits, targets, thresholds, construction="all", penalty="logistic"):
    """Rennie & Srebro ordinal loss (IJCAI 2005).

    Args:
        logits: (batch,) or (batch, 1) — raw predictor output z(x).
        targets: (batch,) — integer labels in [0, K).
        thresholds: (K-1,) — sorted interior thresholds.
        construction: ``'all'`` or ``'immediate'``.
        penalty: ``'hinge'``, ``'smooth_hinge'``, ``'modified_least_squares'``,
            or ``'logistic'``.

    Returns:
        Scalar mean loss over the batch.
    """
    logits = tf.cast(tf.reshape(logits, [-1]), tf.float32)
    targets = tf.cast(targets, tf.int32)
    thresholds = tf.cast(thresholds, tf.float32)
    K = thresholds.shape[0] + 1
    y = targets + 1  # 1-indexed

    if construction == "all":
        l_idx = tf.cast(tf.range(1, K), tf.float32)  # (K-1,)
        y_f = tf.cast(y, tf.float32)
        signs = tf.where(
            tf.expand_dims(l_idx, 0) < tf.expand_dims(y_f, 1), -1.0, 1.0
        )  # (batch, K-1)
        diff = tf.expand_dims(thresholds, 0) - tf.expand_dims(logits, 1)  # (batch, K-1)
        loss = tf.reduce_sum(_penalty(signs * diff, penalty), axis=1)
    elif construction == "immediate":
        t_low = tf.concat([[float("-inf")], thresholds], axis=0)
        t_high = tf.concat([thresholds, [float("inf")]], axis=0)
        theta_low = tf.gather(t_low, targets)
        theta_high = tf.gather(t_high, targets)
        loss = _penalty(logits - theta_low, penalty) + _penalty(theta_high - logits, penalty)
    else:
        raise ValueError(f"Unknown construction: {construction}")

    return tf.reduce_mean(loss)


def ordistic_loss(logits, targets, means, log_priors=None):
    """Ordistic loss (Rennie & Srebro, Section 4).

    Args:
        logits: (batch,) or (batch, 1) — raw predictor output z(x).
        targets: (batch,) — integer labels in [0, K).
        means: (K,) — class means.
        log_priors: (K,) or None — log-prior terms. Defaults to zeros.

    Returns:
        Scalar mean negative log-likelihood over the batch.
    """
    logits = tf.cast(tf.reshape(logits, [-1]), tf.float32)
    targets = tf.cast(targets, tf.int32)
    means = tf.cast(means, tf.float32)
    K = means.shape[0]
    if log_priors is None:
        log_priors = tf.zeros([K], dtype=tf.float32)
    else:
        log_priors = tf.cast(log_priors, tf.float32)
    energy = (
        tf.expand_dims(means, 0) * tf.expand_dims(logits, 1)
        + tf.expand_dims(log_priors, 0)
        - tf.expand_dims(means, 0) ** 2 / 2.0
    )
    batch_idx = tf.range(tf.shape(targets)[0])
    indices = tf.stack([batch_idx, targets], axis=1)
    target_energy = tf.gather_nd(energy, indices)
    log_partition = tf.reduce_logsumexp(energy, axis=1)
    return tf.reduce_mean(log_partition - target_energy)


class SortedInitializer(initializers.Initializer):
    """Wraps a Keras initializer and returns its output sorted along the last axis."""

    def __init__(self, base="glorot_uniform"):
        self.base = initializers.get(base)

    def __call__(self, shape, dtype=None):
        values = self.base(shape, dtype=dtype)
        return tf.sort(values, axis=-1)

    def get_config(self):
        return {"base": initializers.serialize(self.base)}


class OrdinalOutput(Layer):
    """Ordinal regression output layer.

    Projects an arbitrary input down to a single logit and converts it
    into *output_dim* class probabilities via learned, sorted thresholds.

    The layer learns ``output_dim - 1`` interior thresholds ``t(1)…t(K-1)``
    (with ``t(0) = -∞`` and ``t(K) = +∞`` fixed) and computes::

        P(y = k | x) = σ(t(k+1) - logit) - σ(t(k) - logit)
    """

    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )
        self.interior_thresholds = self.add_weight(
            name="thresholds",
            shape=(1, self.output_dim - 1),
            initializer=SortedInitializer("glorot_uniform"),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        logit = tf.matmul(inputs, self.kernel) + self.bias
        t_low = tf.fill([1, 1], -_INF)
        t_high = tf.fill([1, 1], _INF)
        thresholds = tf.concat([t_low, self.interior_thresholds, t_high], axis=-1)
        return tf.sigmoid(thresholds[:, 1:] - logit) - tf.sigmoid(thresholds[:, :-1] - logit)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = super().get_config()
        config["output_dim"] = self.output_dim
        return config
