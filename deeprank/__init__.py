import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer

__version__ = "0.1.0"
__all__ = ["OrdinalOutput"]

_INF = tf.constant(float("inf"))


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
