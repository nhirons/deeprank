import pytest

tf = pytest.importorskip("tensorflow")

from deeprank.tf import OrdinalOutput, SortedInitializer


def test_output_shape():
    layer = OrdinalOutput(output_dim=5)
    x = tf.random.normal((4, 8))
    out = layer(x)
    assert out.shape == (4, 5)


def test_probabilities_sum_to_one():
    layer = OrdinalOutput(output_dim=3)
    x = tf.random.normal((16, 4))
    out = layer(x)
    sums = tf.reduce_sum(out, axis=-1)
    tf.debugging.assert_near(sums, tf.ones(16), atol=1e-5)


def test_probabilities_non_negative():
    layer = OrdinalOutput(output_dim=6)
    x = tf.random.normal((32, 4))
    out = layer(x)
    assert tf.reduce_all(out >= 0).numpy()


def test_thresholds_initialized_sorted():
    layer = OrdinalOutput(output_dim=5)
    layer.build((None, 4))
    t = layer.interior_thresholds.numpy().flatten()
    assert list(t) == sorted(t)


def test_gradients_flow():
    layer = OrdinalOutput(output_dim=3)
    x = tf.random.normal((8, 4))
    with tf.GradientTape() as tape:
        out = layer(x)
        loss = tf.reduce_sum(out)
    grads = tape.gradient(loss, layer.trainable_variables)
    assert all(g is not None for g in grads)


def test_sorted_initializer():
    init = SortedInitializer("glorot_uniform")
    values = init((1, 10))
    v = values.numpy().flatten()
    assert list(v) == sorted(v)


def test_get_config_roundtrip():
    layer = OrdinalOutput(output_dim=4)
    config = layer.get_config()
    assert config["output_dim"] == 4
    restored = OrdinalOutput.from_config(config)
    assert restored.output_dim == 4


def test_single_sample():
    layer = OrdinalOutput(output_dim=4)
    x = tf.random.normal((1, 2))
    out = layer(x)
    assert out.shape == (1, 4)
    tf.debugging.assert_near(tf.reduce_sum(out), 1.0, atol=1e-5)
