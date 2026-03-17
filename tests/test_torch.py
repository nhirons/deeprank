import pytest

torch = pytest.importorskip("torch")

from deeprank.torch import OrdinalOutput


def test_output_shape():
    layer = OrdinalOutput(input_dim=8, output_dim=5)
    x = torch.randn(4, 8)
    out = layer(x)
    assert out.shape == (4, 5)


def test_probabilities_sum_to_one():
    layer = OrdinalOutput(input_dim=4, output_dim=3)
    x = torch.randn(16, 4)
    out = layer(x)
    torch.testing.assert_close(out.sum(dim=-1), torch.ones(16), atol=1e-5, rtol=0)


def test_probabilities_non_negative():
    layer = OrdinalOutput(input_dim=4, output_dim=6)
    x = torch.randn(32, 4)
    out = layer(x)
    assert (out >= 0).all()


def test_thresholds_initialized_sorted():
    layer = OrdinalOutput(input_dim=4, output_dim=5)
    t = layer.interior_thresholds.detach()
    sorted_t, _ = t.sort()
    torch.testing.assert_close(t, sorted_t)


def test_gradients_flow():
    layer = OrdinalOutput(input_dim=4, output_dim=3)
    x = torch.randn(8, 4)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    assert layer.linear.weight.grad is not None
    assert layer.interior_thresholds.grad is not None


def test_seed_reproducibility():
    torch.manual_seed(42)
    a = OrdinalOutput(input_dim=4, output_dim=3)
    torch.manual_seed(42)
    b = OrdinalOutput(input_dim=4, output_dim=3)
    x = torch.randn(4, 4)
    torch.testing.assert_close(a(x), b(x))


def test_single_sample():
    layer = OrdinalOutput(input_dim=2, output_dim=4)
    x = torch.randn(1, 2)
    out = layer(x)
    assert out.shape == (1, 4)
    torch.testing.assert_close(out.sum(), torch.tensor(1.0), atol=1e-5, rtol=0)
