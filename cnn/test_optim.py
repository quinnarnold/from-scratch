import numpy as np
import torch
import torch.nn as nn
import pytest
from layers import Linear, ReLU
from optim import SGD, Adam


def test_sgd_basic_convergence():
    """SGD should minimize a simple quadratic."""
    np.random.seed(42)
    layer = Linear(4, 1)
    opt = SGD(layer.parameters(), lr=0.01)

    x = np.random.randn(8, 4).astype(np.float32)
    target = np.ones((8, 1), dtype=np.float32)

    losses = []
    for _ in range(100):
        out = layer.forward(x)
        loss = np.mean((out - target) ** 2)
        losses.append(loss)
        dout = 2.0 * (out - target) / out.shape[0]
        layer.backward(dout)
        opt.step()
        opt.zero_grad()

    assert losses[-1] < losses[0] * 0.1


def test_sgd_momentum():
    """SGD with momentum should converge faster than without."""
    np.random.seed(42)
    x = np.random.randn(8, 4).astype(np.float32)
    target = np.ones((8, 1), dtype=np.float32)

    results = {}
    for momentum in [0.0, 0.9]:
        np.random.seed(42)
        layer = Linear(4, 1)
        opt = SGD(layer.parameters(), lr=0.01, momentum=momentum)
        for _ in range(50):
            out = layer.forward(x)
            dout = 2.0 * (out - target) / out.shape[0]
            layer.backward(dout)
            opt.step()
            opt.zero_grad()
        results[momentum] = np.mean((layer.forward(x) - target) ** 2)

    assert results[0.9] < results[0.0]


def test_sgd_matches_pytorch():
    """SGD parameter update should match PyTorch exactly."""
    np.random.seed(42)
    layer = Linear(4, 2)
    x = np.random.randn(3, 4).astype(np.float32)
    dout = np.random.randn(3, 2).astype(np.float32)

    pt_linear = nn.Linear(4, 2)
    pt_linear.weight = nn.Parameter(torch.tensor(layer.w.T.copy(), dtype=torch.float32))
    pt_linear.bias = nn.Parameter(torch.tensor(layer.b.copy(), dtype=torch.float32))
    pt_opt = torch.optim.SGD(pt_linear.parameters(), lr=0.01, momentum=0.9)

    opt = SGD(layer.parameters(), lr=0.01, momentum=0.9)

    for _ in range(3):
        layer.forward(x)
        layer.backward(dout)
        opt.step()
        opt.zero_grad()

        x_t = torch.tensor(x)
        out_t = pt_linear(x_t)
        out_t.backward(torch.tensor(dout))
        pt_opt.step()
        pt_opt.zero_grad()

    np.testing.assert_allclose(layer.w, pt_linear.weight.detach().numpy().T, atol=1e-5)
    np.testing.assert_allclose(layer.b, pt_linear.bias.detach().numpy(), atol=1e-5)


def test_adam_convergence():
    """Adam should converge on a simple problem."""
    np.random.seed(42)
    layer = Linear(4, 1)
    opt = Adam(layer.parameters(), lr=0.01)

    x = np.random.randn(8, 4).astype(np.float32)
    target = np.ones((8, 1), dtype=np.float32)

    losses = []
    for _ in range(100):
        out = layer.forward(x)
        loss = np.mean((out - target) ** 2)
        losses.append(loss)
        dout = 2.0 * (out - target) / out.shape[0]
        layer.backward(dout)
        opt.step()
        opt.zero_grad()

    assert losses[-1] < losses[0] * 0.1


def test_adam_matches_pytorch():
    """Adam parameter update should match PyTorch exactly."""
    np.random.seed(42)
    layer = Linear(4, 2)
    x = np.random.randn(3, 4).astype(np.float32)
    dout = np.random.randn(3, 2).astype(np.float32)

    pt_linear = nn.Linear(4, 2)
    pt_linear.weight = nn.Parameter(torch.tensor(layer.w.T.copy(), dtype=torch.float32))
    pt_linear.bias = nn.Parameter(torch.tensor(layer.b.copy(), dtype=torch.float32))
    pt_opt = torch.optim.Adam(pt_linear.parameters(), lr=0.001)

    opt = Adam(layer.parameters(), lr=0.001)

    for _ in range(3):
        layer.forward(x)
        layer.backward(dout)
        opt.step()
        opt.zero_grad()

        x_t = torch.tensor(x)
        out_t = pt_linear(x_t)
        out_t.backward(torch.tensor(dout))
        pt_opt.step()
        pt_opt.zero_grad()

    np.testing.assert_allclose(layer.w, pt_linear.weight.detach().numpy().T, atol=1e-5)
    np.testing.assert_allclose(layer.b, pt_linear.bias.detach().numpy(), atol=1e-5)


def test_zero_grad():
    """zero_grad should reset all gradients to zero."""
    np.random.seed(42)
    layer = Linear(4, 2)
    x = np.random.randn(3, 4).astype(np.float32)
    dout = np.random.randn(3, 2).astype(np.float32)

    layer.forward(x)
    layer.backward(dout)

    assert np.any(layer.dw != 0)
    assert np.any(layer.db != 0)

    opt = SGD(layer.parameters(), lr=0.01)
    opt.zero_grad()

    np.testing.assert_array_equal(layer.dw, 0)
    np.testing.assert_array_equal(layer.db, 0)
