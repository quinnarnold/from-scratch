import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from functional import im2col, col2im, softmax, cross_entropy_loss


def test_im2col_col2im_roundtrip():
    np.random.seed(42)
    x = np.random.randn(2, 3, 8, 8).astype(np.float32)
    kH, kW, stride, padding = 3, 3, 1, 1
    cols, out_h, out_w = im2col(x, kH, kW, stride, padding)
    assert cols.shape == (2 * out_h * out_w, 3 * 3 * 3)


def test_softmax_matches_pytorch():
    np.random.seed(42)
    x = np.random.randn(4, 10).astype(np.float32)
    ours = softmax(x)
    theirs = F.softmax(torch.tensor(x), dim=-1).numpy()
    np.testing.assert_allclose(ours, theirs, atol=1e-6)


def test_cross_entropy_matches_pytorch():
    np.random.seed(42)
    logits = np.random.randn(8, 10).astype(np.float32)
    targets = np.array([0, 3, 5, 7, 2, 1, 9, 4])

    loss_ours, dlogits = cross_entropy_loss(logits, targets)

    logits_t = torch.tensor(logits, requires_grad=True)
    targets_t = torch.tensor(targets, dtype=torch.long)
    loss_t = F.cross_entropy(logits_t, targets_t)
    loss_t.backward()

    np.testing.assert_allclose(loss_ours, loss_t.item(), atol=1e-5)
    np.testing.assert_allclose(dlogits, logits_t.grad.numpy(), atol=1e-5)


from layers import ReLU, Flatten, Linear, Conv2d, MaxPool2d, AvgPool2d, GlobalAvgPool2d, BatchNorm2d, Dropout


def test_relu_forward_backward():
    np.random.seed(42)
    x = np.random.randn(4, 10).astype(np.float32)
    dout = np.random.randn(4, 10).astype(np.float32)

    layer = ReLU()
    out = layer.forward(x)
    dx = layer.backward(dout)

    x_t = torch.tensor(x, requires_grad=True)
    out_t = F.relu(x_t)
    out_t.backward(torch.tensor(dout))

    np.testing.assert_allclose(out, out_t.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(dx, x_t.grad.numpy(), atol=1e-6)


def test_linear_forward_backward():
    np.random.seed(42)
    x = np.random.randn(4, 16).astype(np.float32)
    dout = np.random.randn(4, 8).astype(np.float32)

    layer = Linear(16, 8)

    pt_layer = nn.Linear(16, 8, bias=True)
    pt_layer.weight = nn.Parameter(torch.tensor(layer.w.T.copy(), dtype=torch.float32))
    pt_layer.bias = nn.Parameter(torch.tensor(layer.b.copy(), dtype=torch.float32))

    out = layer.forward(x)
    dx = layer.backward(dout)

    x_t = torch.tensor(x, requires_grad=True)
    out_t = pt_layer(x_t)
    out_t.backward(torch.tensor(dout))

    np.testing.assert_allclose(out, out_t.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(dx, x_t.grad.numpy(), atol=1e-5)


def test_flatten_forward_backward():
    np.random.seed(42)
    x = np.random.randn(2, 3, 4, 4).astype(np.float32)
    layer = Flatten()
    out = layer.forward(x)
    assert out.shape == (2, 48)
    dx = layer.backward(np.random.randn(2, 48).astype(np.float32))
    assert dx.shape == (2, 3, 4, 4)


def test_conv2d_forward_backward():
    np.random.seed(42)
    x = np.random.randn(2, 3, 8, 8).astype(np.float32)

    layer = Conv2d(3, 16, kernel_size=3, stride=1, padding=1)

    pt_layer = nn.Conv2d(3, 16, 3, stride=1, padding=1)
    pt_layer.weight = nn.Parameter(torch.tensor(layer.w.copy(), dtype=torch.float32))
    pt_layer.bias = nn.Parameter(torch.tensor(layer.b.copy(), dtype=torch.float32))

    out = layer.forward(x)
    x_t = torch.tensor(x, requires_grad=True)
    out_t = pt_layer(x_t)

    np.testing.assert_allclose(out, out_t.detach().numpy(), atol=1e-5)

    dout = np.random.randn(*out.shape).astype(np.float32)
    dx = layer.backward(dout)
    out_t.backward(torch.tensor(dout))

    np.testing.assert_allclose(dx, x_t.grad.numpy(), atol=1e-4)
    np.testing.assert_allclose(layer.dw, pt_layer.weight.grad.numpy(), atol=1e-4)
    np.testing.assert_allclose(layer.db, pt_layer.bias.grad.numpy(), atol=1e-4)


def test_conv2d_stride_2():
    np.random.seed(42)
    x = np.random.randn(2, 3, 8, 8).astype(np.float32)

    layer = Conv2d(3, 8, kernel_size=3, stride=2, padding=1)

    pt_layer = nn.Conv2d(3, 8, 3, stride=2, padding=1)
    pt_layer.weight = nn.Parameter(torch.tensor(layer.w.copy(), dtype=torch.float32))
    pt_layer.bias = nn.Parameter(torch.tensor(layer.b.copy(), dtype=torch.float32))

    out = layer.forward(x)
    x_t = torch.tensor(x, requires_grad=True)
    out_t = pt_layer(x_t)

    np.testing.assert_allclose(out, out_t.detach().numpy(), atol=1e-5)

    dout = np.random.randn(*out.shape).astype(np.float32)
    dx = layer.backward(dout)
    out_t.backward(torch.tensor(dout))

    np.testing.assert_allclose(dx, x_t.grad.numpy(), atol=1e-4)


def test_maxpool2d_forward_backward():
    np.random.seed(42)
    x = np.random.randn(2, 3, 8, 8).astype(np.float32)

    layer = MaxPool2d(2, stride=2)
    pt_layer = nn.MaxPool2d(2, stride=2)

    out = layer.forward(x)
    x_t = torch.tensor(x, requires_grad=True)
    out_t = pt_layer(x_t)

    np.testing.assert_allclose(out, out_t.detach().numpy(), atol=1e-6)

    dout = np.random.randn(*out.shape).astype(np.float32)
    dx = layer.backward(dout)
    out_t.backward(torch.tensor(dout))

    np.testing.assert_allclose(dx, x_t.grad.numpy(), atol=1e-6)


def test_avgpool2d_forward_backward():
    np.random.seed(42)
    x = np.random.randn(2, 3, 8, 8).astype(np.float32)

    layer = AvgPool2d(2, stride=2)
    pt_layer = nn.AvgPool2d(2, stride=2)

    out = layer.forward(x)
    x_t = torch.tensor(x, requires_grad=True)
    out_t = pt_layer(x_t)

    np.testing.assert_allclose(out, out_t.detach().numpy(), atol=1e-6)

    dout = np.random.randn(*out.shape).astype(np.float32)
    dx = layer.backward(dout)
    out_t.backward(torch.tensor(dout))

    np.testing.assert_allclose(dx, x_t.grad.numpy(), atol=1e-6)


def test_global_avg_pool_forward_backward():
    np.random.seed(42)
    x = np.random.randn(2, 16, 4, 4).astype(np.float32)

    layer = GlobalAvgPool2d()
    out = layer.forward(x)
    assert out.shape == (2, 16, 1, 1)

    expected = np.mean(x, axis=(2, 3), keepdims=True)
    np.testing.assert_allclose(out, expected, atol=1e-6)

    dout = np.random.randn(2, 16, 1, 1).astype(np.float32)
    dx = layer.backward(dout)
    assert dx.shape == x.shape

    x_t = torch.tensor(x, requires_grad=True)
    out_t = torch.mean(x_t, dim=(2, 3), keepdim=True)
    out_t.backward(torch.tensor(dout))
    np.testing.assert_allclose(dx, x_t.grad.numpy(), atol=1e-6)


def test_batchnorm2d_forward_backward():
    np.random.seed(42)
    x = np.random.randn(4, 8, 6, 6).astype(np.float32)
    dout = np.random.randn(4, 8, 6, 6).astype(np.float32)

    layer = BatchNorm2d(8)

    pt_layer = nn.BatchNorm2d(8)
    pt_layer.weight = nn.Parameter(torch.tensor(layer.gamma.copy(), dtype=torch.float32))
    pt_layer.bias = nn.Parameter(torch.tensor(layer.beta.copy(), dtype=torch.float32))
    pt_layer.running_mean = torch.tensor(layer.running_mean.copy(), dtype=torch.float32)
    pt_layer.running_var = torch.tensor(layer.running_var.copy(), dtype=torch.float32)

    out = layer.forward(x)
    x_t = torch.tensor(x, requires_grad=True)
    out_t = pt_layer(x_t)

    np.testing.assert_allclose(out, out_t.detach().numpy(), atol=1e-5)

    dx = layer.backward(dout)
    out_t.backward(torch.tensor(dout))

    np.testing.assert_allclose(dx, x_t.grad.numpy(), atol=1e-4)
    np.testing.assert_allclose(layer.dgamma, pt_layer.weight.grad.numpy(), atol=1e-4)
    np.testing.assert_allclose(layer.dbeta, pt_layer.bias.grad.numpy(), atol=1e-4)


def test_batchnorm2d_running_stats():
    np.random.seed(42)
    layer = BatchNorm2d(4, momentum=0.1)

    for _ in range(5):
        x = np.random.randn(8, 4, 3, 3).astype(np.float32)
        layer.forward(x)

    assert not np.allclose(layer.running_mean, 0.0)
    assert not np.allclose(layer.running_var, 1.0)

    layer.train_mode(False)
    x_test = np.random.randn(2, 4, 3, 3).astype(np.float32)
    out = layer.forward(x_test)
    assert out.shape == x_test.shape


def test_dropout_forward_backward():
    np.random.seed(42)
    x = np.random.randn(4, 16).astype(np.float32)
    layer = Dropout(p=0.5)

    out = layer.forward(x)
    assert out.shape == x.shape
    zero_frac = np.mean(out == 0)
    assert 0.3 < zero_frac < 0.7

    dout = np.random.randn(4, 16).astype(np.float32)
    dx = layer.backward(dout)
    assert dx.shape == x.shape

    layer.train_mode(False)
    out_eval = layer.forward(x)
    np.testing.assert_array_equal(out_eval, x)
