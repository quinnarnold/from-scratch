# CNN Framework Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a from-scratch CNN framework in NumPy with manual backpropagation, capable of training image classifiers on CIFAR-10.

**Architecture:** Layer-centric design where each layer owns its forward/backward logic. Models are containers that chain layers. im2col trick for efficient convolution. Optional PyTorch backend for GPU speed.

**Tech Stack:** Python, NumPy, PyTorch (validation + optional backend), pytest

---

### Task 1: Project Scaffold + Functional Foundations

**Files:**
- Create: `functional.py`
- Create: `test_layers.py`

**Step 1: Create `functional.py` with im2col, col2im, softmax, cross_entropy_loss**

```python
import numpy as np


def im2col(x, kH, kW, stride=1, padding=0):
    """Reshape input patches into columns for efficient convolution.

    Args:
        x: input array of shape (N, C, H, W)
        kH, kW: kernel height and width
        stride: stride of the convolution
        padding: zero-padding added to both sides

    Returns:
        cols: array of shape (N * out_h * out_w, C * kH * kW)
        out_h, out_w: output spatial dimensions
    """
    N, C, H, W = x.shape
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    H_pad, W_pad = x.shape[2], x.shape[3]
    out_h = (H_pad - kH) // stride + 1
    out_w = (W_pad - kW) // stride + 1

    cols = np.zeros((N, C, kH, kW, out_h, out_w))
    for i in range(kH):
        i_max = i + stride * out_h
        for j in range(kW):
            j_max = j + stride * out_w
            cols[:, :, i, j, :, :] = x[:, :, i:i_max:stride, j:j_max:stride]

    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return cols, out_h, out_w


def col2im(dcols, x_shape, kH, kW, stride=1, padding=0):
    """Inverse of im2col -- scatter column gradients back to input shape.

    Args:
        dcols: gradient array of shape (N * out_h * out_w, C * kH * kW)
        x_shape: original input shape (N, C, H, W)
        kH, kW: kernel height and width
        stride: stride of the convolution
        padding: zero-padding that was applied

    Returns:
        dx: gradient w.r.t. input of shape (N, C, H, W)
    """
    N, C, H, W = x_shape
    H_pad = H + 2 * padding
    W_pad = W + 2 * padding
    out_h = (H_pad - kH) // stride + 1
    out_w = (W_pad - kW) // stride + 1

    dcols_reshaped = dcols.reshape(N, out_h, out_w, C, kH, kW).transpose(0, 3, 4, 5, 1, 2)
    dx_padded = np.zeros((N, C, H_pad, W_pad))

    for i in range(kH):
        i_max = i + stride * out_h
        for j in range(kW):
            j_max = j + stride * out_w
            dx_padded[:, :, i:i_max:stride, j:j_max:stride] += dcols_reshaped[:, :, i, j, :, :]

    if padding > 0:
        return dx_padded[:, :, padding:-padding, padding:-padding]
    return dx_padded


def softmax(x):
    """Numerically stable softmax over last axis."""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def cross_entropy_loss(logits, targets):
    """Cross-entropy loss with built-in softmax.

    Args:
        logits: raw scores of shape (N, C)
        targets: integer class labels of shape (N,)

    Returns:
        loss: scalar average loss
        dlogits: gradient w.r.t. logits of shape (N, C)
    """
    N = logits.shape[0]
    probs = softmax(logits)
    log_probs = -np.log(probs[np.arange(N), targets] + 1e-12)
    loss = np.mean(log_probs)

    dlogits = probs.copy()
    dlogits[np.arange(N), targets] -= 1.0
    dlogits /= N
    return loss, dlogits


def kaiming_init(shape, fan_in):
    """Kaiming He initialization for ReLU networks."""
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(*shape).astype(np.float32) * std


def xavier_init(shape, fan_in, fan_out):
    """Xavier/Glorot initialization."""
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(*shape).astype(np.float32) * std
```

**Step 2: Write validation tests for functional.py**

```python
import numpy as np
import torch
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
```

**Step 3: Run tests to verify they pass**

Run: `cd /Users/quinnarnold/Desktop/from-scratch/cnn && python -m pytest test_layers.py -v`
Expected: 3 tests PASS

**Step 4: Commit**

```bash
git add functional.py test_layers.py
git commit -m "feat: add functional module with im2col, cross_entropy, softmax"
```

---

### Task 2: Base Classes + ReLU + Flatten + Linear

**Files:**
- Create: `layers.py`
- Modify: `test_layers.py`

**Step 1: Write base Layer and Model classes, plus ReLU, Flatten, Linear**

```python
import numpy as np
from functional import im2col, col2im, kaiming_init


class Layer:
    """Base class for all layers."""

    def __init__(self):
        self.training = True
        self.cache = {}

    def forward(self, x):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError

    def parameters(self):
        return []

    def train_mode(self, training=True):
        self.training = training


class Model:
    """Base class for models. Subclass and override forward() for custom architectures."""

    def __init__(self):
        self._layers = []
        self._layer_order = []

    def _register_layers(self):
        self._layers = []
        for name in dir(self):
            attr = getattr(self, name)
            if isinstance(attr, Layer):
                self._layers.append(attr)
            elif isinstance(attr, Model) and attr is not self:
                self._layers.extend(attr._get_all_layers())

    def _get_all_layers(self):
        self._register_layers()
        result = []
        for item in self._layers:
            if isinstance(item, Layer):
                result.append(item)
        return result

    def forward(self, x):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError

    def parameters(self):
        self._register_layers()
        params = []
        seen = set()
        for layer in self._layers:
            for pair in layer.parameters():
                pid = id(pair[0])
                if pid not in seen:
                    seen.add(pid)
                    params.append(pair)
        return params

    def train(self):
        self._register_layers()
        for layer in self._layers:
            layer.train_mode(True)

    def eval(self):
        self._register_layers()
        for layer in self._layers:
            layer.train_mode(False)


class ReLU(Layer):

    def forward(self, x):
        self.cache['mask'] = x > 0
        return x * self.cache['mask']

    def backward(self, dout):
        return dout * self.cache['mask']


class Flatten(Layer):

    def forward(self, x):
        self.cache['shape'] = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.cache['shape'])


class Linear(Layer):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.w = kaiming_init((in_features, out_features), in_features)
        self.b = np.zeros(out_features, dtype=np.float32)
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.cache['x'] = x
        return x @ self.w + self.b

    def backward(self, dout):
        x = self.cache['x']
        self.dw = x.T @ dout
        self.db = np.sum(dout, axis=0)
        return dout @ self.w.T

    def parameters(self):
        return [(self.w, self.dw), (self.b, self.db)]
```

**Step 2: Add tests for ReLU, Flatten, Linear to test_layers.py**

Append to `test_layers.py`:

```python
from layers import ReLU, Flatten, Linear
import torch.nn as nn


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
    pt_layer.weight = nn.Parameter(torch.tensor(layer.w.T.copy()))
    pt_layer.bias = nn.Parameter(torch.tensor(layer.b.copy()))

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
```

**Step 3: Run tests**

Run: `cd /Users/quinnarnold/Desktop/from-scratch/cnn && python -m pytest test_layers.py -v`
Expected: 6 tests PASS

**Step 4: Commit**

```bash
git add layers.py test_layers.py
git commit -m "feat: add base classes, ReLU, Flatten, Linear with tests"
```

---

### Task 3: Conv2d Layer

**Files:**
- Modify: `layers.py`
- Modify: `test_layers.py`

**Step 1: Add Conv2d to layers.py**

Add to `layers.py`:

```python
class Conv2d(Layer):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kH, self.kW = kernel_size
        self.stride = stride
        self.padding = padding

        fan_in = in_channels * self.kH * self.kW
        self.w = kaiming_init((out_channels, in_channels, self.kH, self.kW), fan_in)
        self.b = np.zeros(out_channels, dtype=np.float32)
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        N, C, H, W = x.shape
        cols, out_h, out_w = im2col(x, self.kH, self.kW, self.stride, self.padding)
        self.cache['x_shape'] = x.shape
        self.cache['cols'] = cols
        self.cache['out_h'] = out_h
        self.cache['out_w'] = out_w

        w_flat = self.w.reshape(self.out_channels, -1)
        out = cols @ w_flat.T + self.b
        out = out.reshape(N, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        N = dout.shape[0]
        out_h, out_w = self.cache['out_h'], self.cache['out_w']
        cols = self.cache['cols']

        dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        w_flat = self.w.reshape(self.out_channels, -1)

        self.dw = (dout_flat.T @ cols).reshape(self.w.shape)
        self.db = np.sum(dout_flat, axis=0)

        dcols = dout_flat @ w_flat
        dx = col2im(dcols, self.cache['x_shape'], self.kH, self.kW,
                     self.stride, self.padding)
        return dx

    def parameters(self):
        return [(self.w, self.dw), (self.b, self.db)]
```

**Step 2: Add Conv2d tests**

Append to `test_layers.py`:

```python
from layers import Conv2d


def test_conv2d_forward_backward():
    np.random.seed(42)
    x = np.random.randn(2, 3, 8, 8).astype(np.float32)

    layer = Conv2d(3, 16, kernel_size=3, stride=1, padding=1)

    pt_layer = nn.Conv2d(3, 16, 3, stride=1, padding=1)
    pt_layer.weight = nn.Parameter(torch.tensor(layer.w.copy()))
    pt_layer.bias = nn.Parameter(torch.tensor(layer.b.copy()))

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
    pt_layer.weight = nn.Parameter(torch.tensor(layer.w.copy()))
    pt_layer.bias = nn.Parameter(torch.tensor(layer.b.copy()))

    out = layer.forward(x)
    x_t = torch.tensor(x, requires_grad=True)
    out_t = pt_layer(x_t)

    np.testing.assert_allclose(out, out_t.detach().numpy(), atol=1e-5)

    dout = np.random.randn(*out.shape).astype(np.float32)
    dx = layer.backward(dout)
    out_t.backward(torch.tensor(dout))

    np.testing.assert_allclose(dx, x_t.grad.numpy(), atol=1e-4)
```

**Step 3: Run tests**

Run: `cd /Users/quinnarnold/Desktop/from-scratch/cnn && python -m pytest test_layers.py::test_conv2d_forward_backward test_layers.py::test_conv2d_stride_2 -v`
Expected: 2 tests PASS

**Step 4: Commit**

```bash
git add layers.py test_layers.py
git commit -m "feat: add Conv2d layer with im2col forward/backward"
```

---

### Task 4: Pooling Layers

**Files:**
- Modify: `layers.py`
- Modify: `test_layers.py`

**Step 1: Add MaxPool2d, AvgPool2d, GlobalAvgPool2d to layers.py**

Add to `layers.py`:

```python
class MaxPool2d(Layer):

    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.k = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = (H - self.k) // self.stride + 1
        out_w = (W - self.k) // self.stride + 1

        x_reshaped = np.zeros((N, C, out_h, out_w, self.k, self.k))
        for i in range(self.k):
            for j in range(self.k):
                x_reshaped[:, :, :, :, i, j] = x[:, :,
                    i:i + self.stride * out_h:self.stride,
                    j:j + self.stride * out_w:self.stride]

        out = x_reshaped.max(axis=(4, 5))
        self.cache['x'] = x
        self.cache['x_reshaped'] = x_reshaped
        self.cache['out'] = out
        return out

    def backward(self, dout):
        x = self.cache['x']
        x_reshaped = self.cache['x_reshaped']
        out = self.cache['out']
        N, C, H, W = x.shape
        out_h, out_w = out.shape[2], out.shape[3]

        mask = (x_reshaped == out[:, :, :, :, None, None])
        dx = np.zeros_like(x)
        dout_expanded = dout[:, :, :, :, None, None] * mask

        for i in range(self.k):
            for j in range(self.k):
                dx[:, :,
                   i:i + self.stride * out_h:self.stride,
                   j:j + self.stride * out_w:self.stride] += dout_expanded[:, :, :, :, i, j]
        return dx


class AvgPool2d(Layer):

    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.k = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = (H - self.k) // self.stride + 1
        out_w = (W - self.k) // self.stride + 1
        self.cache['x_shape'] = x.shape
        self.cache['out_h'] = out_h
        self.cache['out_w'] = out_w

        out = np.zeros((N, C, out_h, out_w))
        for i in range(self.k):
            for j in range(self.k):
                out += x[:, :,
                         i:i + self.stride * out_h:self.stride,
                         j:j + self.stride * out_w:self.stride]
        return out / (self.k * self.k)

    def backward(self, dout):
        x_shape = self.cache['x_shape']
        out_h, out_w = self.cache['out_h'], self.cache['out_w']
        dx = np.zeros(x_shape)
        scaled = dout / (self.k * self.k)
        for i in range(self.k):
            for j in range(self.k):
                dx[:, :,
                   i:i + self.stride * out_h:self.stride,
                   j:j + self.stride * out_w:self.stride] += scaled
        return dx


class GlobalAvgPool2d(Layer):

    def forward(self, x):
        self.cache['shape'] = x.shape
        return np.mean(x, axis=(2, 3), keepdims=True)

    def backward(self, dout):
        N, C, H, W = self.cache['shape']
        return np.broadcast_to(dout / (H * W), (N, C, H, W)).copy()
```

**Step 2: Add pooling tests**

Append to `test_layers.py`:

```python
from layers import MaxPool2d, AvgPool2d, GlobalAvgPool2d


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
```

**Step 3: Run tests**

Run: `cd /Users/quinnarnold/Desktop/from-scratch/cnn && python -m pytest test_layers.py::test_maxpool2d_forward_backward test_layers.py::test_avgpool2d_forward_backward test_layers.py::test_global_avg_pool_forward_backward -v`
Expected: 3 tests PASS

**Step 4: Commit**

```bash
git add layers.py test_layers.py
git commit -m "feat: add MaxPool2d, AvgPool2d, GlobalAvgPool2d with tests"
```

---

### Task 5: BatchNorm + Dropout

**Files:**
- Modify: `layers.py`
- Modify: `test_layers.py`

**Step 1: Add BatchNorm and Dropout to layers.py**

Add to `layers.py`:

```python
class BatchNorm(Layer):

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features, dtype=np.float32)
        self.beta = np.zeros(num_features, dtype=np.float32)
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

    def forward(self, x):
        if x.ndim == 4:
            N, C, H, W = x.shape
            x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)
        else:
            x_flat = x

        if self.training:
            mean = np.mean(x_flat, axis=0)
            var = np.var(x_flat, axis=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x_flat - mean) / np.sqrt(var + self.eps)
        out_flat = self.gamma * x_norm + self.beta

        self.cache['x_flat'] = x_flat
        self.cache['x_norm'] = x_norm
        self.cache['mean'] = mean
        self.cache['var'] = var
        self.cache['x_shape'] = x.shape

        if x.ndim == 4:
            return out_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return out_flat

    def backward(self, dout):
        x_shape = self.cache['x_shape']
        if dout.ndim == 4:
            N, C, H, W = dout.shape
            dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, C)
        else:
            dout_flat = dout

        x_flat = self.cache['x_flat']
        x_norm = self.cache['x_norm']
        mean = self.cache['mean']
        var = self.cache['var']
        M = x_flat.shape[0]
        std_inv = 1.0 / np.sqrt(var + self.eps)

        self.dgamma = np.sum(dout_flat * x_norm, axis=0)
        self.dbeta = np.sum(dout_flat, axis=0)

        dx_norm = dout_flat * self.gamma
        dx_flat = (1.0 / M) * std_inv * (
            M * dx_norm - np.sum(dx_norm, axis=0)
            - x_norm * np.sum(dx_norm * x_norm, axis=0)
        )

        if x_shape != x_flat.shape and len(x_shape) == 4:
            N, C, H, W = x_shape
            return dx_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return dx_flat

    def parameters(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]


class Dropout(Layer):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            self.cache['mask'] = (np.random.rand(*x.shape) > self.p).astype(np.float32)
            return x * self.cache['mask'] / (1.0 - self.p)
        return x

    def backward(self, dout):
        if self.training:
            return dout * self.cache['mask'] / (1.0 - self.p)
        return dout
```

**Step 2: Add BatchNorm and Dropout tests**

Append to `test_layers.py`:

```python
from layers import BatchNorm, Dropout


def test_batchnorm_forward_backward():
    np.random.seed(42)
    x = np.random.randn(8, 16, 4, 4).astype(np.float32)

    layer = BatchNorm(16)
    pt_layer = nn.BatchNorm2d(16)
    pt_layer.weight.data = torch.tensor(layer.gamma.copy())
    pt_layer.bias.data = torch.tensor(layer.beta.copy())
    pt_layer.train()

    out = layer.forward(x)
    x_t = torch.tensor(x, requires_grad=True)
    out_t = pt_layer(x_t)

    np.testing.assert_allclose(out, out_t.detach().numpy(), atol=1e-5)

    dout = np.random.randn(*out.shape).astype(np.float32)
    dx = layer.backward(dout)
    out_t.backward(torch.tensor(dout))

    np.testing.assert_allclose(dx, x_t.grad.numpy(), atol=1e-4)


def test_batchnorm_eval_mode():
    np.random.seed(42)
    layer = BatchNorm(8)
    x = np.random.randn(4, 8).astype(np.float32)

    for _ in range(10):
        layer.forward(np.random.randn(16, 8).astype(np.float32))

    layer.train_mode(False)
    out = layer.forward(x)
    out2 = layer.forward(x)
    np.testing.assert_allclose(out, out2, atol=1e-6)


def test_dropout_train_vs_eval():
    np.random.seed(42)
    x = np.random.randn(4, 10).astype(np.float32)
    layer = Dropout(0.5)

    layer.train_mode(True)
    out_train = layer.forward(x)
    assert np.sum(out_train == 0) > 0

    layer.train_mode(False)
    out_eval = layer.forward(x)
    np.testing.assert_allclose(out_eval, x, atol=1e-6)
```

**Step 3: Run tests**

Run: `cd /Users/quinnarnold/Desktop/from-scratch/cnn && python -m pytest test_layers.py::test_batchnorm_forward_backward test_layers.py::test_batchnorm_eval_mode test_layers.py::test_dropout_train_vs_eval -v`
Expected: 3 tests PASS

**Step 4: Commit**

```bash
git add layers.py test_layers.py
git commit -m "feat: add BatchNorm and Dropout with tests"
```

---

### Task 6: Optimizers

**Files:**
- Create: `optim.py`
- Modify: `test_layers.py`

**Step 1: Write SGD and Adam optimizers**

```python
import numpy as np


class SGD:

    def __init__(self, parameters, lr=0.01, momentum=0.9, weight_decay=0.0):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [np.zeros_like(p) for p, _ in parameters]

    def step(self):
        for i, (param, grad) in enumerate(self.parameters):
            g = grad
            if self.weight_decay > 0:
                g = g + self.weight_decay * param
            self.velocities[i] = self.momentum * self.velocities[i] + g
            param -= self.lr * self.velocities[i]

    def zero_grad(self):
        for _, grad in self.parameters:
            grad[:] = 0


class Adam:

    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p) for p, _ in parameters]
        self.v = [np.zeros_like(p) for p, _ in parameters]
        self.t = 0

    def step(self):
        self.t += 1
        for i, (param, grad) in enumerate(self.parameters):
            g = grad
            if self.weight_decay > 0:
                g = g + self.weight_decay * param
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for _, grad in self.parameters:
            grad[:] = 0
```

**Step 2: Add optimizer tests**

Append to `test_layers.py`:

```python
from optim import SGD, Adam


def test_sgd_updates_params():
    np.random.seed(42)
    layer = Linear(4, 2)
    x = np.random.randn(3, 4).astype(np.float32)
    target = np.random.randn(3, 2).astype(np.float32)

    opt = SGD(layer.parameters(), lr=0.01, momentum=0.0)
    w_before = layer.w.copy()

    out = layer.forward(x)
    dout = 2 * (out - target) / out.shape[0]
    layer.backward(dout)
    opt.step()
    opt.zero_grad()

    assert not np.allclose(layer.w, w_before)


def test_adam_updates_params():
    np.random.seed(42)
    layer = Linear(4, 2)
    x = np.random.randn(3, 4).astype(np.float32)
    target = np.random.randn(3, 2).astype(np.float32)

    opt = Adam(layer.parameters(), lr=0.01)
    w_before = layer.w.copy()

    out = layer.forward(x)
    dout = 2 * (out - target) / out.shape[0]
    layer.backward(dout)
    opt.step()
    opt.zero_grad()

    assert not np.allclose(layer.w, w_before)
    assert np.allclose(layer.dw, 0)
```

**Step 3: Run tests**

Run: `cd /Users/quinnarnold/Desktop/from-scratch/cnn && python -m pytest test_layers.py::test_sgd_updates_params test_layers.py::test_adam_updates_params -v`
Expected: 2 tests PASS

**Step 4: Commit**

```bash
git add optim.py test_layers.py
git commit -m "feat: add SGD and Adam optimizers with tests"
```

---

### Task 7: Data Pipeline

**Files:**
- Create: `data.py`
- Modify: `test_layers.py`

**Step 1: Write CIFAR-10 loader, DataLoader, and augmentation**

```python
import numpy as np
import os
import pickle
import tarfile
import urllib.request


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


def load_cifar10(data_dir="./data"):
    """Download (if needed) and load CIFAR-10 as NumPy arrays.

    Returns:
        x_train: (50000, 3, 32, 32) float32 normalized to [0, 1]
        y_train: (50000,) int64
        x_test: (10000, 3, 32, 32) float32 normalized to [0, 1]
        y_test: (10000,) int64
    """
    cifar_dir = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.exists(cifar_dir):
        os.makedirs(data_dir, exist_ok=True)
        tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
        print(f"Downloading CIFAR-10 to {tar_path}...")
        urllib.request.urlretrieve(CIFAR10_URL, tar_path)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_dir)
        os.remove(tar_path)

    x_train, y_train = [], []
    for i in range(1, 6):
        path = os.path.join(cifar_dir, f"data_batch_{i}")
        with open(path, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        x_train.append(batch[b'data'])
        y_train.extend(batch[b'labels'])

    x_train = np.concatenate(x_train).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y_train = np.array(y_train, dtype=np.int64)

    with open(os.path.join(cifar_dir, "test_batch"), 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    x_test = batch[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y_test = np.array(batch[b'labels'], dtype=np.int64)

    return x_train, y_train, x_test, y_test


def normalize(x, mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)):
    """Channel-wise normalization for CIFAR-10."""
    mean = np.array(mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(std, dtype=np.float32).reshape(1, 3, 1, 1)
    return (x - mean) / std


def random_horizontal_flip(x, p=0.5):
    """Randomly flip images horizontally."""
    mask = np.random.rand(x.shape[0]) < p
    x[mask] = x[mask, :, :, ::-1]
    return x


def random_crop(x, padding=4):
    """Random crop with padding."""
    N, C, H, W = x.shape
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    out = np.zeros_like(x)
    for i in range(N):
        top = np.random.randint(0, 2 * padding)
        left = np.random.randint(0, 2 * padding)
        out[i] = padded[i, :, top:top + H, left:left + W]
    return out


class DataLoader:
    """Batched data loader with optional shuffling and augmentation."""

    def __init__(self, x, y, batch_size=64, shuffle=True, augment=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

    def __iter__(self):
        N = self.x.shape[0]
        indices = np.arange(N)
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, N, self.batch_size):
            idx = indices[start:start + self.batch_size]
            x_batch = self.x[idx].copy()
            y_batch = self.y[idx]
            if self.augment:
                x_batch = random_horizontal_flip(x_batch)
                x_batch = random_crop(x_batch)
            yield x_batch, y_batch

    def __len__(self):
        return (self.x.shape[0] + self.batch_size - 1) // self.batch_size
```

**Step 2: Add data pipeline tests**

Append to `test_layers.py`:

```python
from data import DataLoader, random_horizontal_flip, random_crop, normalize


def test_dataloader_batching():
    x = np.random.randn(100, 3, 32, 32).astype(np.float32)
    y = np.random.randint(0, 10, 100)
    loader = DataLoader(x, y, batch_size=32, shuffle=False)

    batches = list(loader)
    assert len(batches) == 4  # 32 + 32 + 32 + 4
    assert batches[0][0].shape[0] == 32
    assert batches[-1][0].shape[0] == 4


def test_augmentation_shapes():
    x = np.random.randn(8, 3, 32, 32).astype(np.float32)
    flipped = random_horizontal_flip(x.copy())
    assert flipped.shape == x.shape
    cropped = random_crop(x.copy(), padding=4)
    assert cropped.shape == x.shape


def test_normalize():
    x = np.random.rand(4, 3, 32, 32).astype(np.float32)
    normed = normalize(x)
    assert normed.shape == x.shape
    assert normed.dtype == np.float32
```

**Step 3: Run tests**

Run: `cd /Users/quinnarnold/Desktop/from-scratch/cnn && python -m pytest test_layers.py::test_dataloader_batching test_layers.py::test_augmentation_shapes test_layers.py::test_normalize -v`
Expected: 3 tests PASS

**Step 4: Commit**

```bash
git add data.py test_layers.py
git commit -m "feat: add CIFAR-10 data pipeline with DataLoader and augmentation"
```

---

### Task 8: SimpleCNN Model + Training Loop

**Files:**
- Create: `train.py`

**Step 1: Write SimpleCNN and training loop**

```python
import numpy as np
import argparse
import time
from layers import (Conv2d, MaxPool2d, ReLU, Flatten, Linear,
                    BatchNorm, Dropout, GlobalAvgPool2d, Model)
from functional import cross_entropy_loss
from optim import SGD
from data import load_cifar10, normalize, DataLoader


class SimpleCNN(Model):
    """Conv-Pool-Conv-Pool-FC architecture. Target ~65-70% on CIFAR-10."""

    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3, 32, 3, padding=1)
        self.bn1 = BatchNorm(32)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(2)

        self.conv2 = Conv2d(32, 64, 3, padding=1)
        self.bn2 = BatchNorm(64)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2d(2)

        self.flatten = Flatten()
        self.fc1 = Linear(64 * 8 * 8, 256)
        self.relu3 = ReLU()
        self.drop = Dropout(0.5)
        self.fc2 = Linear(256, 10)

        self._layer_order = [
            self.conv1, self.bn1, self.relu1, self.pool1,
            self.conv2, self.bn2, self.relu2, self.pool2,
            self.flatten, self.fc1, self.relu3, self.drop, self.fc2,
        ]

    def forward(self, x):
        for layer in self._layer_order:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self._layer_order):
            dout = layer.backward(dout)
        return dout


def evaluate(model, x, y, batch_size=128):
    model.eval()
    correct = 0
    total = 0
    loader = DataLoader(x, y, batch_size=batch_size, shuffle=False)
    for xb, yb in loader:
        logits = model.forward(xb)
        preds = np.argmax(logits, axis=1)
        correct += np.sum(preds == yb)
        total += yb.shape[0]
    return correct / total


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["simple", "resnet"], default="simple")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    print("Loading CIFAR-10...")
    x_train, y_train, x_test, y_test = load_cifar10()
    x_train = normalize(x_train)
    x_test = normalize(x_test)

    if args.model == "simple":
        model = SimpleCNN()
    else:
        model = SmallResNet()

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    train_loader = DataLoader(x_train, y_train, batch_size=args.batch_size,
                              shuffle=True, augment=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        start = time.time()

        for xb, yb in train_loader:
            logits = model.forward(xb)
            loss, dlogits = cross_entropy_loss(logits, yb)
            model.backward(dlogits)
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss
            num_batches += 1

        elapsed = time.time() - start
        avg_loss = epoch_loss / num_batches
        test_acc = evaluate(model, x_test, y_test)
        print(f"Epoch {epoch+1}/{args.epochs} | loss: {avg_loss:.4f} | "
              f"test acc: {test_acc:.4f} | time: {elapsed:.1f}s")


if __name__ == "__main__":
    train()
```

**Step 2: Verify forward/backward with small random data**

Run: `cd /Users/quinnarnold/Desktop/from-scratch/cnn && python -c "
import numpy as np
from train import SimpleCNN
from functional import cross_entropy_loss
model = SimpleCNN()
x = np.random.randn(4, 3, 32, 32).astype(np.float32)
y = np.array([0, 1, 2, 3])
logits = model.forward(x)
loss, dlogits = cross_entropy_loss(logits, y)
model.backward(dlogits)
print(f'Output shape: {logits.shape}, Loss: {loss:.4f}')
print('Forward + backward pass works.')
"`
Expected: prints output shape (4, 10) and loss, no errors.

**Step 3: Commit**

```bash
git add train.py
git commit -m "feat: add SimpleCNN model and training loop"
```

---

### Task 9: SmallResNet Model

**Files:**
- Modify: `train.py`

**Step 1: Add ResidualBlock and SmallResNet to train.py**

Add before the `train()` function:

```python
class ResidualBlock(Model):
    """Two conv layers with a skip connection."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = Conv2d(channels, channels, 3, padding=1)
        self.bn1 = BatchNorm(channels)
        self.relu1 = ReLU()
        self.conv2 = Conv2d(channels, channels, 3, padding=1)
        self.bn2 = BatchNorm(channels)
        self.relu2 = ReLU()

    def forward(self, x):
        identity = x
        out = self.relu1.forward(self.bn1.forward(self.conv1.forward(x)))
        out = self.bn2.forward(self.conv2.forward(out))
        out = out + identity
        out = self.relu2.forward(out)
        return out

    def backward(self, dout):
        dout = self.relu2.backward(dout)
        d_identity = dout.copy()
        d_residual = dout
        d_residual = self.conv2.backward(self.bn2.backward(d_residual))
        d_residual = self.conv1.backward(self.bn1.backward(self.relu1.backward(d_residual)))
        return d_identity + d_residual


class SmallResNet(Model):
    """Small ResNet: conv -> 3 res blocks -> pool -> 2 res blocks -> gap -> fc.
    Target ~85-90% on CIFAR-10.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = Conv2d(3, 64, 3, padding=1)
        self.bn1 = BatchNorm(64)
        self.relu1 = ReLU()

        self.block1 = ResidualBlock(64)
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.pool = MaxPool2d(2)

        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)

        self.gap = GlobalAvgPool2d()
        self.flatten = Flatten()
        self.fc = Linear(64, num_classes)

        self._stem = [self.conv1, self.bn1, self.relu1]
        self._blocks_pre = [self.block1, self.block2, self.block3]
        self._blocks_post = [self.block4, self.block5]
        self._head = [self.gap, self.flatten, self.fc]

    def forward(self, x):
        for layer in self._stem:
            x = layer.forward(x)
        for block in self._blocks_pre:
            x = block.forward(x)
        x = self.pool.forward(x)
        for block in self._blocks_post:
            x = block.forward(x)
        for layer in self._head:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self._head):
            dout = layer.backward(dout)
        for block in reversed(self._blocks_post):
            dout = block.backward(dout)
        dout = self.pool.backward(dout)
        for block in reversed(self._blocks_pre):
            dout = block.backward(dout)
        for layer in reversed(self._stem):
            dout = layer.backward(dout)
        return dout
```

**Step 2: Verify SmallResNet forward/backward**

Run: `cd /Users/quinnarnold/Desktop/from-scratch/cnn && python -c "
import numpy as np
from train import SmallResNet
from functional import cross_entropy_loss
model = SmallResNet()
x = np.random.randn(2, 3, 32, 32).astype(np.float32)
y = np.array([0, 1])
logits = model.forward(x)
loss, dlogits = cross_entropy_loss(logits, y)
model.backward(dlogits)
print(f'Output shape: {logits.shape}, Loss: {loss:.4f}')
print(f'Num parameters: {sum(p.size for p, _ in model.parameters())}')
print('SmallResNet forward + backward works.')
"`
Expected: prints output shape (2, 10), loss, parameter count, no errors.

**Step 3: Commit**

```bash
git add train.py
git commit -m "feat: add SmallResNet with residual blocks"
```

---

### Task 10: PyTorch Backend

**Files:**
- Modify: `layers.py`
- Modify: `train.py`
- Modify: `test_layers.py`

**Step 1: Add torch backend support to Model and layers**

Add `to_torch()` and `to_numpy()` methods to `Model` class. Each parameterized layer (Linear, Conv2d, BatchNorm) needs its numpy arrays converted to torch tensors. The forward/backward methods check `hasattr(self, 'use_torch') and self.use_torch` and use torch ops when enabled.

For Conv2d, the torch path uses `torch.nn.functional.conv2d` for forward and `torch.nn.functional.conv_transpose2d` for backward (or autograd). For Linear, use `torch.mm`. For BatchNorm, use `torch.nn.functional.batch_norm`.

This is the most involved task -- the key pattern for each layer:

```python
def forward(self, x):
    if getattr(self, 'use_torch', False):
        import torch
        # torch implementation
    else:
        # existing numpy implementation
```

Add to `Model`:

```python
def to_torch(self):
    import torch
    self._register_layers()
    for layer in self._layers:
        layer.use_torch = True
        if hasattr(layer, 'w') and not isinstance(layer.w, torch.Tensor):
            layer.w = torch.tensor(layer.w, dtype=torch.float32)
            layer.dw = torch.zeros_like(layer.w)
        if hasattr(layer, 'b') and not isinstance(layer.b, torch.Tensor):
            layer.b = torch.tensor(layer.b, dtype=torch.float32)
            layer.db = torch.zeros_like(layer.b)
        if hasattr(layer, 'gamma') and not isinstance(layer.gamma, torch.Tensor):
            layer.gamma = torch.tensor(layer.gamma, dtype=torch.float32)
            layer.dgamma = torch.zeros_like(layer.gamma)
            layer.beta = torch.tensor(layer.beta, dtype=torch.float32)
            layer.dbeta = torch.zeros_like(layer.beta)
            layer.running_mean = torch.tensor(layer.running_mean, dtype=torch.float32)
            layer.running_var = torch.tensor(layer.running_var, dtype=torch.float32)

def to_numpy(self):
    import torch
    self._register_layers()
    for layer in self._layers:
        layer.use_torch = False
        if hasattr(layer, 'w') and isinstance(layer.w, torch.Tensor):
            layer.w = layer.w.numpy()
            layer.dw = layer.dw.numpy()
        # ... same for b, gamma, beta, running_mean, running_var
```

**Step 2: Add integration test**

```python
def test_torch_backend_forward():
    np.random.seed(42)
    from train import SimpleCNN
    import torch

    model = SimpleCNN()
    x = np.random.randn(2, 3, 32, 32).astype(np.float32)

    out_np = model.forward(x.copy())

    model.to_torch()
    x_t = torch.tensor(x)
    out_torch = model.forward(x_t)

    np.testing.assert_allclose(
        out_np, out_torch.numpy(), atol=1e-4,
        err_msg="Torch backend should match NumPy forward"
    )
```

**Step 3: Run test**

Run: `cd /Users/quinnarnold/Desktop/from-scratch/cnn && python -m pytest test_layers.py::test_torch_backend_forward -v`
Expected: PASS

**Step 4: Commit**

```bash
git add layers.py train.py test_layers.py
git commit -m "feat: add optional PyTorch backend to all layers"
```

---

### Task 11: Integration -- Full Training Smoke Test

**Files:** None new

**Step 1: Run full test suite**

Run: `cd /Users/quinnarnold/Desktop/from-scratch/cnn && python -m pytest test_layers.py -v`
Expected: All tests PASS

**Step 2: Run 1-epoch smoke test on CIFAR-10**

Run: `cd /Users/quinnarnold/Desktop/from-scratch/cnn && python train.py --model simple --epochs 1 --batch-size 128`
Expected: Completes one epoch, prints loss and test accuracy.

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: CNN framework complete -- all tests passing"
```
