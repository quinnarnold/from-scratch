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
