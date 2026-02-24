"""Array backend abstraction: NumPy (CPU) or PyTorch (CPU/MPS/CUDA).

Call set_backend() before constructing any layers or models.
All tensor creation and math operations route through this module
so the same layer code works with either backend.
"""

import numpy as _np

_lib = 'numpy'
_device_str = 'cpu'
_torch = None
_torch_device = None


def set_backend(lib='numpy', device='cpu'):
    global _lib, _device_str, _torch, _torch_device
    _lib = lib
    _device_str = device
    if lib == 'torch':
        import torch
        _torch = torch
        if device == 'mps' and torch.backends.mps.is_available():
            _torch_device = torch.device('mps')
        elif device == 'cuda' and torch.cuda.is_available():
            _torch_device = torch.device('cuda')
        else:
            if device not in ('cpu', 'mps', 'cuda'):
                print(f"Unknown device '{device}', using CPU")
            elif device != 'cpu':
                print(f"{device} not available, falling back to CPU")
            _torch_device = torch.device('cpu')


def using_torch():
    return _lib == 'torch'


def device():
    return _torch_device if using_torch() else 'cpu'


# -- dtype helpers --

def _torch_dtype(np_dtype):
    if np_dtype is None or np_dtype == _np.float32:
        return _torch.float32
    if np_dtype == _np.int64:
        return _torch.int64
    return _torch.float32


# -- array creation --

def zeros(shape, dtype=None):
    if using_torch():
        return _torch.zeros(shape, dtype=_torch_dtype(dtype), device=_torch_device)
    return _np.zeros(shape, dtype=dtype or _np.float32)


def zeros_like(x):
    if using_torch():
        return _torch.zeros_like(x)
    return _np.zeros_like(x)


def ones(shape, dtype=None):
    if using_torch():
        return _torch.ones(shape, dtype=_torch_dtype(dtype), device=_torch_device)
    return _np.ones(shape, dtype=dtype or _np.float32)


def empty_like(x):
    if using_torch():
        return _torch.empty_like(x)
    return _np.empty_like(x)


def arange(n):
    if using_torch():
        return _torch.arange(n, device=_torch_device)
    return _np.arange(n)


# -- random --

def randn(*shape):
    if using_torch():
        return _torch.randn(*shape, dtype=_torch.float32, device=_torch_device)
    return _np.random.randn(*shape).astype(_np.float32)


def rand(*shape):
    if using_torch():
        return _torch.rand(*shape, device=_torch_device)
    return _np.random.rand(*shape).astype(_np.float32)


# -- math --

def sum(x, axis=None, keepdims=False):
    if using_torch():
        if axis is None:
            return _torch.sum(x)
        return _torch.sum(x, dim=axis, keepdim=keepdims)
    return _np.sum(x, axis=axis, keepdims=keepdims)


def mean(x, axis=None, keepdims=False):
    if using_torch():
        if axis is None:
            return _torch.mean(x)
        return _torch.mean(x, dim=axis, keepdim=keepdims)
    return _np.mean(x, axis=axis, keepdims=keepdims)


def var(x, axis=None):
    if using_torch():
        if axis is None:
            return _torch.var(x, correction=0)
        return _torch.var(x, dim=axis, correction=0)
    return _np.var(x, axis=axis)


def amax(x, axis=None, keepdims=False):
    if using_torch():
        if axis is None:
            return _torch.amax(x)
        return _torch.amax(x, dim=axis, keepdim=keepdims)
    return _np.max(x, axis=axis, keepdims=keepdims)


def argmax(x, axis=None):
    if using_torch():
        return _torch.argmax(x, dim=axis)
    return _np.argmax(x, axis=axis)


def sqrt(x):
    if using_torch():
        return _torch.sqrt(x)
    return _np.sqrt(x)


def exp(x):
    if using_torch():
        return _torch.exp(x)
    return _np.exp(x)


def log(x):
    if using_torch():
        return _torch.log(x)
    return _np.log(x)


# -- array operations --

def pad(x, pad_widths):
    """Pad array. pad_widths uses numpy convention: ((before, after), ...) per dim."""
    if using_torch():
        torch_pad = []
        for before, after in reversed(pad_widths):
            torch_pad.extend([before, after])
        return _torch.nn.functional.pad(x, torch_pad)
    return _np.pad(x, pad_widths)


def transpose(x, axes):
    if using_torch():
        return x.permute(*axes)
    return x.transpose(axes)


def broadcast_to(x, shape):
    if using_torch():
        return x.expand(shape).clone()
    return _np.broadcast_to(x, shape).copy()


def copy(x):
    if using_torch():
        return x.clone()
    return x.copy()


def to_float(x):
    if using_torch():
        return x.float()
    return x.astype(_np.float32)


def numel(x):
    if using_torch():
        return x.numel()
    return x.size


# -- conversion --

def to_numpy(x):
    if using_torch():
        return x.detach().cpu().numpy()
    return x


def from_numpy(x, dtype='float'):
    """Convert numpy array to current backend tensor."""
    if using_torch():
        dt = _torch.float32 if dtype == 'float' else _torch.long
        return _torch.tensor(x, dtype=dt, device=_torch_device)
    return x
