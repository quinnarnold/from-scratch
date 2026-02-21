# CNN Framework Design

A from-scratch CNN framework in NumPy with manual backpropagation, targeting image classification on CIFAR-10. PyTorch used for gradient validation and as an optional runtime backend for speed.

## Architecture: Layer-Centric

Each layer is a class with `forward(x)` and `backward(dout)` methods. Layers cache what they need for backward internally. Models are containers that chain layers, with support for custom `forward()` to enable residual connections.

## File Structure

```
cnn/
  layers.py        -- All layer classes with forward/backward
  functional.py    -- Pure math (im2col, col2im, cross_entropy, weight init)
  optim.py         -- SGD (momentum, weight decay), Adam
  data.py          -- CIFAR-10 loading, DataLoader, augmentation
  train.py         -- Model definitions, training loop, entry point
  test_layers.py   -- PyTorch validation tests for each layer
```

## Layer API

```python
class Layer:
    def forward(self, x):       # cache internals, return output
    def backward(self, dout):   # use cache, return dx
    def parameters(self):       # list of (param, grad) tuples
    def train_mode(self, training=True):  # toggle train/eval
```

`Model` base class provides:
- `forward(x)` / `backward(dout)` -- sequential layer chaining
- `parameters()` -- collect from all layers
- `train()` / `eval()` -- toggle mode on all layers

Custom models subclass `Model` and override `forward()` for skip connections. Backward must be manually wired to route gradients through residual paths.

## Layers

| Layer | Key Implementation Detail |
|---|---|
| Conv2d(in_c, out_c, k, stride, padding) | im2col trick turns convolution into matrix multiply |
| MaxPool2d(k, stride) | Cache argmax indices for backward |
| AvgPool2d(k, stride) | Distribute gradient equally across window |
| GlobalAvgPool2d() | Average over (H,W), gradient is 1/(H*W) |
| BatchNorm(num_features) | Batch stats in train, running stats in eval |
| Dropout(p) | Random mask in train, identity in eval, scale by 1/(1-p) |
| ReLU | Cache mask of positive values |
| Flatten | Cache original shape for backward reshape |
| Linear(in, out) | Standard matmul + bias |
| Softmax | Used inside cross_entropy_loss for numerical stability |

## Functional Module

- `im2col(x, kH, kW, stride, padding)` / `col2im(...)` -- the core optimization for Conv2d
- `cross_entropy_loss(logits, targets)` -- returns (scalar loss, dlogits). Uses log-sum-exp for numerical stability. This is the backward entry point.
- `softmax(x)` -- stable softmax computation
- Weight initialization helpers (Kaiming, Xavier)

## Optimizers

- **SGD**: momentum, weight decay. Operates on `model.parameters()`.
- **Adam**: first/second moment tracking, bias correction.
- Learning rate scheduling: mutate `optimizer.lr` from the training loop directly.
- Both implement `step()` and `zero_grad()`.

## Data Pipeline

- `load_cifar10(data_dir)` -- download + return (x_train, y_train, x_test, y_test) as NumPy arrays in (N, C, H, W) float32 normalized to [0, 1].
- `DataLoader(x, y, batch_size, shuffle)` -- iterable yielding batches, shuffles per epoch.
- Augmentation: random horizontal flip, random crop with padding.

## PyTorch Backend (Optional)

Each layer has a `use_torch` flag. When set, forward/backward use `torch.Tensor` ops instead of NumPy. Activated via `model.to_torch()` which converts all parameters and flips the flag. Enables GPU training without changing model code.

## Validation Strategy

`test_layers.py` validates each layer by:
1. Creating random input
2. Running NumPy forward + backward
3. Running equivalent `torch.nn` layer
4. Asserting outputs and gradients match (`atol=1e-5`)

## Reference Models

- **SimpleCNN**: conv-pool-conv-pool-fc. Target ~65-70% on CIFAR-10.
- **SmallResNet**: residual blocks, BatchNorm, GlobalAvgPool. Target ~85-90% on CIFAR-10.
