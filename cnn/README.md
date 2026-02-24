# CNN From Scratch

A convolutional neural network framework built from scratch with NumPy, trained on CIFAR-10. Every layer, optimizer, and backpropagation routine is implemented manually -- no autograd, no `torch.nn`.

## Project Structure

```
functional.py    im2col/col2im, softmax, cross-entropy loss, weight init
layers.py        Layer/Model base classes, Conv2d, Linear, BatchNorm2d,
                 ReLU, MaxPool2d, AvgPool2d, GlobalAvgPool2d, Dropout, Flatten
models.py        Sequential, SimpleCNN, SmallResNet (with residual blocks)
optim.py         SGD (with momentum), Adam, weight decay
data.py          CIFAR-10 loader, normalization, data augmentation, DataLoader
train.py         Training loop with LR scheduling, early stopping, eval
backend.py       Array backend abstraction (NumPy or PyTorch for GPU)
```

### Test Suite

```
test_layers.py   Layer forward/backward validated against PyTorch
test_models.py   Model construction, training convergence, mode switching
test_optim.py    Optimizer convergence and PyTorch equivalence
test_data.py     Data pipeline, normalization, augmentation, DataLoader
```

**38 tests**, all passing. Each layer's forward and backward pass is numerically validated against PyTorch to ensure correctness.

## Models

**SimpleCNN** (620,810 params)
```
Conv(3->32, 3x3) -> BN -> ReLU -> MaxPool(2)
Conv(32->64, 3x3) -> BN -> ReLU -> MaxPool(2)
Conv(64->128, 3x3) -> BN -> ReLU -> MaxPool(2)
Flatten -> Linear(2048, 256) -> ReLU -> Dropout(0.5) -> Linear(256, 10)
```

**SmallResNet** (175,818 params)
```
Conv(3->16, 3x3) -> BN -> ReLU
ResBlock(16->16) x2
ResBlock(16->32, stride=2) + ResBlock(32->32)
ResBlock(32->64, stride=2) + ResBlock(64->64)
GlobalAvgPool -> Linear(64, 10)
```

## Usage

```bash
# SimpleCNN, 5 epochs, no augmentation
python train.py --model simple --epochs 5

# SmallResNet, 50 epochs, augmentation + cosine LR schedule
python train.py --model resnet --epochs 50 --augment --schedule cosine

# GPU-accelerated via PyTorch backend (same from-scratch models, just faster math)
python train.py --model resnet --epochs 50 --augment --schedule cosine --backend torch --device auto

# Step LR schedule with early stopping
python train.py --model resnet --epochs 50 --augment --schedule step --early-stop 10 --backend torch
```

## Training Results

All runs use SGD with momentum 0.9, weight decay 1e-4, batch size 64.

### SimpleCNN

| Run | Epochs | Augment | Schedule | Backend | Best Test Acc |
|-----|--------|---------|----------|---------|---------------|
| 1   | 5      | No      | None     | NumPy   | 70.3%         |
| 2   | 20     | No      | None     | Torch   | 78.8%         |
| 3   | 20     | Yes     | None     | Torch   | 79.4%         |
| 4   | 5      | No      | None     | Torch   | 71.7%         |

SimpleCNN converges quickly to ~70% in 5 epochs. Extending to 20 epochs pushes test accuracy to ~79%, but the train/test gap (89% vs 79%) shows clear overfitting. Data augmentation narrows this gap (77% train vs 79% test) with slightly better generalization.

### SmallResNet

| Run | Epochs | Augment | Schedule | Best Test Acc |
|-----|--------|---------|----------|---------------|
| 1   | 50     | Yes     | Step (30,40) | **88.3%** |
| 2   | 50     | Yes     | Cosine       | 87.6%     |

The step schedule achieves the best result at **88.3%** test accuracy, with a sharp jump at epoch 30 when LR drops from 0.01 to 0.001. Cosine annealing reaches 87.6% with smoother convergence. Both runs hit early 90s on training accuracy, indicating the model has capacity remaining.

### Run Details

#### SimpleCNN, 5 epochs, no augmentation (NumPy)
```
Epoch   1/5 | Loss: 1.6566 | Train: 0.3919 | Test: 0.5387 | Time: 58.7s
Epoch   2/5 | Loss: 1.3506 | Train: 0.5119 | Test: 0.5994 | Time: 58.9s
Epoch   3/5 | Loss: 1.1775 | Train: 0.5830 | Test: 0.6572 | Time: 59.2s
Epoch   4/5 | Loss: 1.0502 | Train: 0.6308 | Test: 0.6878 | Time: 60.8s
Epoch   5/5 | Loss: 0.9572 | Train: 0.6660 | Test: 0.7028 | Time: 63.8s
```

#### SimpleCNN, 20 epochs, no augmentation
```
Epoch   1/20 | Loss: 1.6636 | Train: 0.3842 | Test: 0.5037
Epoch   5/20 | Loss: 0.9426 | Train: 0.6694 | Test: 0.7157
Epoch  10/20 | Loss: 0.6216 | Train: 0.7842 | Test: 0.7726
Epoch  15/20 | Loss: 0.4268 | Train: 0.8502 | Test: 0.7813
Epoch  20/20 | Loss: 0.3112 | Train: 0.8899 | Test: 0.7772
```

#### SimpleCNN, 20 epochs, with augmentation
```
Epoch   1/20 | Loss: 1.7589 | Train: 0.3509 | Test: 0.5062
Epoch   5/20 | Loss: 1.1438 | Train: 0.5953 | Test: 0.6780
Epoch  10/20 | Loss: 0.8715 | Train: 0.6949 | Test: 0.7395
Epoch  15/20 | Loss: 0.7461 | Train: 0.7418 | Test: 0.7795
Epoch  20/20 | Loss: 0.6645 | Train: 0.7728 | Test: 0.7937
```

#### SmallResNet, 50 epochs, step schedule (early stopped at 48)
```
Epoch   1/50 | Loss: 1.5862 | Train: 0.4119 | Test: 0.4818 | LR: 0.010000
Epoch  10/50 | Loss: 0.5681 | Train: 0.8043 | Test: 0.8003 | LR: 0.010000
Epoch  20/50 | Loss: 0.4141 | Train: 0.8573 | Test: 0.8402 | LR: 0.010000
Epoch  29/50 | Loss: 0.3462 | Train: 0.8791 | Test: 0.8406 | LR: 0.010000
Epoch  30/50 | Loss: 0.2851 | Train: 0.9017 | Test: 0.8771 | LR: 0.001000  <- step decay
Epoch  38/50 | Loss: 0.2493 | Train: 0.9147 | Test: 0.8834 | LR: 0.001000  <- best
Epoch  40/50 | Loss: 0.2406 | Train: 0.9171 | Test: 0.8812 | LR: 0.000100
Early stopping: no improvement for 10 epochs. Best test acc: 0.8834
```

#### SmallResNet, 50 epochs, cosine schedule
```
Epoch   1/50 | Loss: 1.5673 | Train: 0.4178 | Test: 0.5513 | LR: 0.009990
Epoch  10/50 | Loss: 0.6017 | Train: 0.7899 | Test: 0.7857 | LR: 0.009045
Epoch  20/50 | Loss: 0.4167 | Train: 0.8560 | Test: 0.8288 | LR: 0.006545
Epoch  30/50 | Loss: 0.3197 | Train: 0.8897 | Test: 0.8612 | LR: 0.003455
Epoch  40/50 | Loss: 0.2571 | Train: 0.9109 | Test: 0.8708 | LR: 0.000955
Epoch  50/50 | Loss: 0.2412 | Train: 0.9170 | Test: 0.8763 | LR: 0.000000
```

## Runtime

Per-epoch wall-clock times on Apple Silicon:

| Model | NumPy (CPU) | PyTorch (MPS) |
|-------|-------------|---------------|
| SimpleCNN  | ~59s   | ~69s          |
| SmallResNet | --    | ~103s         |

NumPy and PyTorch MPS show comparable per-epoch times for SimpleCNN. The MPS backend doesn't provide a speedup here because the from-scratch im2col convolution implementation can't fully exploit GPU parallelism the way fused CUDA/MPS kernels in native PyTorch would -- the computation is still dominated by Python-level loops and memory copies between CPU and GPU. The benefit of the torch backend is access to larger GPU memory and potential speedups on models with heavier matrix multiplications.

SmallResNet has 3.5x fewer parameters than SimpleCNN but takes ~1.5x longer per epoch due to the deeper residual block structure (more sequential conv layers, skip connections adding extra forward/backward passes).

## Backend

The framework defaults to NumPy for all tensor operations. An optional PyTorch backend (`backend.py`) swaps in `torch.Tensor` for array math, enabling GPU acceleration on Apple Silicon (MPS) or NVIDIA (CUDA) without changing any model or training code. The models, layers, and backprop are still hand-written -- only the underlying array library changes.
