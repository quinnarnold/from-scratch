"""Training script for CIFAR-10 with from-scratch CNN framework."""

import argparse
import math
import time
import numpy as np

import backend as B
from data import load_cifar10, normalize, DataLoader
from functional import cross_entropy_loss


class StepLR:
    """Decay LR by gamma at each milestone epoch."""

    def __init__(self, optimizer, milestones, gamma=0.1):
        self.optimizer = optimizer
        self.milestones = set(milestones)
        self.gamma = gamma

    def step(self, epoch):
        if epoch in self.milestones:
            self.optimizer.lr *= self.gamma


class CosineLR:
    """Cosine annealing from initial LR to zero over total epochs."""

    def __init__(self, optimizer, total_epochs):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.total_epochs = total_epochs

    def step(self, epoch):
        self.optimizer.lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * epoch / self.total_epochs))


def compute_accuracy(model, loader):
    """Compute accuracy over a DataLoader."""
    model.set_eval()
    correct = 0
    total = 0
    for x_batch, y_batch in loader:
        xb = B.from_numpy(x_batch)
        logits = model.forward(xb)
        preds = B.to_numpy(B.argmax(logits, axis=1))
        correct += int(np.sum(preds == y_batch))
        total += len(y_batch)
    model.train()
    return correct / total


def train(args):
    if args.backend == 'torch':
        device = args.device
        if device == 'auto':
            import torch
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        B.set_backend('torch', device)
        print(f"Backend: PyTorch ({B.device()})")
    else:
        print("Backend: NumPy (CPU)")

    print("Loading CIFAR-10...")
    x_train, y_train, x_test, y_test = load_cifar10()
    x_train, mean, std = normalize(x_train)
    x_test, _, _ = normalize(x_test, mean, std)

    train_loader = DataLoader(x_train, y_train, batch_size=args.batch_size,
                              shuffle=True, augment=args.augment)
    test_loader = DataLoader(x_test, y_test, batch_size=args.batch_size,
                             shuffle=False, augment=False)

    from models import SimpleCNN, SmallResNet
    if args.model == "simple":
        model = SimpleCNN()
        print("Model: SimpleCNN")
    else:
        model = SmallResNet()
        print("Model: SmallResNet")

    params = model.parameters()
    print(f"Parameters: {sum(B.numel(p) for p, _ in params)}")

    from optim import SGD, Adam
    if args.optimizer == "sgd":
        optimizer = SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.schedule == "step":
        milestones = [int(m) for m in args.lr_milestones.split(",")]
        scheduler = StepLR(optimizer, milestones, gamma=args.lr_gamma)
        print(f"Optimizer: {args.optimizer}, LR: {args.lr}, Schedule: step (milestones={milestones}, gamma={args.lr_gamma})")
    elif args.schedule == "cosine":
        scheduler = CosineLR(optimizer, args.epochs)
        print(f"Optimizer: {args.optimizer}, LR: {args.lr}, Schedule: cosine")
    else:
        print(f"Optimizer: {args.optimizer}, LR: {args.lr}, Schedule: none")

    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, Augmentation: {args.augment}")
    print("-" * 60)

    model.train()

    best_test_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        if scheduler:
            scheduler.step(epoch)
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        t0 = time.time()

        for x_batch, y_batch in train_loader:
            xb = B.from_numpy(x_batch)
            yb = B.from_numpy(y_batch, dtype='long')

            optimizer.zero_grad()
            logits = model.forward(xb)
            loss, dlogits = cross_entropy_loss(logits, yb)
            model.backward(dlogits)
            optimizer.step()

            epoch_loss += float(loss) * len(y_batch)
            preds = B.to_numpy(B.argmax(logits, axis=1))
            epoch_correct += int(np.sum(preds == y_batch))
            epoch_total += len(y_batch)

        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total
        elapsed = time.time() - t0

        lr_str = f" | LR: {optimizer.lr:.6f}" if scheduler else ""

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            test_acc = compute_accuracy(model, test_loader)
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train: {train_acc:.4f} | "
                  f"Test: {test_acc:.4f} | "
                  f"Time: {elapsed:.1f}s{lr_str}")

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += args.eval_interval

            if args.early_stop and epochs_without_improvement >= args.early_stop:
                print(f"Early stopping: no improvement for {args.early_stop} epochs. "
                      f"Best test acc: {best_test_acc:.4f}")
                break
        else:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train: {train_acc:.4f} | "
                  f"Time: {elapsed:.1f}s{lr_str}")


def main():
    parser = argparse.ArgumentParser(description="Train CNN on CIFAR-10")
    parser.add_argument("--model", choices=["simple", "resnet"], default="simple")
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--early-stop", type=int, default=0,
                        help="Stop after N epochs with no test accuracy improvement (0=disabled)")
    parser.add_argument("--schedule", choices=["none", "step", "cosine"], default="none")
    parser.add_argument("--lr-milestones", default="30,40",
                        help="Comma-separated epochs for step decay (default: 30,40)")
    parser.add_argument("--lr-gamma", type=float, default=0.1,
                        help="Multiplicative factor for step decay (default: 0.1)")
    parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    parser.add_argument("--device", default="auto",
                        help="Device for torch backend: auto, cpu, mps, cuda")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
