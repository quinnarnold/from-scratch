"""Training script for CIFAR-10 with from-scratch CNN framework."""

import argparse
import time
import numpy as np

from data import load_cifar10, normalize, DataLoader
from models import SimpleCNN, SmallResNet
from optim import SGD, Adam
from functional import cross_entropy_loss


def compute_accuracy(model, loader):
    """Compute accuracy over a DataLoader."""
    model.eval()
    correct = 0
    total = 0
    for x_batch, y_batch in loader:
        logits = model.forward(x_batch)
        preds = np.argmax(logits, axis=1)
        correct += np.sum(preds == y_batch)
        total += len(y_batch)
    model.train()
    return correct / total


def train(args):
    print(f"Loading CIFAR-10...")
    x_train, y_train, x_test, y_test = load_cifar10()

    x_train, mean, std = normalize(x_train)
    x_test, _, _ = normalize(x_test, mean, std)

    train_loader = DataLoader(x_train, y_train, batch_size=args.batch_size,
                              shuffle=True, augment=args.augment)
    test_loader = DataLoader(x_test, y_test, batch_size=args.batch_size,
                             shuffle=False, augment=False)

    if args.model == "simple":
        model = SimpleCNN()
        print("Model: SimpleCNN")
    else:
        model = SmallResNet()
        print("Model: SmallResNet")

    params = model.parameters()
    print(f"Parameters: {sum(p.size for p, _ in params)}")

    if args.optimizer == "sgd":
        optimizer = SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    print(f"Optimizer: {args.optimizer}, LR: {args.lr}, Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}, Augmentation: {args.augment}")
    print("-" * 60)

    model.train()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        t0 = time.time()

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            logits = model.forward(x_batch)
            loss, dlogits = cross_entropy_loss(logits, y_batch)

            model.backward(dlogits)
            optimizer.step()

            epoch_loss += loss * len(y_batch)
            preds = np.argmax(logits, axis=1)
            epoch_correct += np.sum(preds == y_batch)
            epoch_total += len(y_batch)

        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total
        elapsed = time.time() - t0

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            test_acc = compute_accuracy(model, test_loader)
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train: {train_acc:.4f} | "
                  f"Test: {test_acc:.4f} | "
                  f"Time: {elapsed:.1f}s")
        else:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train: {train_acc:.4f} | "
                  f"Time: {elapsed:.1f}s")


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
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
