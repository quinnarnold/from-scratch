"""Tests for model architectures and training integration."""

import numpy as np
import pytest
from models import Sequential, SimpleCNN, ResidualBlock, SmallResNet
from layers import Linear, ReLU
from functional import cross_entropy_loss
from optim import SGD


class TestSequential:

    def test_forward_backward(self):
        seq = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
        x = np.random.randn(3, 4).astype(np.float32)
        out = seq.forward(x)
        assert out.shape == (3, 2)

        dout = np.random.randn(3, 2).astype(np.float32)
        dx = seq.backward(dout)
        assert dx.shape == (3, 4)

    def test_parameters(self):
        seq = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
        params = seq.parameters()
        assert len(params) == 4  # 2 layers x (weight, bias)


class TestSimpleCNN:

    def test_forward_shape(self):
        model = SimpleCNN(num_classes=10)
        x = np.random.randn(2, 3, 32, 32).astype(np.float32)
        out = model.forward(x)
        assert out.shape == (2, 10)

    def test_backward_shape(self):
        model = SimpleCNN(num_classes=10)
        x = np.random.randn(2, 3, 32, 32).astype(np.float32)
        out = model.forward(x)
        loss, dlogits = cross_entropy_loss(out, np.array([3, 7]))
        dx = model.backward(dlogits)
        assert dx.shape == (2, 3, 32, 32)

    def test_parameter_count(self):
        model = SimpleCNN()
        params = model.parameters()
        total = sum(p.size for p, _ in params)
        assert total > 0
        ids = [id(p) for p, _ in params]
        assert len(ids) == len(set(ids))

    def test_train_and_mode_switching(self):
        model = SimpleCNN()
        model.train()
        model.eval()
        model.train()


class TestResidualBlock:

    def test_same_channels(self):
        block = ResidualBlock(16, 16)
        x = np.random.randn(2, 16, 8, 8).astype(np.float32)
        out = block.forward(x)
        assert out.shape == (2, 16, 8, 8)

        dout = np.random.randn(2, 16, 8, 8).astype(np.float32)
        dx = block.backward(dout)
        assert dx.shape == (2, 16, 8, 8)

    def test_downsample(self):
        block = ResidualBlock(16, 32, stride=2)
        x = np.random.randn(2, 16, 8, 8).astype(np.float32)
        out = block.forward(x)
        assert out.shape == (2, 32, 4, 4)

        dout = np.random.randn(2, 32, 4, 4).astype(np.float32)
        dx = block.backward(dout)
        assert dx.shape == (2, 16, 8, 8)


class TestSmallResNet:

    def test_forward_shape(self):
        model = SmallResNet(num_classes=10)
        x = np.random.randn(2, 3, 32, 32).astype(np.float32)
        out = model.forward(x)
        assert out.shape == (2, 10)

    def test_backward_shape(self):
        model = SmallResNet(num_classes=10)
        x = np.random.randn(2, 3, 32, 32).astype(np.float32)
        out = model.forward(x)
        loss, dlogits = cross_entropy_loss(out, np.array([3, 7]))
        dx = model.backward(dlogits)
        assert dx.shape == (2, 3, 32, 32)


class TestTrainingIntegration:
    """Verify the full train loop converges on a tiny dataset."""

    def test_simplecnn_overfits_tiny_batch(self):
        np.random.seed(42)
        model = SimpleCNN(num_classes=10)
        x = np.random.randn(8, 3, 32, 32).astype(np.float32)
        y = np.array([0, 1, 2, 3, 4, 5, 6, 7])

        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        model.train()

        initial_loss = None
        for step in range(50):
            optimizer.zero_grad()
            logits = model.forward(x)
            loss, dlogits = cross_entropy_loss(logits, y)
            model.backward(dlogits)
            optimizer.step()
            if initial_loss is None:
                initial_loss = loss

        assert loss < initial_loss, "Loss should decrease during training"

    def test_resnet_overfits_tiny_batch(self):
        np.random.seed(42)
        model = SmallResNet(num_classes=10)
        x = np.random.randn(4, 3, 32, 32).astype(np.float32)
        y = np.array([0, 1, 2, 3])

        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        model.train()

        initial_loss = None
        for step in range(30):
            optimizer.zero_grad()
            logits = model.forward(x)
            loss, dlogits = cross_entropy_loss(logits, y)
            model.backward(dlogits)
            optimizer.step()
            if initial_loss is None:
                initial_loss = loss

        assert loss < initial_loss, "Loss should decrease during training"
