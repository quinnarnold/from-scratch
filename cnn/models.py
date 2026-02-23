import numpy as np
from layers import (
    Model, Layer, Conv2d, BatchNorm2d, ReLU, MaxPool2d,
    Linear, Flatten, Dropout, GlobalAvgPool2d,
)
from functional import cross_entropy_loss


class Sequential(Model):
    """Container that chains layers in order."""

    def __init__(self, *layers):
        super().__init__()
        self.layer_list = list(layers)
        for i, layer in enumerate(self.layer_list):
            setattr(self, f'_seq_{i}', layer)

    def forward(self, x):
        for layer in self.layer_list:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layer_list):
            dout = layer.backward(dout)
        return dout


class SimpleCNN(Model):
    """Simple CNN for CIFAR-10. Target: ~65-70% accuracy.

    Architecture:
        Conv(3->32, 3x3, pad=1) -> BN -> ReLU -> MaxPool(2)
        Conv(32->64, 3x3, pad=1) -> BN -> ReLU -> MaxPool(2)
        Conv(64->128, 3x3, pad=1) -> BN -> ReLU -> MaxPool(2)
        Flatten -> Linear(2048, 256) -> ReLU -> Dropout(0.5) -> Linear(256, 10)
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = Sequential(
            Conv2d(3, 32, 3, padding=1),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(2),
        )
        self.block2 = Sequential(
            Conv2d(32, 64, 3, padding=1),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(2),
        )
        self.block3 = Sequential(
            Conv2d(64, 128, 3, padding=1),
            BatchNorm2d(128),
            ReLU(),
            MaxPool2d(2),
        )
        self.flatten = Flatten()
        self.classifier = Sequential(
            Linear(128 * 4 * 4, 256),
            ReLU(),
            Dropout(0.5),
            Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1.forward(x)
        x = self.block2.forward(x)
        x = self.block3.forward(x)
        x = self.flatten.forward(x)
        x = self.classifier.forward(x)
        return x

    def backward(self, dout):
        dout = self.classifier.backward(dout)
        dout = self.flatten.backward(dout)
        dout = self.block3.backward(dout)
        dout = self.block2.backward(dout)
        dout = self.block1.backward(dout)
        return dout


class ResidualBlock(Model):
    """Basic residual block with two 3x3 convolutions and a skip connection.

    If in_channels != out_channels, a 1x1 convolution projects the shortcut.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.cache = {}
        self.conv1 = Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = BatchNorm2d(out_channels)
        self.relu1 = ReLU()

        self.conv2 = Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = BatchNorm2d(out_channels)

        self.use_shortcut = (stride != 1) or (in_channels != out_channels)
        if self.use_shortcut:
            self.shortcut_conv = Conv2d(in_channels, out_channels, 1, stride=stride, padding=0)
            self.shortcut_bn = BatchNorm2d(out_channels)

        self.relu_out = ReLU()

    def forward(self, x):
        self.cache['input'] = x

        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)

        if self.use_shortcut:
            shortcut = self.shortcut_conv.forward(x)
            shortcut = self.shortcut_bn.forward(shortcut)
        else:
            shortcut = x

        self.cache['shortcut'] = shortcut
        out = out + shortcut
        out = self.relu_out.forward(out)
        return out

    def backward(self, dout):
        dout = self.relu_out.backward(dout)

        d_main = self.bn2.backward(dout)
        d_main = self.conv2.backward(d_main)
        d_main = self.relu1.backward(d_main)
        d_main = self.bn1.backward(d_main)
        d_main = self.conv1.backward(d_main)

        if self.use_shortcut:
            d_shortcut = self.shortcut_bn.backward(dout)
            d_shortcut = self.shortcut_conv.backward(d_shortcut)
        else:
            d_shortcut = dout

        return d_main + d_shortcut


class SmallResNet(Model):
    """Small ResNet for CIFAR-10. Target: ~85-90% accuracy.

    Architecture:
        Conv(3->16, 3x3, pad=1) -> BN -> ReLU
        ResBlock(16->16) x2
        ResBlock(16->32, stride=2) + ResBlock(32->32)
        ResBlock(32->64, stride=2) + ResBlock(64->64)
        GlobalAvgPool -> Linear(64, 10)
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = Conv2d(3, 16, 3, padding=1)
        self.bn1 = BatchNorm2d(16)
        self.relu1 = ReLU()

        self.layer1_0 = ResidualBlock(16, 16)
        self.layer1_1 = ResidualBlock(16, 16)

        self.layer2_0 = ResidualBlock(16, 32, stride=2)
        self.layer2_1 = ResidualBlock(32, 32)

        self.layer3_0 = ResidualBlock(32, 64, stride=2)
        self.layer3_1 = ResidualBlock(64, 64)

        self.pool = GlobalAvgPool2d()
        self.flatten = Flatten()
        self.fc = Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.bn1.forward(x)
        x = self.relu1.forward(x)

        x = self.layer1_0.forward(x)
        x = self.layer1_1.forward(x)

        x = self.layer2_0.forward(x)
        x = self.layer2_1.forward(x)

        x = self.layer3_0.forward(x)
        x = self.layer3_1.forward(x)

        x = self.pool.forward(x)
        x = self.flatten.forward(x)
        x = self.fc.forward(x)
        return x

    def backward(self, dout):
        dout = self.fc.backward(dout)
        dout = self.flatten.backward(dout)
        dout = self.pool.backward(dout)

        dout = self.layer3_1.backward(dout)
        dout = self.layer3_0.backward(dout)

        dout = self.layer2_1.backward(dout)
        dout = self.layer2_0.backward(dout)

        dout = self.layer1_1.backward(dout)
        dout = self.layer1_0.backward(dout)

        dout = self.relu1.backward(dout)
        dout = self.bn1.backward(dout)
        dout = self.conv1.backward(dout)
        return dout
