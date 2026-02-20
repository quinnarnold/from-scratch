import math
import random

import matplotlib.pyplot as plt
import numpy as np

from microgradv2 import Value

x = [2, 3, 4, -1]
##4 Dimensional Input. Input must go into a neuron


class Neuron:
    def __init__(self, nin):
        """Initialize neuron with weights for inputs
        and single bias term"""
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


a = Neuron(4)
a(x)


class Layer:
    def __init__(self, nin, nout):
        """Initialize layer with # of inputs and
        number of desired neurons."""
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MCP:
    def __init__(self, nin, nout):
        """Initialize MCP with number of inputs and list
        of desired layer sizes."""
        sz = [nin] + nout
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nout))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)  # output of one layer becomes input to the next
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


a = MCP(4, [6, 6, 6, 4, 1])
print(len(a.parameters()))
a(x)
