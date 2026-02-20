import math
from tkinter.constants import X
import numpy as np
import matplotlib.pyplot as plt
import random
%matplotlib inline

class Value():

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = _children
        self._op = _op
        self.grad = 0.0
        self._backward = lambda: None
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        ##a+b = out // dout/da = 1.0, dout/db = 1.0 // multiply by out.grad for chain rule to get proper grad
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        ##a*b = out // dout/da = b, dout/db = a // opposite variable of the one derivation is taking respect to // multiply by out.grad
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return -self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), '-')

        ##a - b = out // dout/da = 1, dout/db = -1 // must be careful with signs // multiply by out.grad
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += -1.0 * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data ** other.data, (self, ), f'**{other.data}')

        ##Everybody knows the power rule  :)
        def _backward():
            self.grad += other.data * (self.data**(other.data-1)) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        p = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(p, (self,), 'tanh')
        def _backward():
            self.grad += (1 - p**2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visit = set()
        def build_topo(v):
            if v not in visit:
                visit.add(v)
                for child in v._prev:
                    build_topo(child)
                    visit.add(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(self):
            node._backward()



##Building a Neuron
class Neuron:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:

    def __init__(self, nin, nout):
        sz = [nin] + nout
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nout))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        ps = [p for layer in self.layers for p in layer.parameters()]
        return ps

x = [2.0,3.0,-1.0]
n = MLP(3, [4,4,4,1])
n(x)
print(n.parameters())
print(len(n.parameters()))
