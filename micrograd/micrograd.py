import math
import random

import matplotlib.pyplot as plt
import numpy as np


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self._prev = _children
        self._op = _op
        self.grad = 0.0
        self._backward = lambda: None

    def __add__(self, x):
        x = x if isinstance(x, Value) else Value(x)
        out = Value(self.data + x.data, (self, x), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            x.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, x):
        x = x if isinstance(x, Value) else Value(x)
        out = Value(self.data * x.data, (self, x), "*")

        def _backward():
            self.grad += x.data * out.grad
            x.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, x):
        x = x if isinstance(x, Value) else Value(x)
        out = Value(self.data - x.data, (self, x), "-")

        def _backward():
            self.grad += 1.0 * out.grad
            x.grad += -1.0 * out.grad

        out._backward = _backward
        return out

    def __pow__(self, x):
        """Attempted to remake power function
        with derivative for both exponent
        and base. Used partial derivative log
        rules for constant base gradient."""
        x = x if isinstance(x, Value) else Value(x)
        out = Value(self.data**x.data, (self, x), f"**{x.data}")

        def _backward():
            self.grad += x.data * self.data ** (x.data - 1) * out.grad
            x.grad += self.data**x.data * math.log(self.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        p = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(p, (self,), "tanh")

        def _backward():
            self.grad += (1 - p**2) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other**-1

    def __repr__(self):
        return f"Value(data={self.data})"

    def __rmul__(self, x):
        return self * x

    def __rsub__(self, x):
        out = Value(x + (-self.data), (self,), "-")
        return out

    def __radd__(self, x):
        return self + x

    def backward(self):
        topo = []
        visit = set()

        def buildtopo(v):
            if v not in visit:
                visit.add(v)
                for child in v._prev:
                    buildtopo(child)
                    visit.add(child)
                topo.append(v)

        buildtopo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


a = Value(2)
b = Value(3)
c = a + b
d = c**2
k = c + d - a**2
L = k + b
L.backward()
print(L.grad, k.grad, b.grad, d.grad)
