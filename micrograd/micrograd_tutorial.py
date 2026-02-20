import math
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

class Value():

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
        self.label = label

    def __repr__(self):
        return f'Value={self.data}'

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other**-1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supports int or float as of now"
        out = Value(self.data**other, (self,), f"**{other}")
        def _backward():
            self.grad += other * (self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), '-')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += -1.0 * out.grad
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

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def buildtopo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    buildtopo(child)
                    visited.add(child)
                topo.append(v)
        buildtopo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


a = Value(6.0, label='a')
b = Value(-4.0, label='b')
c = a-b ; c.label='c'
d = ((c+a) - b) + a*a ; d.label='d'
L = d * c ; L.label='L'

L.grad = 1.0

def standbyderiv():
    h = 0.0001
    a = Value(6.0, label='a')
    b = Value(-4.0, label='b')
    c = a-b ; c.label='c'
    d = ((c+a) - b) + a*a ; d.label='d'
    L = d * c ; L.label='L'
    L1 = L.data

    a = Value(6.0+h, label='a')
    b = Value(-4.0, label='b')
    c = a-b ; c.label='c'
    d = ((c+a) - b) + a*a ; d.label='d'
    L = d * c ; L.label='L'
    L2 = L.data

    print((L2-L1) / h)

standbyderiv()


##Simple Neuron
#inputs
x1 = Value(5.0, label='x1')
x2 = Value(1.0, label='x2')
#weights
w1 = Value(-1.0, label='w1')
w2 = Value(4.0, label='w2')
#bias
b = Value(2.0, label='b')
#Multiplication (simple matrix mult / linear equation)
x1w1 = x1*w1; x1w1.label='x1w1'
x2w2 = x2*w2; x2w2.label='x2w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label='x1w1+x2w2'
n = x1w1x2w2 + b; n.label='n'
o = n.tanh(); o.label='o'
print(n, o)

## do / do = 1.0 (gradient)
## do / w1 = do/dn * dn/x1w1x2w2 * dx1w1x2w2/x1w1 * dx1w1/w1
## do / x1 = do/dn * dn/x1w1x2w2 * dx1w1x2w2/x1w1 * dx1w1/x1


##Day 2
o.grad = 1.0
o._backward()
n._backward()
x1w1x2w2._backward()
b._backward()
x1w1._backward()
x2w2._backward()
print(o.grad, n.grad, x1w1x2w2.grad, b.grad, x1w1.grad, x2w2.grad, x1.grad, x2.grad, w1.grad, w2.grad)


x1 = Value(5.0, label='x1')
x2 = Value(1.0, label='x2')
#weights
w1 = Value(-1.0, label='w1')
w2 = Value(4.0, label='w2')
#bias
b = Value(2.0, label='b')
#Multiplication (simple matrix mult / linear equation)
x1w1 = x1*w1; x1w1.label='x1w1'
x2w2 = x2*w2; x2w2.label='x2w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label='x1w1+x2w2'
n = x1w1x2w2 + b; n.label='n'
o = n.tanh(); o.label='o'
print(n, o)
o.backward()
print(o.grad, n.grad, x1w1x2w2.grad, b.grad, x1w1.grad, x2w2.grad, x1.grad, x2.grad, w1.grad, w2.grad)
