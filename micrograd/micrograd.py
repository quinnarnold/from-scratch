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
        self.label = label

    def __repr__(self):
        return f'Value={self.data}'

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out

    def __sub__(self, other):
        out = Value(self.data + other.data, (self, other), '-')
        return out

    def tanh(self):
        x = self.data
        p = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(p, (self,), 'tanh')
        return out

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
