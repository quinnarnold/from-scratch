##Implementing micrograd from scratch using memory and my own intuition. Code likely will not be exact. Hope to further conceptualize
# Currently understand all steps, but gradually confusing myself the more I think about the backward pass gradient applications.
# Implementing with no video will likely help.
import math
import numpy as np
import matplotlib.pyplot as plt
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


        ##start with parent node, add it to set
        #for each child in parent node, go into that child, if that child has no more children, add itself to the visit, if it has more children, go into
        # those children, etc etc, until no further children, where then the final child adds itself to the set, then recurse backwards, taking each node out of the loop and adding
        # itself into the visit set. Once all nodes in visit set, we append v, aka self. But within each child loop, they also not only add themself to the set, but if they dont
        # have more children, they also add themselves to the topo list. Thus, we have a set of visits and a topology list where deepest nodes with no children
        # occur first, and our parent occurs last. By reversing, we back propogate without errors.
        # if child has nodes:
            # child -> childs -> add childs to visit -> add childs to topo <- back out <- add child to visit <- add child to topo
        # if child has no nodes:
            # parent -> child -> add child to visit -> add child to topo <- back out <- add parent to visit <- add parent to topo
        #even more verbose explanation:
            # start from parent. go into child node, child node runs build topo, adds itself to visit, checks for any children
            # if no children, adds itself to visit again, then appends itself to topo
            # if children, go into children, etc etc, children append themselves, child appends itself to visit, to topo,
            # parent appends self to topo
        #beautiful representation of recursion and loops. very difficult to wrap the mind around though, since calling itself within the loop.

a = Value(2.0)
b = Value(6.0)
c = Value(-1.0)

print(a+b, a*b, b-a, b**a, b**c, b.exp(), b.tanh())
