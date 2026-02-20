import math
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

##Testing
def f(x):
    """Input scalar; output scalar"""
    return (4*x**2) - 6*x + 2

print(f(5.0))

xs = np.arange(-5, 5.25, 0.25)
ys = f(xs)
plt.plot(xs,ys)
plt.show()

##Derivatives
h = 0.01
x1 = 2.0
x2 = -2.0
print(f'f(x1): {f(x1)}')
print(f'f(x2): {f(x2)}')

print(f'Derivative at f1: {f(x1+h) - f(x1) / h}')
print(f'Derivative at f2: {f(x2+h) - f(x2) / h}')

##Complexify
a = 6.0
b = -2.0
c = 3.0
d = c*a + b
print(d)

##Manual Derivative Calculation
h = 0.0001
d1 = c*a + b
a += h
d2 = c*a + b
print('d1:', d1)
print('d2:', d2)
print('Derivative of dd/da:', (d2 - d1) / h) ## Derivative of our function D with respect to a, where D(a, b, c) = c*a + b
