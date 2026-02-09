from micrograd.engine import Value

a = Value(6.0)
b = Value(-9.0)
c = a + b
d = c**2
d += (c - a).relu()
c = (d - a).relu() + c
e = (c + d) - (a + b) - d.relu()
f = e.relu()
g = f - b

print(f"g.data: {g.data:.4f}")  ## Forward pass outcome
g.backward()
print(f"a.grad: {a.grad:.4f}")  ## Backward pass dg/da
print(f"b.grad: {b.grad:.4f}")  ## Backward pass dg/db

x = g**2
y = (x - g).relu()
print(f"y.data: {y.data:.4f}")  ## Forward pass outcome
y.backward()
print(f"g.grad: {g.grad:.4f}")  ## Backward pass dy/dg
print(f"x.grad: {x.grad:.4f}")  ## Backward pass dy/dx
