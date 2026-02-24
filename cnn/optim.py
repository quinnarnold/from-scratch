import backend as B


class SGD:
    """Stochastic Gradient Descent with momentum and weight decay."""

    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [B.zeros_like(p) for p, _ in params]

    def step(self):
        for i, (param, grad) in enumerate(self.params):
            g = B.copy(grad)
            if self.weight_decay != 0:
                g += self.weight_decay * param
            self.velocities[i] = self.momentum * self.velocities[i] + g
            param[:] -= self.lr * self.velocities[i]

    def zero_grad(self):
        for _, grad in self.params:
            grad[:] = 0.0


class Adam:
    """Adam optimizer with bias correction."""

    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8,
                 weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [B.zeros_like(p) for p, _ in params]
        self.v = [B.zeros_like(p) for p, _ in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, (param, grad) in enumerate(self.params):
            g = B.copy(grad)
            if self.weight_decay != 0:
                g += self.weight_decay * param
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            param[:] -= self.lr * m_hat / (B.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for _, grad in self.params:
            grad[:] = 0.0
