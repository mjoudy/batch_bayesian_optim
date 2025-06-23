# optimizer/skopt_optimizer.py

from skopt import Optimizer
from skopt.space import Real


class SkoptOptimizer:
    def __init__(self, dim, batch_size=5):
        self.dim = dim
        self.batch_size = batch_size
        self.space = [Real(0.0, 1.0, name=f"x{i}") for i in range(dim)]
        self.opt = Optimizer(self.space, base_estimator="GP")
        self.X, self.y = [], []

    def suggest_next_batch(self):
        return self.opt.ask(n_points=self.batch_size)

    def update(self, Xb, yb):
        self.opt.tell(Xb, yb)
        self.X += Xb
        self.y += yb
