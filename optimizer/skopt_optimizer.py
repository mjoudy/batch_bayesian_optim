# optimizer/skopt_optimizer.py

from skopt import Optimizer
from skopt.space import Real, Integer, Categorical


class SkoptOptimizer:
    def __init__(
        self,
        dim,
        batch_size=1,
        acq_func="EI",
        acq_optimizer="auto",
        n_initial_points=10,
        space_bounds=None,
        space_type="real",
        categorical_values=None,
    ):
        self.dim = dim
        self.batch_size = batch_size
        self.X, self.y = [], []

        if space_bounds:
            self.space = [Real(*b, name=f"x{i}") for i, b in enumerate(space_bounds)]
        elif space_type == "real":
            self.space = [Real(0.0, 1.0, name=f"x{i}") for i in range(dim)]
        elif space_type == "int":
            self.space = [Integer(0, 10, name=f"x{i}") for i in range(dim)]
        elif space_type == "categorical" and categorical_values:
            self.space = [Categorical(vals, name=f"x{i}") for i, vals in enumerate(categorical_values)]
        else:
            raise ValueError("Invalid space configuration")

        self.opt = Optimizer(
            dimensions=self.space,
            base_estimator="GP",
            acq_func=acq_func,
            acq_optimizer=acq_optimizer,
            n_initial_points=n_initial_points,
            random_state=42
        )

    def suggest_next_batch(self):
        return self.opt.ask(n_points=self.batch_size)

    def update(self, Xb, yb):
        self.opt.tell(Xb, yb)
        self.X += Xb
        self.y += yb

    @property
    def best_point(self):
        if not self.y:
            return None
        idx = self.y.index(min(self.y))
        return self.X[idx]

    @property
    def best_value(self):
        if not self.y:
            return None
        return min(self.y)
