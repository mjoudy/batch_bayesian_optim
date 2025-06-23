# optimizer/botorch_optimizer.py

import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


class BoTorchOptimizer:
    def __init__(self, dim, batch_size=5):
        self.dim = dim
        self.batch_size = batch_size
        self.bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
        self.train_X = torch.empty((0, dim), dtype=torch.double)
        self.train_Y = torch.empty((0, 1), dtype=torch.double)

    def suggest_next_batch(self):
        if self.train_X.shape[0] == 0:
            return np.random.rand(self.batch_size, self.dim).tolist()

        model = SingleTaskGP(self.train_X, self.train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        acq_func = qExpectedImprovement(model=model, best_f=self.train_Y.max())
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=self.batch_size,
            num_restarts=5,
            raw_samples=64,
        )
        return candidates.detach().numpy().tolist()

    def update(self, Xb, yb):
        X_tensor = torch.tensor(Xb, dtype=torch.double)
        y_tensor = torch.tensor(yb, dtype=torch.double).unsqueeze(-1)
        self.train_X = torch.cat([self.train_X, X_tensor], dim=0)
        self.train_Y = torch.cat([self.train_Y, y_tensor], dim=0)

    @property
    def y(self):
        return self.train_Y.squeeze(-1).tolist()
