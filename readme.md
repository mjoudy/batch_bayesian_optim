# 🔬 Batch Bayesian Optimization for Expensive Experiments

This repository provides a modular and extensible framework to compare different Batch Bayesian Optimization (BBO) strategies on multiple datasets, targeting expensive black-box functions—where each experiment or evaluation can cost time, resources, or money.

Bayesian Optimization is ideal for such problems due to its sample efficiency. Batch BO methods extend this by enabling parallel evaluations, speeding up the search process.


#### ⚙️ Two backend libraries supported:

**BoTorch:** Flexible and powerful PyTorch-based BO library.

**scikit-optimize (skopt):** Lightweight and easy-to-use BO toolkit.

