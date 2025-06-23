# main.py

import argparse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from optimizer.skopt_optimizer import SkoptOptimizer
from optimizer.botorch_optimizer import BoTorchOptimizer
from simulator import NearestNeighborSimulator
from utils.helpers import plot_trace


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna()
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


def run_bo(optimizer, simulator, iterations=10):
    for i in range(iterations):
        Xb = optimizer.suggest_next_batch()
        yb = simulator.evaluate(Xb)
        optimizer.update(Xb, yb)
        print(f"Iteration {i+1}/{iterations} - Best so far: {max(optimizer.y):.4f}")
    return optimizer.y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="skopt", choices=["skopt", "botorch"])
    parser.add_argument("--dataset", type=str, default="CO2_to_methanol_catalysts.csv")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=5)
    args = parser.parse_args()

    X_scaled, y = load_data(args.dataset)
    simulator = NearestNeighborSimulator(X_scaled, y)

    dim = X_scaled.shape[1]
    if args.optimizer == "skopt":
        optimizer = SkoptOptimizer(dim, batch_size=args.batch_size)
    else:
        optimizer = BoTorchOptimizer(dim, batch_size=args.batch_size)

    history = run_bo(optimizer, simulator, iterations=args.iterations)
    plot_trace(history, title=f"{args.optimizer.upper()} Optimization Trace")
