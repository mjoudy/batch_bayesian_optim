# utils/helpers.py

import matplotlib.pyplot as plt


def plot_trace(y_history, title="Optimization Trace"):
    import matplotlib.pyplot as plt

    def compute_best_so_far(trace):
        best_so_far = []
        best = -float("inf")
        for y in trace:
            best = max(best, y)
            best_so_far.append(best)
        return best_so_far

    plt.figure(figsize=(8, 5))

    if isinstance(y_history, dict):
        for label, trace in y_history.items():
            best_trace = compute_best_so_far(trace)
            plt.plot(best_trace, label=label)
        plt.legend()
    else:
        best_trace = compute_best_so_far(y_history)
        plt.plot(best_trace)

    plt.title(title)
    plt.xlabel("Evaluations")
    plt.ylabel("Best Value So Far")
    plt.grid(True)
    plt.show()

def run_skopt(optimizer, simulator, iterations):
    history = []
    for i in range(iterations):
        Xb = optimizer.suggest_next_batch()
        yb = simulator.evaluate(Xb)
        optimizer.update(Xb, yb)
        history.extend(yb)
        print(f"[skopt] Iter {i+1}/{iterations} - Best: {max(optimizer.y):.4f}")
    return history


def run_botorch(optimizer, simulator, iterations):
    history = []
    for i in range(iterations):
        Xb = optimizer.suggest_next_batch()
        yb = simulator.evaluate(Xb)
        optimizer.update(Xb, yb)
        history.extend(yb)
        print(f"[botorch] Iter {i+1}/{iterations} - Best: {max(optimizer.y):.4f}")
    return history
