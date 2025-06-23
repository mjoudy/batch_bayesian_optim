# utils/helpers.py

import matplotlib.pyplot as plt


def plot_trace(y_history, title="Optimization Trace"):
    best_so_far = []
    best = -float("inf")
    for y in y_history:
        best = max(best, y)
        best_so_far.append(best)
    plt.plot(best_so_far)
    plt.xlabel("Evaluations")
    plt.ylabel("Best Yield So Far")
    plt.title(title)
    plt.show()
