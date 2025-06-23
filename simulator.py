# simulator.py

from sklearn.neighbors import NearestNeighbors


class NearestNeighborSimulator:
    def __init__(self, X_scaled, y):
        self.nn = NearestNeighbors(n_neighbors=1).fit(X_scaled)
        self.y = y

    def evaluate(self, X_batch):
        _, idx = self.nn.kneighbors(X_batch)
        return [self.y[i] for i in idx.flatten()]
