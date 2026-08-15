"""General metric-measure spaces: precomputed distances and graphs.

The theory lives on mm-spaces, not on R^n: everything the dimension needs
is a distance matrix (any metric) or, even more generally, feature values.
That makes graphs first-class citizens.
"""

import numpy as np

from obsdim import GeometricID, OnePointDistances
from obsdim.graphs import khop_features, shortest_path_distances

rng = np.random.default_rng(0)

# --- Route 1: a graph metric ------------------------------------------------
# Cycle graph C_n with the shortest-path metric.
n = 60
adj = np.zeros((n, n))
for i in range(n):
    adj[i, (i + 1) % n] = adj[(i + 1) % n, i] = 1

dist = shortest_path_distances(adj, unweighted=True)
est = GeometricID(OnePointDistances(metric="precomputed")).fit(dist)
print(f"cycle graph C_{n} (shortest paths):  dimension = {est.dimension_:.2f}")

# A random graph concentrates much more (most nodes are 2-3 hops apart):
adj_rnd = (rng.uniform(size=(n, n)) < 0.15).astype(float)
adj_rnd = np.triu(adj_rnd, 1)
adj_rnd += adj_rnd.T
dist = shortest_path_distances(adj_rnd, unweighted=True)
est = GeometricID(OnePointDistances(metric="precomputed")).fit(dist)
print(f"G(n, 0.15) random graph:            dimension = {est.dimension_:.2f}")

# --- Route 2: k-hop attribute features (TMLR 2023, Def. 5.1) ----------------
# For attributed graphs: coordinate features of (X, A_hat X, ..., A_hat^k X),
# normalized by the largest attribute range -- the construction used for the
# OGB experiments in the TMLR paper.
attrs = rng.normal(size=(n, 8))
est = GeometricID(khop_features(adj, attrs, n_hops=2)).fit(None)
print(f"k-hop features (k=2) on C_{n}:      dimension = {est.dimension_:.2f}")
