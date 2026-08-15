"""Graph adapters: geometric data sets from graphs.

Two routes from a graph to an intrinsic dimension:

1. **Graph metric** — compute a distance matrix (shortest paths, diffusion
   distances, ...) and use ``OnePointDistances(metric="precomputed")``.
   This is the general mm-space route of the Tohoku paper.
2. **k-hop features** (TMLR 2023, Definition 5.1) — for an attributed graph,
   use the coordinate features of ``(X, ÂX, ..., Â^k X)`` normalized by the
   largest coordinate range, where ``Â`` is the self-loop-augmented
   symmetrically normalized adjacency matrix.  :func:`khop_features` builds
   exactly this family.

Only NumPy/SciPy are required; adjacency may be dense or ``scipy.sparse``.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from ._families import PrecomputedFeatures

__all__ = ["khop_feature_matrix", "khop_features", "shortest_path_distances"]


def _normalized_adjacency(adjacency):
    """``Â`` of TMLR Def. 5.1: ``Â_ij = 1/sqrt(deg_i deg_j)`` for j in N(v_i).

    ``N(v_i)`` includes ``v_i`` itself (self-loops), ``deg_i = |N(v_i)|``.
    """
    a = sparse.csr_matrix(adjacency, dtype=np.float64)
    if a.shape[0] != a.shape[1]:
        raise ValueError("adjacency must be square")
    a = a + sparse.eye(a.shape[0], format="csr")
    a.data[:] = 1.0  # unweighted neighborhoods, as in TMLR Def. 5.1
    deg = np.asarray(a.sum(axis=1)).ravel()
    inv_sqrt = sparse.diags(1.0 / np.sqrt(deg))
    return inv_sqrt @ a @ inv_sqrt


def khop_feature_matrix(adjacency, attributes, n_hops: int = 1):
    """Value matrix ``Phi`` of the k-hop feature functions (TMLR Def. 5.1).

    Rows are the features ``v_i -> (Â^m X)_{i, j} / d_max`` for hop depths
    ``m = 0..n_hops`` and attribute columns ``j``, with ``d_max`` the
    largest coordinate range of the attribute matrix ``X`` — so all
    features have range at most 1 and the dimension needs no further
    normalization.  Shape: ``((n_hops + 1) * n_attrs, n_nodes)``.
    """
    x = np.asarray(attributes, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("attributes must be 2-D (n_nodes, n_attrs)")
    d_max = float(np.max(x.max(axis=0) - x.min(axis=0))) if x.size else 0.0
    a_hat = _normalized_adjacency(adjacency)
    if a_hat.shape[0] != x.shape[0]:
        raise ValueError("adjacency and attributes disagree on n_nodes")
    blocks, current = [x.T], x
    for _ in range(n_hops):
        current = a_hat @ current
        blocks.append(current.T)
    phi = np.concatenate(blocks, axis=0)
    return phi / d_max if d_max > 0 else phi


def khop_features(adjacency, attributes, n_hops: int = 1) -> PrecomputedFeatures:
    """The k-hop feature family as a plug-in for :class:`~obsdim.GeometricID`.

    >>> GeometricID(khop_features(A, X, n_hops=2)).fit(None)  # doctest: +SKIP
    """
    return PrecomputedFeatures(khop_feature_matrix(adjacency, attributes, n_hops))


def shortest_path_distances(adjacency, *, directed: bool = False, unweighted=False):
    """Shortest-path distance matrix, ready for ``metric="precomputed"``.

    Thin wrapper over :func:`scipy.sparse.csgraph.shortest_path`; the result
    can be handed to ``GeometricID(OnePointDistances(metric="precomputed"))``.
    Raises if the graph is disconnected (the mm-space metric must be finite).
    """
    from scipy.sparse.csgraph import shortest_path

    d = shortest_path(
        sparse.csr_matrix(adjacency), directed=directed, unweighted=unweighted
    )
    if np.isinf(d).any():
        raise ValueError(
            "graph is disconnected; restrict to a connected component first"
        )
    return d
