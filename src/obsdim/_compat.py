"""Array API compatibility helpers.

The numerical core is written against the Python Array API standard so that
the same code runs on NumPy arrays, PyTorch tensors, CuPy arrays, etc.  With
plain NumPy (>= 2.1) no extra dependency is needed; for other array libraries
install ``obsdim[array-api]`` (i.e. ``array-api-compat``), which provides the
standardized namespace for them.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - trivial import guard
    from array_api_compat import array_namespace as _array_namespace

    HAVE_ARRAY_API_COMPAT = True
except ImportError:  # pragma: no cover
    _array_namespace = None
    HAVE_ARRAY_API_COMPAT = False


def get_namespace(*arrays):
    """Return the array namespace of ``arrays`` (NumPy as fallback)."""
    arrays = [a for a in arrays if a is not None and not isinstance(a, (int, float))]
    if _array_namespace is not None:
        try:
            return _array_namespace(*arrays)
        except TypeError:
            return np
    return np


def as_float_array(x, xp):
    """Coerce ``x`` to a floating-point array of the namespace ``xp``."""
    x = xp.asarray(x)
    if not xp.isdtype(x.dtype, "real floating"):
        x = xp.astype(x, xp.float64)
    return x
