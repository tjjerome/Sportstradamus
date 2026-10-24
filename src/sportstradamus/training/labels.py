"""Over/under outcome label shared by training's fits, metrics and ship gates."""

import numpy as np

# A tie pushes and the stake comes back, so it is half an Over. Matches
# helpers.distributions, whose P(under) is P(X < line) + P(X == line) / 2.
PUSH = 0.5


def over_label(result, line) -> np.ndarray:
    """Return 1.0 where ``result`` beats ``line``, 0.0 where it falls short, ``PUSH`` at a tie.

    Positional: ``result`` and ``line`` are same-length arrays or Series in the same row order.
    """
    result = np.asarray(result, dtype=float)
    line = np.asarray(line, dtype=float)
    return np.where(result == line, PUSH, (result > line).astype(float))
