# ARCHIVED 2026-04-21 from src/sportstradamus/helpers.py
# Reason: Function-level orphans — defined in helpers.py but never called from
#         any reachable entry point. Bundled here so the live helpers package
#         can shrink without losing the implementations.
# Last live SHA: 871657e
# Original imports (now unresolved here):
#   import requests
#   import numpy as np
#   import pandas as pd
#   from scipy.integrate import dblquad
#   from sportstradamus import creds
#   # `odds_api` was a module-level constant loaded from creds/keys.json.


def get_active_sports():
    """Returned the subset of NBA/MLB/NHL/NFL currently marked active by the odds API.

    Originally lived at helpers.py:80. Replaced in practice by the per-league
    flags in `data/active_leagues.json` (or equivalent) and never called.
    """
    # Get available sports from the API
    url = f"https://api.the-odds-api.com/v4/sports/?apiKey={odds_api}"  # noqa: F821
    res = requests.get(url)  # noqa: F821
    res = res.json()

    # Filter sports
    sports = [
        s["title"]
        for s in res
        if s["title"] in ["NBA", "MLB", "NHL", "NFL"] and s["active"]
    ]

    return sports


def prob_diff(X, Y, line):
    """Probability that X - Y > line, computed as a 2D integral of the joint PDF.

    `X`, `Y` are PDFs (callables). Originally lived at helpers.py:1128. Used
    nowhere; the project's predictive math takes a different route.
    """

    def joint_pdf(x, y):
        return X(x) * Y(y)

    return dblquad(joint_pdf, -np.inf, np.inf, lambda x: x - line, np.inf)  # noqa: F821


def prob_sum(X, Y, line):
    """Probability that X + Y < line, computed as a 2D integral of the joint PDF.

    Mirror of `prob_diff`. Originally helpers.py:1135. Never called.
    """

    def joint_pdf(x, y):
        return X(x) * Y(y)

    return dblquad(joint_pdf, -np.inf, np.inf, -np.inf, lambda x: line - x)  # noqa: F821


def accel_asc(n):
    """Generator yielding all integer partitions of `n` (Kelleher's algorithm).

    Originally helpers.py:1164. Likely intended for parlay-leg combinatorics
    (the same problem opt_parlay.py solved), but never wired in.
    """
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1  # noqa: E741
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[: k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[: k + 1]
