# ARCHIVED 2026-10-11 from src/sportstradamus/helpers/text.py
# Reason: merge_dict, get_trends and hmean had no caller in src/sportstradamus/, tests/ or
#         scripts/; only the package re-export in helpers/__init__.py named them.
# Last live SHA: a63b9107
# Original imports (now unresolved here):
#   import numpy as np
#   import pandas as pd

# From src/sportstradamus/helpers/text.py. Only get_trends read these two constants.
#
# Trailing rows used to fit the recent-form slope in ``get_trends``; below
# MIN_TREND_ROWS the fit is too noisy to trust, so the slope is reported as 0.
TREND_WINDOW = 5
MIN_TREND_ROWS = 3


# From src/sportstradamus/helpers/text.py; re-exported as sportstradamus.helpers.merge_dict.
# Archived code in this folder still calls it: Archive.merge and Archive.rename_market in
# helpers_orphans.py, and the script in clean_archive.py.
def merge_dict(a, b, path=None):
    """Recursively merge ``b`` into ``a``, in place. Returns ``a``.

    The archive uses a few keys (``Line``, ``EV``) with domain-specific
    merge semantics rather than the default overwrite — ``Line`` lists
    extend, ``EV`` dicts update, everything else gets overwritten.
    """
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dict(a[key], b[key], [*path, str(key)])
            elif np.all(a[key] == b[key]):
                pass  # same leaf value
            elif key == "Line":
                a[key].extend(b[key])
            elif key == "EV":
                a[key].update(b[key])
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a


# From src/sportstradamus/helpers/text.py; re-exported as sportstradamus.helpers.get_trends.
def get_trends(x):
    """Return per-column slope of the trailing 5 rows of ``x`` as a Series."""
    if len(x) < MIN_TREND_ROWS:
        trend = np.zeros(len(x.columns))
    else:
        recent = x.tail(TREND_WINDOW)
        trend = np.polyfit(np.arange(0, len(recent)), recent, 1)[0]
    return pd.Series(trend, index=x.columns)


# From src/sportstradamus/helpers/text.py; re-exported as sportstradamus.helpers.hmean.
def hmean(items):
    """Harmonic mean of numeric ``items``, ignoring zeros and strings.

    Returns ``0`` when no finite non-zero values are present (callers rely
    on the zero sentinel rather than a raised exception).
    """
    total = 0
    count = 0
    for i in items:
        if (i != 0) and type(i) is not str:
            count += 1
            total += 1 / i

    if total != 0:
        return count / total
    return 0
