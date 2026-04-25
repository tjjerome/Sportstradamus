"""Back-compat shim: ``sportstradamus.sportstradamus`` → ``prediction`` package.

The prediction pipeline was split into ``sportstradamus.prediction.*`` in
Phase 4b of the maintainability refactor.  Any external code that imports
from this module continues to work via these re-exports.
"""

from sportstradamus.prediction import (
    find_correlation,
    main,
    match_offers,
    model_prob,
    process_offers,
    save_data,
)

__all__ = [
    "find_correlation",
    "main",
    "match_offers",
    "model_prob",
    "process_offers",
    "save_data",
]
