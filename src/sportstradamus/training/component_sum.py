"""Price a combo cell as a weighted sum of its component cells' own predictives.

Lane B of the combo-market handoff. ``Stats.combo_quote`` feeds
:func:`helpers.combined_markets.combo_sum_quote` with book-inverted component
marginals; this module feeds the same kernel with each component cell's *served*
predictive (:mod:`training.component_cells`). No pickle, feature matrix, or archive
read is involved, which is what makes the cells Lane A's ``book_direct`` admission
could never reach gradeable. Nothing here produces a ``TrainingQuote``: this is
evidence for the ship scorecard, not a serving path.
"""

from collections.abc import Callable
from functools import partial
from importlib import resources as pkg_resources
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from sportstradamus import data
from sportstradamus.helpers.combined_markets import (
    ComboComponent,
    ComboSum,
    combo_sum_quote,
    load_same_player_rho,
)
from sportstradamus.helpers.io import market_file_slug
from sportstradamus.training.component_cells import (
    SHAPE_FIELDS,
    ComponentCell,
    load_component_cell,
    spec_weights,
)
from sportstradamus.training.model_strategy import STRATEGY_IDENTITY_CSV_COLUMNS
from sportstradamus.training.scorecard import COMPONENT_SUM_QUANTILE_COLUMNS

# The incumbent's own predictive no longer describes a component-sum candidate, so
# its per-row family params (and the recal knots warping its PIT) come off the frame.
_INCUMBENT_PREDICTIVE_COLUMNS = frozenset(
    {
        "SN_Loc",
        "SN_Scale",
        "SN_Alpha",
        "R",
        "NB_P",
        "DP_MU",
        "DP_PHI",
        "Alpha",
        "Gate",
        "DenomCol",
        "GlobalMean",
        "PITRecalKnots",
    }
)

# Shared rows a component pair needs before its residual correlation is trusted over
# the kernel's missing-pair convention (rho = 0): the Fisher-z SE at 200 is ~0.07.
_RHO_MIN_PAIRS = 200

# Below this the gates have nothing to say: Gate 4's own KS threshold is 1.358/sqrt(n)
# and Gate 1's bootstrap CI spans the whole effect. Report the cell, do not grade it.
_MIN_GRADED_ROWS = 30

_QUANTILES = tuple(COMPONENT_SUM_QUANTILE_COLUMNS)
_ENDPOINT_COLUMNS = [
    "Blended_EV",
    "P",
    "SUM_CDF",
    "SUM_PMF",
    *COMPONENT_SUM_QUANTILE_COLUMNS.values(),
]


def _constant_term(offset: float, draws: dict[str, np.ndarray]) -> np.ndarray:
    """A deterministic additive term for the kernel's ``post`` hook."""
    return np.full(next(iter(draws.values())).shape, offset)


def _thinned(draws: np.ndarray, count: int) -> np.ndarray:
    """``count`` systematic quantile points out of a sorted draw vector."""
    return draws[np.floor((np.arange(count) + 0.5) * draws.size / count).astype(int)]


def _pooled(sums: ComboSum, incumbent: ComboSum, weight: float) -> ComboSum:
    """Linear pool ``weight * F_sum + (1 - weight) * F_incumbent``, as one draw vector.

    Mixing the samples rather than the CDFs is what keeps the pool invertible: the
    six persisted endpoints all read off a sorted draw vector, and a mixture CDF has
    no closed-form quantile to persist. Each side contributes its own quantiles at
    its own weight, so the pool is exact up to the Sobol resolution and stays
    mean-preserving — the property a log pool would lose.
    """
    take = round(weight * sums.draws_sorted.size)
    draws = np.concatenate(
        (
            _thinned(sums.draws_sorted, take),
            _thinned(incumbent.draws_sorted, sums.draws_sorted.size - take),
        )
    )
    draws.sort()
    mean = weight * sums.mean + (1.0 - weight) * incumbent.mean
    return ComboSum(mean=mean, sd=float(draws.std()), draws_sorted=draws)


def _endpoints(quote: ComboSum, result: float, line: float) -> tuple[float, ...]:
    """The scorecard's six endpoint values for one priced row.

    ``ComboSum.under_prob`` splits push mass like ``get_odds``, which is right for
    the served ``P`` and wrong for a CDF: ``SUM_CDF`` reads the sorted draws directly
    so it is ``F(y)``, with ``SUM_PMF`` its atom at the realized outcome. The
    quantiles take the generalized (floor) inverse so a discrete sum stays on its own
    settlement lattice.
    """
    draws = quote.draws_sorted
    lo = np.searchsorted(draws, result, side="left")
    hi = np.searchsorted(draws, result, side="right")
    return (
        quote.mean,
        1.0 - quote.under_prob(line),
        hi / draws.size,
        (hi - lo) / draws.size,
        *np.quantile(draws, _QUANTILES, method="inverted_cdf"),
    )


def _model_rho(
    cells: dict[str, ComponentCell], graded: pd.MultiIndex, min_pairs: int
) -> tuple[Callable[[str, str], float], dict[str, float]]:
    """NORTA ``rho_Z`` of the components' own predictive residuals.

    Each component's mid-PIT is pushed through ``Phi^-1`` — the same Gaussian layer
    the kernel colours with its Cholesky — and a pair's correlation is the Pearson of
    those scores on the two cells' shared rows *outside* the graded set, so the grade
    stays out-of-sample. A thin pair reads 0, the kernel's own missing-pair convention.
    """
    scores = {m: c.params.loc[~c.params.index.isin(graded), "z"] for m, c in cells.items()}
    pairs: dict[tuple[str, str], float] = {}
    for market_a, market_b in combinations(sorted(scores), 2):
        shared = pd.DataFrame({"a": scores[market_a], "b": scores[market_b]}).dropna()
        rho_z = float(shared.corr().iloc[0, 1]) if len(shared) >= min_pairs else 0.0
        pairs[(market_a, market_b)] = pairs[(market_b, market_a)] = rho_z

    def rho(market_a: str, market_b: str) -> float:
        return 1.0 if market_a == market_b else pairs.get((market_a, market_b), 0.0)

    return rho, {f"{a}|{b}": r for (a, b), r in pairs.items() if a < b}


def _price_rows(
    cells: dict[str, ComponentCell],
    aligned: dict[str, dict[str, np.ndarray]],
    row_specs: list[tuple[tuple[str, float], ...]],
    rho: Callable[[str, str], float],
    post: Callable[[dict[str, np.ndarray]], np.ndarray] | None,
    results: np.ndarray,
    lines: np.ndarray,
    incumbent: dict[str, np.ndarray | str | float] | None = None,
    mixture_weight: float | None = None,
) -> pd.DataFrame:
    """One NORTA quote per row, read out as the scorecard's endpoint columns.

    With ``incumbent`` and ``mixture_weight`` supplied, each row is priced twice —
    the component sum, and the combo cell's own predictive sampled as a
    single-component quote on the same Sobol net — and the two draw vectors are
    linear-pooled. Routing the incumbent through the same kernel rather than through
    its family's closed form is what makes the pool exact: both sides then carry the
    same sampling error, and a pool at ``mixture_weight = 0`` reproduces the
    incumbent's own gate row.
    """
    priced = []
    for i, weights in enumerate(tqdm(row_specs, desc="component sum")):
        quote = combo_sum_quote(
            [
                ComboComponent(
                    sub,
                    weight,
                    float(aligned[sub]["mean"][i]),
                    cells[sub].dist,
                    cells[sub].cv,
                    **_shape_kwargs(aligned[sub], i),
                )
                for sub, weight in weights
            ],
            rho,
            post=post,
        )
        if incumbent is not None:
            marginal = combo_sum_quote(
                [
                    ComboComponent(
                        "incumbent",
                        1.0,
                        float(incumbent["mean"][i]),
                        incumbent["dist"],
                        incumbent["cv"],
                        **_shape_kwargs(incumbent, i),
                    )
                ],
                rho,
            )
            quote = _pooled(quote, marginal, mixture_weight)
        priced.append(_endpoints(quote, results[i], lines[i]))
    return pd.DataFrame(priced, columns=_ENDPOINT_COLUMNS)


def _shape_kwargs(row_params: dict[str, np.ndarray], i: int) -> dict[str, float | None]:
    """A row's optional per-family shape fields, with NaN read back as absent."""
    return {
        field: None if np.isnan(row_params[field][i]) else float(row_params[field][i])
        for field in SHAPE_FIELDS
    }


def _aligned(cell: ComponentCell, keys: pd.MultiIndex) -> dict[str, np.ndarray]:
    """One cell's decoded parameters as row-aligned arrays over the graded keys."""
    reindexed = cell.params.reindex(keys)
    return {field: reindexed[field].to_numpy(dtype=float) for field in cell.params}


def _unpriced_cell(diagnostics: dict, reason: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Empty candidate/baseline pair for a cell that can't be priced, reason attached."""
    return pd.DataFrame(), pd.DataFrame(), diagnostics | {"reason": reason}


def component_sum_frame(
    league: str,
    market: str,
    *,
    test_sets_dir: Path | None = None,
    rho_source: str = "book",
    min_rows: int = _MIN_GRADED_ROWS,
    mixture_weight: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Price ``(league, market)`` as a sum of its component cells' served predictives.

    Returns ``(candidate, baseline, diagnostics)``. The candidate is the combo cell's
    own test rows restricted to the join, with ``Blended_EV`` / ``P`` overwritten by
    the sum and the six ``SUM_*`` endpoints appended; the baseline is the same rows
    otherwise untouched. Both drop the strategy identity — the sum is not the signed
    artifact, and ``ship scorecard`` refuses a frame pair that mixes identity
    conventions — and the candidate additionally drops the incumbent's per-row family
    params. A cell whose components cannot all be read, or whose join is too thin to
    grade, comes back empty with the reason in ``diagnostics``.

    Args:
        league: League code.
        market: Combo market name (``combo_props`` key or a fantasy score).
        test_sets_dir: Override for the package ``data/test_sets`` directory.
        rho_source: ``"book"`` for the shipped same-player residual Spearman,
            ``"model"`` for the components' own out-of-sample residual correlation.
        min_rows: Graded rows below which the cell is reported rather than priced.
        mixture_weight: When given, linear-pool the sum with the incumbent's own
            predictive at this weight on the sum instead of returning the sum alone.
    """
    # style: allow-complexity — one assembly pass; the join, the pricing call and the
    # frame surgery all key off the same restricted row set, and splitting them would
    # only thread that set through forwarders.
    started = perf_counter()
    root = test_sets_dir or Path(str(pkg_resources.files(data) / "test_sets"))
    combo_path = root / f"{market_file_slug(league, market)}.csv"
    combo = pd.read_csv(combo_path)
    specs, offset, provenance = spec_weights(league, market, combo)
    diagnostics: dict = {
        "league": league,
        "market": market,
        "rho_source": rho_source,
        "combo_rows": len(combo),
        "provenance": provenance,
        "offset": offset,
        "reason": None,
    }
    if not specs:
        return _unpriced_cell(diagnostics, "; ".join(provenance))

    wanted = sorted({sub for weights in specs.values() for sub, _ in weights})
    paths = {sub: root / f"{market_file_slug(league, sub)}.csv" for sub in wanted}
    missing = [sub for sub, path in paths.items() if not path.is_file()]
    if missing:
        return _unpriced_cell(diagnostics, "no test set for component(s): " + ", ".join(missing))
    cells = {sub: load_component_cell(league, sub, path) for sub, path in paths.items()}

    keys = pd.MultiIndex.from_frame(combo[["Player", "Date"]])
    row_specs = [specs.get(player) for player in combo["Player"]]
    covered = np.array(
        [
            weights is not None and all(key in cells[sub].params.index for sub, _ in weights)
            for weights, key in zip(row_specs, keys, strict=True)
        ]
    )
    diagnostics["components"] = {
        sub: {
            "dist": cell.dist,
            "cell_rows": len(cell.params),
            "reach": int(cell.params.index.isin(keys).sum()),
        }
        for sub, cell in cells.items()
    }
    diagnostics["graded_rows"] = int(covered.sum())
    if covered.sum() < min_rows:
        reason = f"only {int(covered.sum())} of {len(combo)} rows join every component"
        return _unpriced_cell(diagnostics, reason)

    graded = combo[covered].reset_index(drop=True)
    graded_keys = pd.MultiIndex.from_frame(graded[["Player", "Date"]])
    if rho_source == "book":
        rho, diagnostics["rho"] = load_same_player_rho(league), {}
    else:
        rho, diagnostics["rho"] = _model_rho(cells, graded_keys, _RHO_MIN_PAIRS)
    incumbent = None
    if mixture_weight is not None:
        own = load_component_cell(league, market, combo_path)
        incumbent = {"dist": own.dist, "cv": own.cv, **_aligned(own, graded_keys)}
        diagnostics["mixture_weight"] = mixture_weight
    endpoints = _price_rows(
        cells,
        {sub: _aligned(cell, graded_keys) for sub, cell in cells.items()},
        [weights for weights, keep in zip(row_specs, covered, strict=True) if keep],
        rho,
        partial(_constant_term, offset) if offset else None,
        graded["Result"].to_numpy(dtype=float),
        graded["Line"].to_numpy(dtype=float),
        incumbent,
        mixture_weight,
    )

    identity = [
        column
        for column in graded.columns
        if column in STRATEGY_IDENTITY_CSV_COLUMNS or column.startswith("Structural")
    ]
    baseline = graded.drop(columns=identity)
    candidate = baseline.drop(columns=list(_INCUMBENT_PREDICTIVE_COLUMNS & set(graded.columns)))
    for column in _ENDPOINT_COLUMNS:
        candidate[column] = endpoints[column].to_numpy()
    diagnostics["seconds_per_row"] = (perf_counter() - started) / len(graded)
    return candidate, baseline, diagnostics
