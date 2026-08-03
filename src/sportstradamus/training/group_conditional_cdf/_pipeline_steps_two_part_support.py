"""Nested Player-grouped support audit for the two-part role×position candidate.

Checked across the full/outer/inner CV partitions before the candidate is ever
allowed to inspect the independent test outcomes.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold

from sportstradamus.training.group_conditional_cdf._config import discover_codes
from sportstradamus.training.group_conditional_cdf._contracts import (
    ROLE_VALUES as TWO_PART_ROLE_VALUES,
)

# Frozen two-part support protocol. These floors were checked across the
# full/outer/inner Player-grouped fit partitions before the candidate was ever
# allowed to inspect the independent test outcomes.
_TWO_PART_SUPPORT_FLOORS: dict[str, int] = {
    "fit_rows": 1000,
    "fit_players": 150,
    "positive_rows": 100,
    "positive_players": 20,
    "nonpositive_rows": 25,
    "nonpositive_players": 20,
    "rb_rows": 250,
    "rb_players": 40,
    "rb_positive_rows": 150,
    "rb_nonpositive_rows": 50,
    "rb_class_players": 20,
    "temperature_players": 50,
    "temperature_class_rows": 50,
    "authentic_rows": 1000,
    "authentic_players": 100,
    "authentic_hold_rows": 250,
    "authentic_hold_players": 50,
    "authentic_hold_class_rows": 100,
}


def _two_part_group_partitions(players: np.ndarray, all_rows: np.ndarray):
    """Build the nested Player-grouped CV partitions the two-part audit walks.

    Returns ``(outer, top_partitions, fit_partitions, hold_partitions,
    required_fit)`` — the outer folds, the full/outer-train tops, and every
    inner fit/hold split derived from them.
    """

    def _group_splits(index: np.ndarray):
        local_players = players[index]
        splitter = GroupKFold(n_splits=5)
        for fit_local, hold_local in splitter.split(np.zeros(len(index)), groups=local_players):
            yield index[fit_local], index[hold_local]

    outer = list(_group_splits(all_rows))
    top_partitions = [("full", all_rows)] + [
        (f"outer_{fold}_train", fit) for fold, (fit, _) in enumerate(outer, start=1)
    ]
    fit_partitions = list(top_partitions)
    hold_partitions = [
        (f"outer_{fold}_hold", hold) for fold, (_, hold) in enumerate(outer, start=1)
    ]
    for top_name, top_rows in top_partitions:
        for fold, (fit, hold) in enumerate(_group_splits(top_rows), start=1):
            fit_partitions.append((f"{top_name}_inner_{fold}_train", fit))
            hold_partitions.append((f"{top_name}_inner_{fold}_hold", hold))

    required_fit = [
        {
            "partition": name,
            "rows": len(index),
            "players": len(np.unique(players[index])),
        }
        for name, index in fit_partitions
    ]
    return outer, top_partitions, fit_partitions, hold_partitions, required_fit


def _positive_group_masks(
    roles: np.ndarray, positions: np.ndarray, grouping: str
) -> list[tuple[str, np.ndarray]]:
    """Group key + membership mask per positive group at the chosen granularity.

    ``role_by_position`` crosses each role with the discovered position codes;
    ``role_only`` collapses the position axis so each role is one dense group.
    """
    groups: list[tuple[str, np.ndarray]] = []
    for role in TWO_PART_ROLE_VALUES:
        role_mask = roles == role
        if grouping == "role_only":
            groups.append((role, role_mask))
        else:
            groups.extend(
                (f"{role}_pos{position}", role_mask & (positions == position))
                for position in discover_codes(positions)
            )
    return groups


def _two_part_positive_support(
    result: np.ndarray,
    players: np.ndarray,
    roles: np.ndarray,
    positions: np.ndarray,
    fit_partitions: list,
    hold_partitions: list,
    grouping: str = "role_by_position",
) -> tuple[dict, dict]:
    """Minimum positive-support fit/hold cell for every positive group."""
    positive_support: dict[str, dict[str, int | str]] = {}
    positive_hold_support: dict[str, dict[str, int | str]] = {}
    for group, group_mask in _positive_group_masks(roles, positions, grouping):
        records = []
        for name, index in fit_partitions:
            mask = group_mask[index] & (result[index] > 0.0)
            records.append(
                {
                    "partition": name,
                    "rows": int(mask.sum()),
                    "players": len(np.unique(players[index][mask])),
                }
            )
        row_min = min(records, key=lambda item: (item["rows"], item["players"]))
        player_min = min(records, key=lambda item: (item["players"], item["rows"]))
        positive_support[group] = {
            "minimum_rows": row_min["rows"],
            "minimum_rows_partition": row_min["partition"],
            "minimum_players": player_min["players"],
            "minimum_players_partition": player_min["partition"],
        }
        hold_records = [
            {"partition": name, "rows": int((group_mask[index] & (result[index] > 0.0)).sum())}
            for name, index in hold_partitions
        ]
        positive_hold_support[group] = min(hold_records, key=lambda item: item["rows"])
    return positive_support, positive_hold_support


def _positive_map_support_ok(positive_support: dict, floor: dict) -> bool:
    """Whether every positive group clears the per-fold row/player floor."""
    return all(
        values["minimum_rows"] >= floor["positive_rows"]
        and values["minimum_players"] >= floor["positive_players"]
        for values in positive_support.values()
    )


def _positive_holds_nonempty(positive_hold_support: dict) -> bool:
    """Whether every positive group holds at least one row in each hold fold."""
    return all(values["rows"] > 0 for values in positive_hold_support.values())


def _two_part_nonpositive_support(
    result: np.ndarray,
    players: np.ndarray,
    roles: np.ndarray,
    positions: np.ndarray,
    fit_partitions: list,
    residual_positions,
) -> tuple[dict, dict]:
    """Non-positive per-role support plus the residual boundary-cell minimums."""
    nonpositive_support: dict[str, dict[str, int | str]] = {}
    for role in TWO_PART_ROLE_VALUES:
        records = []
        for name, index in fit_partitions:
            mask = (roles[index] == role) & (result[index] <= 0.0)
            records.append(
                {
                    "partition": name,
                    "rows": int(mask.sum()),
                    "players": len(np.unique(players[index][mask])),
                }
            )
        row_min = min(records, key=lambda item: (item["rows"], item["players"]))
        player_min = min(records, key=lambda item: (item["players"], item["rows"]))
        nonpositive_support[role] = {
            "minimum_rows": row_min["rows"],
            "minimum_rows_partition": row_min["partition"],
            "minimum_players": player_min["players"],
            "minimum_players_partition": player_min["partition"],
        }

    rb_records = []
    for name, index in fit_partitions:
        rb = np.isin(positions[index], residual_positions)
        positive = rb & (result[index] > 0.0)
        nonpositive = rb & (result[index] <= 0.0)
        rb_records.append(
            {
                "partition": name,
                "rows": int(rb.sum()),
                "players": len(np.unique(players[index][rb])),
                "positive_rows": int(positive.sum()),
                "positive_players": len(np.unique(players[index][positive])),
                "nonpositive_rows": int(nonpositive.sum()),
                "nonpositive_players": len(np.unique(players[index][nonpositive])),
            }
        )
    rb_minimum = {
        field: min(record[field] for record in rb_records)
        for field in (
            "rows",
            "players",
            "positive_rows",
            "positive_players",
            "nonpositive_rows",
            "nonpositive_players",
        )
    }
    return nonpositive_support, rb_minimum


def _two_part_temperature_support(
    outcome: np.ndarray,
    players: np.ndarray,
    authentic: np.ndarray,
    top_partitions: list,
    outer: list,
) -> tuple[list, list]:
    """Per-top class balance and per-outer-fold authentic-hold class balance."""
    temperature_support = []
    for name, index in top_partitions:
        classes = np.bincount(outcome[index].astype(int), minlength=2)
        temperature_support.append(
            {
                "partition": name,
                "rows": len(index),
                "players": len(np.unique(players[index])),
                "class_0_rows": int(classes[0]),
                "class_1_rows": int(classes[1]),
            }
        )
    authentic_holds = []
    for fold, (_, hold) in enumerate(outer, start=1):
        use = hold[authentic[hold]]
        classes = np.bincount(outcome[use].astype(int), minlength=2)
        authentic_holds.append(
            {
                "fold": fold,
                "rows": len(use),
                "players": len(np.unique(players[use])),
                "class_0_rows": int(classes[0]),
                "class_1_rows": int(classes[1]),
            }
        )
    return temperature_support, authentic_holds


def _two_part_rb_boundary_ok(rb_minimum: dict, floor: dict) -> bool:
    """Whether the RB boundary cell clears every RB support floor."""
    return (
        rb_minimum["rows"] >= floor["rb_rows"]
        and rb_minimum["players"] >= floor["rb_players"]
        and rb_minimum["positive_rows"] >= floor["rb_positive_rows"]
        and rb_minimum["nonpositive_rows"] >= floor["rb_nonpositive_rows"]
        and rb_minimum["positive_players"] >= floor["rb_class_players"]
        and rb_minimum["nonpositive_players"] >= floor["rb_class_players"]
    )


def _two_part_support_guards(
    players: np.ndarray,
    roles: np.ndarray,
    authentic: np.ndarray,
    required_fit: list,
    positive_support: dict,
    positive_hold_support: dict,
    nonpositive_support: dict,
    rb_minimum: dict,
    temperature_support: list,
    authentic_holds: list,
    residual_positions,
) -> dict:
    """Evaluate every two-part support floor against the computed partitions."""
    floor = _TWO_PART_SUPPORT_FLOORS
    return {
        "roles_exact": set(np.unique(roles)) == set(TWO_PART_ROLE_VALUES),
        "required_fit_size": all(
            values["rows"] >= floor["fit_rows"] and values["players"] >= floor["fit_players"]
            for values in required_fit
        ),
        "positive_map_support": _positive_map_support_ok(positive_support, floor),
        "positive_holds_nonempty": _positive_holds_nonempty(positive_hold_support),
        "nonpositive_map_support": all(
            values["minimum_rows"] >= floor["nonpositive_rows"]
            and values["minimum_players"] >= floor["nonpositive_players"]
            for values in nonpositive_support.values()
        ),
        "rb_boundary_support": not residual_positions
        or _two_part_rb_boundary_ok(rb_minimum, floor),
        "temperature_support": all(
            values["players"] >= floor["temperature_players"]
            and min(values["class_0_rows"], values["class_1_rows"])
            >= floor["temperature_class_rows"]
            for values in temperature_support
        ),
        "authentic_overall_support": (
            int(authentic.sum()) >= floor["authentic_rows"]
            and len(np.unique(players[authentic])) >= floor["authentic_players"]
        ),
        "authentic_hold_support": all(
            values["rows"] >= floor["authentic_hold_rows"]
            and values["players"] >= floor["authentic_hold_players"]
            and min(values["class_0_rows"], values["class_1_rows"])
            >= floor["authentic_hold_class_rows"]
            for values in authentic_holds
        ),
    }


def _two_part_nested_support_audit(
    result: np.ndarray,
    outcome: np.ndarray,
    authentic: np.ndarray,
    players: np.ndarray,
    roles: np.ndarray,
    positions: np.ndarray,
    residual_positions,
    pinned_grouping: str | None = None,
) -> dict:
    """Audit every full/outer/inner Player-grouped partition used by two-part v3.

    ``pinned_grouping`` forces one granularity instead of demoting to ``role_only`` when
    ``role_by_position`` is unsupported: a cross-fit run pins one grouping across every
    calibration-fit partition up front, so a fold may not quietly change it (and fails its
    support guards instead).
    """
    n_rows = len(result)
    all_rows = np.arange(n_rows)

    outer, top_partitions, fit_partitions, hold_partitions, required_fit = (
        _two_part_group_partitions(players, all_rows)
    )
    grouping = pinned_grouping or "role_by_position"
    positive_support, positive_hold_support = _two_part_positive_support(
        result, players, roles, positions, fit_partitions, hold_partitions, grouping
    )
    if pinned_grouping is None and not (
        _positive_map_support_ok(positive_support, _TWO_PART_SUPPORT_FLOORS)
        and _positive_holds_nonempty(positive_hold_support)
    ):
        grouping = "role_only"
        positive_support, positive_hold_support = _two_part_positive_support(
            result, players, roles, positions, fit_partitions, hold_partitions, grouping
        )
    nonpositive_support, rb_minimum = _two_part_nonpositive_support(
        result, players, roles, positions, fit_partitions, residual_positions
    )
    temperature_support, authentic_holds = _two_part_temperature_support(
        outcome, players, authentic, top_partitions, outer
    )
    guards = _two_part_support_guards(
        players,
        roles,
        authentic,
        required_fit,
        positive_support,
        positive_hold_support,
        nonpositive_support,
        rb_minimum,
        temperature_support,
        authentic_holds,
        residual_positions,
    )
    audit = {
        "guards": guards,
        "validation_rows": n_rows,
        "validation_players": len(np.unique(players)),
        "required_fit_minimum": {
            "rows": min(values["rows"] for values in required_fit),
            "players": min(values["players"] for values in required_fit),
        },
        "positive_map_minimum_support": positive_support,
        "positive_hold_minimum_support": positive_hold_support,
        "nonpositive_map_minimum_support": nonpositive_support,
        "rb_boundary_minimum_support": rb_minimum,
        "temperature_support": temperature_support,
        "authentic": {
            "rows": int(authentic.sum()),
            "players": len(np.unique(players[authentic])),
            "outer_holds": authentic_holds,
            "fallback_rows": int((~authentic).sum()),
        },
    }
    if grouping != "role_by_position":
        # Absent tag ⇒ role_by_position, so a position-granular cell's persisted
        # audit stays byte-identical; only a role-only fallback records the switch.
        audit["grouping"] = grouping
    failed = [name for name, passed in guards.items() if not passed]
    if failed:
        raise ValueError("two-part candidate failed support guard(s): " + ", ".join(failed))
    return audit
