"""Read-only integrity audit for cached and quarantined training matrices."""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd

from sportstradamus.helpers.training_quotes import PROVENANCE_COLUMNS, archive_ev_is_runaway
from sportstradamus.stats.nhl_position_policy import allowed_nhl_position_codes

_BOOK_COLUMNS = ("Line", "Odds", "EV", "Archived")
_IDENTITY_COLUMNS = ("Player", "Date")
# The three a quote cannot be interpreted without; the other two are descriptive.
_REQUIRED_PROVENANCE_COLUMNS = ("QuoteSource", "QuoteAuthenticity", "QuoteBookCount")
# Shared by audit_matrix (writer) and main's legacy census (reader) so a reword
# of one can't silently desync it from the count it's supposed to drive.
_MISSING_PROVENANCE_VIOLATION = "missing quote provenance"


def incomplete_provenance_rows(frame: pd.DataFrame) -> int:
    """Count rows whose explicit provenance block is present but not fully populated.

    Zero for a legacy matrix that carries no block at all — that one has a defined fallback.
    A nonzero count means the two eras were concatenated, which no reader can interpret.
    """
    if not all(column in frame for column in _REQUIRED_PROVENANCE_COLUMNS):
        return 0
    return int(frame.loc[:, list(_REQUIRED_PROVENANCE_COLUMNS)].isna().any(axis=1).sum())


def audit_matrix(path: Path) -> dict[str, object]:
    """Report identity, numeric, lattice, position, and quote provenance integrity."""
    # style: allow-complexity — a flat sequence of independent integrity checks, each
    # appending its own violation; splitting it would only scatter one report's assembly.
    frame = pd.read_parquet(path)
    report: dict[str, object] = {"path": str(path), "rows": len(frame), "violations": []}
    violations = report["violations"]
    if not all(column in frame for column in (*_IDENTITY_COLUMNS, "Result", *_BOOK_COLUMNS)):
        violations.append("missing required matrix columns")
        return report

    report["duplicate_identity_rows"] = int(frame.duplicated(list(_IDENTITY_COLUMNS)).sum())
    if report["duplicate_identity_rows"]:
        violations.append("duplicate Player/Date identity")

    numeric = frame.loc[:, ["Result", "Line", "Odds", "EV"]].apply(pd.to_numeric, errors="coerce")
    report["nonfinite_rows"] = int((~np.isfinite(numeric)).any(axis=1).sum())
    report["odds_out_of_range_rows"] = int(((numeric["Odds"] < 0) | (numeric["Odds"] > 1)).sum())
    archived = frame["Archived"].fillna(False).astype(bool)
    blown_ev = pd.Series(
        [
            archive_ev_is_runaway(ev, line)
            for ev, line in zip(numeric["EV"], numeric["Line"], strict=True)
        ],
        index=frame.index,
    )
    report["blown_ev_rows"] = int(blown_ev.sum())
    invalid_archived = archived & (
        (numeric["Line"] <= 0)
        | (numeric["EV"] <= 0)
        | (numeric["Odds"] < 0)
        | (numeric["Odds"] > 1)
        | (~np.isfinite(numeric)).any(axis=1)
        | blown_ev
    )
    report["invalid_archived_rows"] = int(invalid_archived.sum())
    for key, message in (
        ("nonfinite_rows", "nonfinite core value"),
        ("odds_out_of_range_rows", "Odds outside [0, 1]"),
        ("invalid_archived_rows", "invalid archived quote"),
        ("blown_ev_rows", "runaway EV value"),
    ):
        if report[key]:
            violations.append(message)

    unique_result = np.sort(numeric["Result"].dropna().unique())
    positive_steps = np.diff(unique_result)
    positive_steps = positive_steps[positive_steps > 0]
    report["target_lattice"] = {
        "unique_values": len(unique_result),
        "minimum_step": float(positive_steps.min()) if len(positive_steps) else None,
        "integer": bool(np.equal(np.mod(unique_result, 1), 0).all()),
    }
    report["positions"] = (
        frame["Player position"].value_counts(dropna=False).sort_index().to_dict()
        if "Player position" in frame
        else {}
    )
    try:
        league, market_slug = path.stem.split("_", 1)
    except ValueError:
        league, market_slug = "", ""
    if league == "NHL" and "Player position" in frame:
        market = market_slug.replace("-", " ")
        try:
            allowed_codes = allowed_nhl_position_codes(market)
        except ValueError:
            report["invalid_nhl_position_rows"] = len(frame)
            violations.append("missing NHL market position policy")
        else:
            position_codes = pd.to_numeric(frame["Player position"], errors="coerce")
            invalid_positions = ~position_codes.isin(allowed_codes)
            report["invalid_nhl_position_rows"] = int(invalid_positions.sum())
            if report["invalid_nhl_position_rows"]:
                violations.append("ineligible NHL market position")

    synthetic = (
        frame["Odds_synthetic"].fillna(False).astype(bool)
        if "Odds_synthetic" in frame
        else pd.Series(False, index=frame.index)
    )
    missing_provenance = [column for column in PROVENANCE_COLUMNS if column not in frame]
    report["missing_provenance_columns"] = missing_provenance
    report["quote_sources"] = (
        frame["QuoteSource"].value_counts(dropna=False).to_dict() if "QuoteSource" in frame else {}
    )
    if missing_provenance:
        authentic = archived & ~synthetic
        report["quote_counts"] = {
            "authentic": int(authentic.sum()),
            "synthetic": int(synthetic.sum()),
            "bookless": int((~archived & ~synthetic).sum()),
            "archive_coverage": float(archived.mean()) if len(frame) else 0.0,
        }
        violations.append(_MISSING_PROVENANCE_VIOLATION)
    else:
        authenticity = frame["QuoteAuthenticity"]
        authentic = authenticity.eq("authentic")
        report["quote_counts"] = {
            "authentic": int(authentic.sum()),
            "synthetic": int((authenticity == "synthetic").sum()),
            "derived": int((authenticity == "derived").sum()),
            "bookless": int((frame["QuoteBookCount"] == 0).sum()),
            "archive_coverage": float(authentic.mean()) if len(frame) else 0.0,
        }
        if incomplete_provenance_rows(frame):
            violations.append("null required quote provenance")
        if not authenticity.isin(("authentic", "derived", "synthetic")).all():
            violations.append("invalid quote authenticity")
        reasons = frame["QuoteSyntheticReason"]
        if reasons[authentic].notna().any() or reasons[~authentic].isna().any():
            violations.append("inconsistent synthetic reason")
        observed = frame["QuoteObservedAt"]
        if observed[authentic].isna().any():
            violations.append("authentic quote missing observation time")
        book_count = pd.to_numeric(frame["QuoteBookCount"], errors="coerce")
        if (
            book_count.isna().any()
            or (book_count < 0).any()
            or (authentic & (book_count < 1)).any()
        ):
            violations.append("invalid quote book count")
        if not archived.equals(authentic.astype(bool)):
            violations.append("Archived disagrees with explicit authenticity")
        if not synthetic.equals((~authentic).astype(bool)):
            violations.append("Odds_synthetic disagrees with explicit authenticity")
    authentic_count = report["quote_counts"]["authentic"]
    if authentic_count == 0:
        report["quote_classification"] = "bookless"
    elif authentic_count == len(frame):
        report["quote_classification"] = "authentic-supported"
    else:
        report["quote_classification"] = "partial"
    return report


def audit_paths(paths: list[Path]) -> tuple[list[dict[str, object]], int]:
    """Audit paths in stable order and return reports plus violating-matrix count."""
    reports = [audit_matrix(path) for path in sorted(paths)]
    return reports, sum(bool(report["violations"]) for report in reports)


@click.command()
@click.argument("paths", nargs=-1, type=click.Path(path_type=Path, exists=True))
@click.option(
    "--root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="Audit every parquet directly below this directory.",
)
def main(paths: tuple[Path, ...], root: Path | None) -> None:
    """Print JSON reports and exit nonzero when any matrix violates a contract."""
    selected = list(paths)
    if root is not None:
        selected.extend(root.glob("*.parquet"))
    if not selected:
        raise click.UsageError("provide at least one matrix path or --root")
    reports, failures = audit_paths(selected)
    click.echo(json.dumps(reports, indent=2, default=str))
    # A legacy matrix is not yet broken, but it self-poisons on its league's next
    # in-season append, so the census is what says how much repair work is queued.
    legacy = [
        report["path"]
        for report in reports
        if _MISSING_PROVENANCE_VIOLATION in report["violations"]
    ]
    click.echo(
        f"{len(reports)} audited, {len(legacy)} legacy (no provenance block), {failures} failing"
    )
    for path in legacy:
        click.echo(f"  legacy: {path}")
    if failures:
        raise click.ClickException(f"{failures} matrix file(s) failed integrity audit")


if __name__ == "__main__":
    main()
