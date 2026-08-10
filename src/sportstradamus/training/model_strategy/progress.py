"""Terminal rendering for the strategy sweep: a bar per running cell, notable corners only, clamped.

A sweep is a multi-day sit, so its screen is a status display, not a transcript: the board CSV is the
durable per-corner record and this module shows only what changes a decision — a new best, a ship, a
failure. :class:`SearchProgress` owns one cell's bar and :class:`BoardProgress` the summary line above
however many are running; ``tqdm(disable=None)`` collapses all of them to nothing when stdout is
redirected, which is what keeps carriage returns out of the overnight driver's logs. A redirected run
instead gets a periodic heartbeat naming the corner currently training, so a stalled subprocess is
still visible.
"""

import contextlib
import queue
import shutil
import textwrap
import threading
import time
from collections.abc import Iterator, Mapping
from statistics import fmean

import click
from tqdm import tqdm

# Redirected output has no width to measure. 120 is where the driver-log audit found readability
# collapsing (1972 of 3064 lines ran past it); an explicit COLUMNS still overrides via shutil.
_LOG_WIDTH: int = 120
# Longest corner-axis value kept intact. The elide is symmetric because truncating from the left
# collides `centered_additive_mean10` with `centered_additive_eb_meanyr_k10` — the two values the
# normalization axis turns on most often.
_MAX_TOKEN_CHARS: int = 17
# Verdict column width, sized for the common `KILL: g4 g5`; a longer one just pushes the label right.
_VERDICT_CHARS: int = 14
# Below three samples the in-run mean is one 1084 s outlier away from nonsense, so the seed estimate
# from prior board rows holds until then.
_ETA_MIN_CORNERS: int = 3
# Heartbeat spacing for a redirected run. A corner's median is 19 s but its p95 is 221 s, so five
# minutes is quiet during normal progress and prompt once something is genuinely stuck.
_HEARTBEAT_S: float = 300.0
# Floor for soft-wrapped text, so a narrow window degrades to slightly-long lines rather than to one
# word per line.
_MIN_WRAP_WIDTH: int = 60
# How often the confirm ticker repaints. The step it watches runs 21-168 min; a faster tick would
# only burn a wakeup to move a bar by a pixel.
_TICK_S: float = 5.0

QUIET, NORMAL, VERBOSE = 0, 1, 2


def line_width() -> int:
    return shutil.get_terminal_size(fallback=(_LOG_WIDTH, 24)).columns


def human_seconds(seconds: float | None) -> str:
    """``47s`` / ``21m`` / ``2h10m``. Coarse on purpose — a sweep ETA showing seconds would imply a
    precision that a 12x spread between a corner's median and its p95 does not support.
    """
    if seconds is None:
        return "?"
    if seconds < 90:
        return f"{seconds:.0f}s"
    minutes = round(seconds / 60)
    return f"{minutes}m" if minutes < 60 else f"{minutes // 60}h{minutes % 60:02d}m"


def _shorten(token: str) -> str:
    if len(token) <= _MAX_TOKEN_CHARS:
        return token
    head = (_MAX_TOKEN_CHARS - 1) // 2
    return f"{token[:head]}…{token[-(_MAX_TOKEN_CHARS - 1 - head) :]}"


def compress_corner(corner: Mapping[str, str], budget: int) -> str:
    """A corner as space-joined values, each middle-elided, clamped to ``budget`` characters.

    The axis names are dropped: the label is positional in registry order and the board CSV holds
    every axis by name, so repeating them cost 100 characters a line and told the reader nothing.
    """
    label = " ".join(_shorten(str(value)) for value in corner.values())
    return label if len(label) <= budget else f"{label[: budget - 1]}…"


def wrapped(text: str, indent: str, hang: str) -> str:
    """``text`` soft-wrapped to the terminal, never splitting an axis value or a slug mid-token."""
    return textwrap.fill(
        text,
        width=max(line_width(), _MIN_WRAP_WIDTH),
        initial_indent=indent,
        subsequent_indent=hang,
        break_long_words=False,
        break_on_hyphens=False,
    )


def echo_above_bar(text: str, *, fg: str | None = None, err: bool = False) -> None:
    """Print one line, clamped to the terminal and lifted above any live bar."""
    clamped = text if len(text) <= line_width() else f"{text[: line_width() - 1]}…"
    with tqdm.external_write_mode():
        click.secho(clamped, fg=fg, err=err)


class BoardProgress:
    """The board's summary row, and the bar rows the cells running under it occupy.

    Rows ``1..jobs`` are handed out by :meth:`slot` so concurrent cells never fight for one line, and
    releasing a slot is what counts a cell finished — a cell that raised still leaves the board's
    count honest. A serial run gets no summary row and its cell sits at row 0, exactly where a
    one-cell-at-a-time sweep has always drawn it.
    """

    def __init__(self, total_cells: int, jobs: int) -> None:
        self.total = total_cells
        self.done = 0
        self.running = 0
        self.started = time.monotonic()
        self._lock = threading.Lock()
        self._free: queue.Queue[int] = queue.Queue()
        for position in range(jobs):
            self._free.put(position if jobs < 2 else position + 1)
        self.bar = tqdm(
            total=total_cells,
            disable=True if jobs < 2 else None,
            position=0,
            leave=False,
            bar_format="  {desc} {n_fmt}/{total_fmt} cells{postfix}",
        )
        self.bar.set_description_str("board")

    def close(self) -> None:
        self.bar.close()

    @contextlib.contextmanager
    def slot(self) -> Iterator[int]:
        position = self._free.get()
        self._repaint(running=1)
        try:
            yield position
        finally:
            self._repaint(running=-1, done=1)
            self._free.put(position)

    def _repaint(self, *, running: int = 0, done: int = 0) -> None:
        """Move the counters and redraw, holding the lock across both.

        The count and the postfix are two halves of one line, so a repaint that reads them a moment
        apart renders three cells done and three running out of a board of five. Painting under the
        lock is also why ``self.bar.n`` is assigned rather than ``update()``-ed: one snapshot, one
        line. Threads only ever take this lock before tqdm's, never the reverse, so the write here
        cannot deadlock against a cell bar's own refresh.
        """
        with self._lock:
            self.running += running
            self.done += done
            self.bar.n = self.done
            self.bar.set_postfix_str(
                f"{self.running} running · {self.total - self.done - self.running} queued"
                f" · run {human_seconds(time.monotonic() - self.started)}",
                refresh=True,
            )


class SearchProgress:
    """One cell's live bar over its trained-corner budget, plus the lines worth interrupting it for.

    ``seed_seconds`` is the mean corner duration measured for this cell on a previous run; it carries
    the ETA until this run has enough samples of its own.
    """

    def __init__(
        self,
        league: str,
        market: str,
        budget: int,
        *,
        verbosity: int = NORMAL,
        seed_seconds: float | None = None,
        reused: int = 0,
        position: int = 0,
    ) -> None:
        self.verbosity = verbosity
        self.seed_seconds = seed_seconds
        self.durations: list[float] = []
        self.ships = 0
        self.best = float("-inf")
        self.last_line_at = time.monotonic()
        self.bar = tqdm(
            total=budget,
            disable=None,
            position=position,
            leave=False,
            bar_format="  {desc} {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt}{postfix}",
        )
        self.bar.set_description_str(f"{league} {market}")
        header = f"{league} {market} · <={budget} corners"
        if reused:
            header += f" · {reused} reusable"
        if seed_seconds is not None:
            header += f" · <={human_seconds(budget * seed_seconds)}"
        echo_above_bar(header)

    def close(self) -> None:
        self.bar.close()

    def _eta(self) -> float | None:
        mean_s = (
            fmean(self.durations) if len(self.durations) >= _ETA_MIN_CORNERS else self.seed_seconds
        )
        return None if mean_s is None else max(self.bar.total - self.bar.n, 0) * mean_s

    def _best_text(self) -> str:
        return "none yet" if self.best == float("-inf") else f"{self.best:+.3f}"

    def starting(self, corner: Mapping[str, str]) -> None:
        """Name the corner about to train, but only on a redirected run and only periodically.

        The bar makes a stall obvious on a terminal; a log needs the in-flight corner named, because
        a corner that hangs is the one case the board CSV cannot record.
        """
        if not self.bar.disable or time.monotonic() - self.last_line_at < _HEARTBEAT_S:
            return
        echo_above_bar(
            f"  … {self.bar.n} corners · best {self._best_text()}"
            f" · <={human_seconds(self._eta())} left"
            f" · now training {compress_corner(corner, line_width() // 2)}"
        )
        self.last_line_at = time.monotonic()

    def scored(
        self,
        corner: Mapping[str, str],
        *,
        verdict: str,
        slack: float,
        elapsed_s: float | None,
        trained: bool,
        failure: str | None = None,
    ) -> None:
        """Record one corner's outcome and print it if it is notable (or if verbosity asks)."""
        if trained:
            self.bar.update(1)
            if verdict == "SHIP":
                self.ships += 1  # must land before _worth_a_line's self.ships == 1 check
            if elapsed_s is not None:
                self.durations.append(elapsed_s)
        improved = slack > self.best
        self.best = max(self.best, slack)
        eta = self._eta()
        self.bar.set_postfix_str(
            f"best {self._best_text()}" + (f" · <={human_seconds(eta)}" if eta else ""),
            refresh=True,
        )
        if self._worth_a_line(verdict, trained, improved, failure):
            self._echo_outcome(corner, verdict, slack, elapsed_s, trained, failure)

    def _worth_a_line(
        self, verdict: str, trained: bool, improved: bool, failure: str | None
    ) -> bool:
        """A failure, a new best, or this cell's first ship.

        A later ship that beat nothing is noise: the bar already carries ``best`` and the cell summary
        lists every shipping corner. Ships are counted rather than inferred from ``improved`` because
        the board's slack omits g6, so a corner can lead on slack and still not ship.
        """
        if self.verbosity >= VERBOSE:
            return True
        if self.verbosity < NORMAL or not trained:
            return False
        return failure is not None or improved or (verdict == "SHIP" and self.ships == 1)

    def _echo_outcome(
        self,
        corner: Mapping[str, str],
        verdict: str,
        slack: float,
        elapsed_s: float | None,
        trained: bool,
        failure: str | None,
    ) -> None:
        """One corner as ``VERDICT  <corner>  slack  elapsed``, the label given the width left over."""
        if failure is not None:
            label, fg, tail = "FAILED", "yellow", failure
        elif not trained:
            label, fg, tail = "cached", None, f"{slack:+.3f}"
        else:
            label = verdict
            fg = "green" if verdict == "SHIP" else "red"
            tail = f"{slack:+.3f}  {human_seconds(elapsed_s)}"
        prefix = f"  {label.ljust(_VERDICT_CHARS)}  "
        echo_above_bar(
            f"{prefix}{compress_corner(corner, line_width() - len(prefix) - len(tail) - 2)}  {tail}",
            fg=fg,
        )
        self.last_line_at = time.monotonic()


@contextlib.contextmanager
def elapsed_ticker(desc: str, expected_s: float | None) -> Iterator[None]:
    """Tick elapsed time against ``expected_s`` while a blocking subprocess runs (terminal only).

    A confirm retrain is 21-168 minutes of a captured subprocess with nothing on screen. A daemon
    thread repainting a bar is all this needs: the work happens in ``subprocess.run``, which keeps
    its own timeout and kill semantics rather than having them reimplemented around a poll loop.
    """
    bar = tqdm(
        total=expected_s,
        disable=None,
        leave=False,
        bar_format="  {desc} {percentage:3.0f}%|{bar:24}| {postfix}"
        if expected_s
        else "  {desc} {postfix}",
    )
    if bar.disable:
        bar.close()
        yield
        return
    bar.set_description_str(desc)
    stop = threading.Event()
    started = time.monotonic()

    def tick() -> None:
        while not stop.wait(_TICK_S):
            elapsed = time.monotonic() - started
            bar.n = min(elapsed, expected_s) if expected_s else 0
            bar.set_postfix_str(
                human_seconds(elapsed)
                + (f" / ~{human_seconds(expected_s)}" if expected_s else " elapsed"),
                refresh=True,
            )

    thread = threading.Thread(target=tick, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=_TICK_S)
        bar.close()
