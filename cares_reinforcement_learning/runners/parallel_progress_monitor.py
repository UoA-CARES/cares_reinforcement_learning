"""
Progress monitoring for parallel execution.

Interactive terminals receive a live Rich progress display. Non-interactive
outputs, such as Docker logs or redirected files, receive periodic line-based
progress messages.
"""

import concurrent.futures
import logging
import time
from dataclasses import dataclass
from multiprocessing.queues import Queue
from queue import Empty
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)


@dataclass(frozen=True)
class ProgressUpdate:
    """Progress reported by a single worker."""

    seed: int
    step: int
    total: int
    status: str

    @classmethod
    def from_message(cls, message: dict[str, Any]) -> "ProgressUpdate":
        """Create a progress update from a worker queue message."""
        return cls(
            seed=int(message["seed"]),
            step=max(0, int(message.get("step", 0))),
            total=max(1, int(message.get("total", 1))),
            status=str(message.get("status", "")),
        )

    @property
    def percentage(self) -> int:
        """Return progress as an integer percentage between 0 and 100."""
        return max(0, min(100, int(self.step / self.total * 100)))

    @property
    def is_complete(self) -> bool:
        """Return whether the worker reported completion."""
        return self.status == "done"


class ParallelProgressMonitor:
    """
    Monitor progress messages and futures for parallel worker executions.

    Rich progress bars are displayed when the output console is interactive.
    Otherwise, progress is written as ordinary log messages at fixed percentage
    intervals.
    """

    def __init__(
        self,
        progress_queue: Queue,
        futures: list[concurrent.futures.Future[Any]],
        logger: logging.Logger,
        log_interval_percent: int = 5,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        """
        Initialise the progress monitor.

        Args:
            progress_queue: Queue containing worker progress messages.
            futures: Futures representing the parallel worker executions.
            logger: Logger used for non-interactive progress output.
            log_interval_percent: Minimum percentage change between log messages.
            poll_interval_seconds: Delay when no new activity is detected.
        """
        if log_interval_percent < 1:
            raise ValueError("log_interval_percent must be at least 1")

        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be greater than 0")

        self.progress_queue = progress_queue
        self.futures = futures
        self.logger = logger
        self.log_interval_percent = log_interval_percent
        self.poll_interval_seconds = poll_interval_seconds

        self.console = Console()
        self.use_live_display = self.console.is_interactive

        self.progress = Progress(
            TextColumn("[bold blue]{task.fields[seed]}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("{task.fields[status]}"),
            TextColumn("[cyan]{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=self.console,
            disable=not self.use_live_display,
        )

        self.task_ids: dict[int, TaskID] = {}
        self.latest_updates: dict[int, ProgressUpdate] = {}
        self.completed_seeds: set[int] = set()
        self.completed_futures: set[int] = set()
        self.last_logged_percentages: dict[int, int] = {}

    def run(self) -> None:
        """Monitor progress until all worker futures have completed."""
        with self.progress:
            while not self._all_futures_complete:
                received_progress = self._drain_progress_queue()
                completed_future = self._check_futures()

                if not received_progress and not completed_future:
                    time.sleep(self.poll_interval_seconds)

            # A worker may place its final update immediately before its future
            # completes, so drain the queue once more before closing.
            self._drain_with_grace_period()

        if not self.use_live_display:
            self._log_latest_states()

    @property
    def _all_futures_complete(self) -> bool:
        """Return whether all submitted futures have completed."""
        return len(self.completed_futures) == len(self.futures)

    def _drain_progress_queue(self) -> bool:
        """
        Process all currently available progress messages.

        Returns:
            True if at least one message was processed.
        """
        received_message = False

        while True:
            try:
                message = self.progress_queue.get_nowait()
            except Empty:
                return received_message

            received_message = True
            self._handle_update(ProgressUpdate.from_message(message))

    def _handle_update(self, update: ProgressUpdate) -> None:
        """Process one worker progress update."""
        self.latest_updates[update.seed] = update

        if self.use_live_display:
            self._update_live_display(update)
        else:
            self._log_update_if_due(update)

        if update.is_complete:
            self._mark_seed_complete(update.seed)

    def _update_live_display(self, update: ProgressUpdate) -> None:
        """Create or update the Rich task for a worker seed."""
        task_id = self.task_ids.get(update.seed)

        if task_id is None:
            task_id = self.progress.add_task(
                f"Seed {update.seed}",
                total=update.total,
                seed=f"Seed {update.seed}",
                status=update.status,
            )
            self.task_ids[update.seed] = task_id

        self.progress.update(
            task_id,
            completed=update.step,
            total=update.total,
            status=update.status,
        )

    def _log_update_if_due(self, update: ProgressUpdate) -> None:
        """Log a progress update when the configured interval is reached."""
        previous = self.last_logged_percentages.get(update.seed)

        if (
            previous is not None
            and not update.is_complete
            and update.percentage < previous + self.log_interval_percent
        ):
            return

        self._write_log(update)
        self.last_logged_percentages[update.seed] = update.percentage

    def _mark_seed_complete(self, seed: int) -> None:
        """Record seed completion and show an interactive completion message."""
        if seed in self.completed_seeds:
            return

        self.completed_seeds.add(seed)

        if self.use_live_display:
            self.progress.console.log(f"[green]Seed {seed} completed!")

    def _check_futures(self) -> bool:
        """
        Record completed futures and propagate worker exceptions.

        Returns:
            True if at least one new future completed.
        """
        found_completed_future = False

        for index, future in enumerate(self.futures):
            if index in self.completed_futures or not future.done():
                continue

            self.completed_futures.add(index)
            found_completed_future = True

            exception = future.exception()
            if exception is not None:
                self._cancel_pending_futures()
                raise exception

        return found_completed_future

    def _cancel_pending_futures(self) -> None:
        """Cancel futures that have not yet completed."""
        for future in self.futures:
            if not future.done():
                future.cancel()

    def _log_latest_states(self) -> None:
        """Ensure the latest known state for every seed appears in the logs."""
        for seed in sorted(self.latest_updates):
            update = self.latest_updates[seed]

            if self.last_logged_percentages.get(seed) == update.percentage:
                continue

            self._write_log(update, final=True)

    def _write_log(
        self,
        update: ProgressUpdate,
        final: bool = False,
    ) -> None:
        """Write a line-based progress message."""
        label = "final progress" if final else "progress"

        self.logger.info(
            "Seed %s %s: %d/%d (%d%%) - %s",
            update.seed,
            label,
            update.step,
            update.total,
            update.percentage,
            update.status,
        )

    def _drain_with_grace_period(
        self,
        timeout_seconds: float = 2.0,
        quiet_period_seconds: float = 0.2,
    ) -> None:
        """
        Drain progress messages until the queue remains quiet.

        Stops after no messages arrive for ``quiet_period_seconds`` or once the
        overall timeout is reached.
        """
        deadline = time.monotonic() + timeout_seconds
        quiet_since = time.monotonic()

        while time.monotonic() < deadline:
            if self._drain_progress_queue():
                quiet_since = time.monotonic()
            elif time.monotonic() - quiet_since >= quiet_period_seconds:
                return
            else:
                time.sleep(self.poll_interval_seconds)
