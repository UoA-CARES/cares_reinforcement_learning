"""
Progress monitoring for parallel execution.

Provides an interactive Rich progress display when running in a terminal and
periodic line-based progress logging when output is redirected to a file,
Docker log stream, or another non-interactive destination.
"""

import concurrent.futures
import logging
import sys
import time
from dataclasses import dataclass
from multiprocessing.queues import Queue
from queue import Empty
from typing import Any

from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)


@dataclass
class SeedProgress:
    """Latest reported progress state for a single execution seed."""

    step: int
    total: int
    status: str

    @classmethod
    def from_message(cls, message: dict[str, Any]) -> "SeedProgress":
        """Create a progress state from a worker queue message."""
        return cls(
            step=max(0, int(message.get("step", 0))),
            total=max(1, int(message.get("total", 1))),
            status=str(message.get("status", "")),
        )

    @property
    def percentage(self) -> int:
        """Return progress as an integer percentage bounded to 0–100."""
        percentage = int((self.step / self.total) * 100)
        return min(100, max(0, percentage))

    @property
    def is_done(self) -> bool:
        """Return whether the worker reported completion."""
        return self.status == "done"


class ParallelProgressMonitor:
    """
    Monitor worker progress and completion during parallel execution.

    Interactive terminals receive a live Rich progress display. Non-interactive
    outputs receive periodic ordinary log messages suitable for Docker logs and
    redirected log files.
    """

    def __init__(
        self,
        progress_queue: Queue,
        futures: list[concurrent.futures.Future[Any]],
        logger: logging.Logger,
        log_interval_percent: int = 1,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        """
        Initialise the parallel progress monitor.

        Args:
            progress_queue: Queue containing progress messages from workers.
            futures: Futures representing the parallel worker executions.
            logger: Logger used for non-interactive progress output.
            log_interval_percent: Percentage change required between log entries.
            poll_interval_seconds: Delay when no progress or future changes occur.
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

        self.interactive = sys.stdout.isatty()

        self.task_ids: dict[int, TaskID] = {}
        self.seed_states: dict[int, SeedProgress] = {}
        self.completed_seeds: set[int] = set()
        self.completed_futures: set[int] = set()
        self.last_logged_percentages: dict[int, int] = {}

        self.progress = self._create_progress_display()

    def run(self) -> None:
        """
        Monitor progress until all futures finish.

        Worker exceptions are propagated to the caller after pending futures
        have been cancelled where possible.
        """
        with self.progress:
            while not self._all_futures_completed:
                consumed_message = self._consume_progress_messages()
                completed_future = self._process_completed_futures()

                if not consumed_message and not completed_future:
                    time.sleep(self.poll_interval_seconds)

            # Futures may complete immediately after placing their final queue
            # messages, so drain the queue once more before closing the display.
            self._consume_progress_messages()

        if not self.interactive:
            self._log_final_progress_states()

    @property
    def _all_futures_completed(self) -> bool:
        """Return whether every submitted future has completed."""
        return len(self.completed_futures) == len(self.futures)

    def _create_progress_display(self) -> Progress:
        """Create the Rich progress display."""
        return Progress(
            TextColumn("[bold blue]{task.fields[seed]}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("{task.fields[status]}"),
            TextColumn("[cyan]{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            disable=not self.interactive,
        )

    def _consume_progress_messages(self) -> bool:
        """
        Consume all currently available worker progress messages.

        Returns:
            True when at least one message was consumed.
        """
        consumed_message = False

        while True:
            try:
                message = self.progress_queue.get_nowait()
            except Empty:
                break

            consumed_message = True
            self._handle_progress_message(message)

        return consumed_message

    def _handle_progress_message(self, message: dict[str, Any]) -> None:
        """Process one worker progress message."""
        seed = int(message["seed"])
        state = SeedProgress.from_message(message)

        self.seed_states[seed] = state

        if self.interactive:
            self._update_progress_display(seed, state)
        else:
            self._log_progress_update(seed, state)

        if state.is_done:
            self._mark_seed_completed(seed)

    def _update_progress_display(
        self,
        seed: int,
        state: SeedProgress,
    ) -> None:
        """Create or update the Rich progress task for a seed."""
        task_id = self.task_ids.get(seed)

        if task_id is None:
            task_id = self.progress.add_task(
                f"Seed {seed}",
                total=state.total,
                seed=f"Seed {seed}",
                status=state.status,
            )
            self.task_ids[seed] = task_id

        self.progress.update(
            task_id,
            completed=state.step,
            total=state.total,
            status=state.status,
        )

    def _log_progress_update(
        self,
        seed: int,
        state: SeedProgress,
    ) -> None:
        """Emit a periodic line-based progress log entry."""
        previous_percentage = self.last_logged_percentages.get(seed)

        should_log = (
            previous_percentage is None
            or state.percentage >= (previous_percentage + self.log_interval_percent)
            or state.is_done
        )

        if not should_log:
            return

        self._write_progress_log(seed, state)
        self.last_logged_percentages[seed] = state.percentage

    def _mark_seed_completed(self, seed: int) -> None:
        """Record and report completion for a seed once."""
        if seed in self.completed_seeds:
            return

        self.completed_seeds.add(seed)

        if self.interactive:
            self.progress.console.log(f"[green]Seed {seed} completed!")

    def _process_completed_futures(self) -> bool:
        """
        Record newly completed futures and propagate worker exceptions.

        Returns:
            True when at least one newly completed future was found.
        """
        found_completed_future = False

        for index, future in enumerate(self.futures):
            if index in self.completed_futures or not future.done():
                continue

            self.completed_futures.add(index)
            found_completed_future = True

            exception = future.exception()
            if exception is not None:
                self._cancel_pending_futures(exclude=future)
                raise exception

        return found_completed_future

    def _cancel_pending_futures(
        self,
        exclude: concurrent.futures.Future[Any],
    ) -> None:
        """Cancel incomplete futures other than the failed future."""
        for future in self.futures:
            if future is exclude or future.done():
                continue

            future.cancel()

    def _log_final_progress_states(self) -> None:
        """Ensure the latest progress state for every seed is logged."""
        for seed, state in sorted(self.seed_states.items()):
            last_logged = self.last_logged_percentages.get(seed)

            if last_logged == state.percentage:
                continue

            self._write_progress_log(seed, state, final=True)

    def _write_progress_log(
        self,
        seed: int,
        state: SeedProgress,
        final: bool = False,
    ) -> None:
        """Write one formatted progress entry to the configured logger."""
        label = "final progress" if final else "progress"

        self.logger.info(
            "Seed %s %s: %d/%d (%d%%) - %s",
            seed,
            label,
            state.step,
            state.total,
            state.percentage,
            state.status,
        )
