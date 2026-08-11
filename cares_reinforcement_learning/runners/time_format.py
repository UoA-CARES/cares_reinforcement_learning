"""Utilities for formatting elapsed durations for logs."""


def format_elapsed_duration(total_seconds: float) -> str:
    """Return a human-readable duration string including days.

    Example output: "2 days 3 hours 4 minutes 5 seconds"
    """
    seconds = max(0, int(total_seconds))

    days, rem = divmod(seconds, 24 * 60 * 60)
    hours, rem = divmod(rem, 60 * 60)
    minutes, secs = divmod(rem, 60)

    parts: list[str] = []
    if days:
        parts.append(f"{days} day" if days == 1 else f"{days} days")
    if hours:
        parts.append(f"{hours} hour" if hours == 1 else f"{hours} hours")
    if minutes:
        parts.append(f"{minutes} minute" if minutes == 1 else f"{minutes} minutes")
    if secs or not parts:
        parts.append(f"{secs} second" if secs == 1 else f"{secs} seconds")

    return " ".join(parts)
