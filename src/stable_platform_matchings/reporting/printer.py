"""Consistent terminal formatting for optimization solver output."""

from __future__ import annotations

from pprint import pformat
from typing import Any


class Printer:
    """Small, dependency-free formatter for hierarchical solver logs."""

    def __init__(self, width: int = 80, enabled: bool = True) -> None:
        if width < 40:
            raise ValueError("print width must be at least 40 characters")
        self.width = width
        self.enabled = enabled

    def _print(self, message: str = "") -> None:
        if self.enabled:
            print(message)

    def section(self, title: str, fill: str = "=") -> None:
        """Print a major centered section heading."""
        self._print()
        self._print(f" {title} ".center(self.width, fill))

    def subsection(self, title: str, fill: str = "-") -> None:
        """Print a centered heading within the current section."""
        self._print(f" {title} ".center(self.width, fill))

    def message(self, message: str, indent: int = 0) -> None:
        prefix = "  " * indent
        for line in str(message).splitlines() or [""]:
            self._print(f"{prefix}{line}")

    def metric(
        self,
        label: str,
        value: Any,
        *,
        indent: int = 1,
        precision: int = 3,
    ) -> None:
        if isinstance(value, float):
            formatted = f"{value:,.{precision}f}"
        else:
            formatted = str(value)

        prefix = "  " * indent
        label_width = min(30, max(18, self.width // 3))
        self._print(f"{prefix}{label:<{label_width}} {formatted}")

    def collection(self, label: str, values: Any, *, indent: int = 1) -> None:
        prefix = "  " * indent
        available_width = max(20, self.width - len(prefix) - 2)
        formatted = pformat(values, sort_dicts=True, width=available_width)
        self._print(f"{prefix}{label}:")
        for line in formatted.splitlines():
            self._print(f"{prefix}  {line}")

    def status(self, message: str, indent: int = 1) -> None:
        self.message(f"[STATUS] {message}", indent)

    def success(self, message: str, indent: int = 1) -> None:
        self.message(f"[OK] {message}", indent)

    def warning(self, message: str, indent: int = 1) -> None:
        self.message(f"[WARN] {message}", indent)

    def iteration(self, iteration: int, label: str = "Iteration") -> None:
        self.subsection(f"{label} {iteration}", fill=".")

    def blank(self) -> None:
        self._print()
