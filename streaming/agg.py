"""
streaming.agg — Aggregation helpers re-exported from Deephaven.

Usage::

    from streaming import agg

    summary = live.agg_by([
        agg.sum(["TotalMV=MarketValue"]),
        agg.avg(["AvgGamma=Gamma"]),
        agg.count("NumPositions"),
    ])

**Dual-mode:** In-process mode returns native Deephaven agg objects.
Remote (Docker) mode returns ``RemoteAgg`` wrappers whose ``repr()``
produces valid DH Python code for ``run_script()``.
"""

from __future__ import annotations

from typing import Any

from streaming.admin import _needs_docker

_REMOTE = _needs_docker()


# ===========================================================================
# In-process mode — thin lazy wrappers around deephaven.agg
# ===========================================================================

if not _REMOTE:

    def _dh_agg() -> Any:
        from deephaven import agg as _agg
        return _agg

    def sum(cols: list[str]) -> Any:
        """Sum columns.  Wraps ``deephaven.agg.sum_``."""
        return _dh_agg().sum_(cols)

    def avg(cols: list[str]) -> Any:
        """Average columns.  Wraps ``deephaven.agg.avg``."""
        return _dh_agg().avg(cols)

    def count(col: str) -> Any:
        """Count rows.  Wraps ``deephaven.agg.count_``."""
        return _dh_agg().count_(col)

    def min(cols: list[str]) -> Any:
        """Min columns.  Wraps ``deephaven.agg.min_``."""
        return _dh_agg().min_(cols)

    def max(cols: list[str]) -> Any:
        """Max columns.  Wraps ``deephaven.agg.max_``."""
        return _dh_agg().max_(cols)

    def first(cols: list[str]) -> Any:
        """First value per group.  Wraps ``deephaven.agg.first``."""
        return _dh_agg().first(cols)

    def last(cols: list[str]) -> Any:
        """Last value per group.  Wraps ``deephaven.agg.last``."""
        return _dh_agg().last(cols)

    def std(cols: list[str]) -> Any:
        """Standard deviation.  Wraps ``deephaven.agg.std``."""
        return _dh_agg().std(cols)

    def var(cols: list[str]) -> Any:
        """Variance.  Wraps ``deephaven.agg.var``."""
        return _dh_agg().var(cols)

    def median(cols: list[str]) -> Any:
        """Median.  Wraps ``deephaven.agg.median``."""
        return _dh_agg().median(cols)


# ===========================================================================
# Remote mode — produces repr()-safe objects for run_script()
# ===========================================================================

else:

    class RemoteAgg:
        """Serializable agg descriptor for remote Deephaven servers.

        ``repr()`` produces valid DH Python: ``agg.sum_(['col'])``.
        """

        __slots__ = ("_expr",)

        def __init__(self, expr: str) -> None:
            self._expr = expr

        def __repr__(self) -> str:
            return self._expr

    def sum(cols: list[str]) -> RemoteAgg:
        return RemoteAgg(f"agg.sum_({cols!r})")

    def avg(cols: list[str]) -> RemoteAgg:
        return RemoteAgg(f"agg.avg({cols!r})")

    def count(col: str) -> RemoteAgg:
        return RemoteAgg(f"agg.count_({col!r})")

    def min(cols: list[str]) -> RemoteAgg:
        return RemoteAgg(f"agg.min_({cols!r})")

    def max(cols: list[str]) -> RemoteAgg:
        return RemoteAgg(f"agg.max_({cols!r})")

    def first(cols: list[str]) -> RemoteAgg:
        return RemoteAgg(f"agg.first({cols!r})")

    def last(cols: list[str]) -> RemoteAgg:
        return RemoteAgg(f"agg.last({cols!r})")

    def std(cols: list[str]) -> RemoteAgg:
        return RemoteAgg(f"agg.std({cols!r})")

    def var(cols: list[str]) -> RemoteAgg:
        return RemoteAgg(f"agg.var({cols!r})")

    def median(cols: list[str]) -> RemoteAgg:
        return RemoteAgg(f"agg.median({cols!r})")

