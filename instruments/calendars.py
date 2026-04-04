"""
calendars.py — Flexible Holiday Calendar System

Implements a lightweight, local-first business calendar that can be
configured via YAML. Supports deterministic rules (weekends, fixed dates)
and explicit ad-hoc holiday overrides (e.g., extracted from QuantLib).
"""

from __future__ import annotations

import datetime
from typing import Optional, Sequence, Union, Protocol
import yaml
from pathlib import Path


from instruments.ir_scheduling import BDConvention


class BusinessCalendar:
    """A calendar that defines business and non-business days."""
    
    def __init__(
        self, 
        name: str, 
        weekend: Sequence[int] = (5, 6), # 0=Mon, 5=Sat, 6=Sun
        fixed_holidays: Optional[Sequence[tuple[int, int]]] = None, # (month, day)
        ad_hoc_holidays: Optional[Sequence[datetime.date]] = None
    ):
        self._name = name
        self.weekend = set(weekend)
        self.fixed = set(fixed_holidays or [])
        self.ad_hoc = set(ad_hoc_holidays or [])

    def name(self) -> str:
        return self._name

    def is_business_day(self, d: datetime.date) -> bool:
        """Check if a date is a business day (not weekend, not holiday)."""
        if d.weekday() in self.weekend:
            return False
        if (d.month, d.day) in self.fixed:
            return False
        if d in self.ad_hoc:
            return False
        return True

    def adjust(self, d: datetime.date, convention: BDConvention = BDConvention.ModifiedFollowing) -> datetime.date:
        """Adjust a holiday to the next/previous business day according to convention."""
        if convention == BDConvention.Unadjusted or self.is_business_day(d):
            return d
            
        if convention == BDConvention.Following:
            return self._shift(d, 1)
        elif convention == BDConvention.Preceding:
            return self._shift(d, -1)
        elif convention == BDConvention.ModifiedFollowing:
            new_date = self._shift(d, 1)
            # If we crossed into a new month, go backwards instead
            if new_date.month != d.month:
                return self._shift(d, -1)
            return new_date
        elif convention == BDConvention.ModifiedPreceding:
            new_date = self._shift(d, -1)
            if new_date.month != d.month:
                return self._shift(d, 1)
            return new_date
            
        return d

    def _shift(self, d: datetime.date, direction: int) -> datetime.date:
        """Step day by day until a business day is found."""
        curr = d
        while not self.is_business_day(curr):
            curr += datetime.timedelta(days=direction)
        return curr

    def business_days_between(self, start: datetime.date, end: datetime.date) -> int:
        """Count business days in [start, end)."""
        if start >= end: return 0
        count = 0
        curr = start
        while curr < end:
            if self.is_business_day(curr):
                count += 1
            curr += datetime.timedelta(days=1)
        return count


class CalendarFactory:
    """Loader for the YAML calendar configuration."""
    
    _cache: dict[str, BusinessCalendar] = {}
    
    @classmethod
    def load_from_yaml(cls, path: str = "calendars.yaml") -> None:
        """Load all calendar definitions from a YAML file."""
        if not Path(path).exists():
            return
            
        with open(path, "r") as f:
            data = yaml.safe_load(f)
            
        for ccy, cfg in data.items():
            fixed = [tuple(f) for f in cfg.get("fixed", [])]
            ad_hoc = [datetime.date.fromisoformat(d) for d in cfg.get("ad_hoc", [])]
            
            cls._cache[ccy.upper()] = BusinessCalendar(
                name=ccy.upper(),
                weekend=cfg.get("weekend", (5, 6)),
                fixed_holidays=fixed,
                ad_hoc_holidays=ad_hoc
            )

    @classmethod
    def get_calendar(cls, name: str) -> BusinessCalendar:
        """Get a calendar for a specific currency/name."""
        # Initial load if cache empty
        if not cls._cache:
            cls.load_from_yaml()
            
        # Fallback to a default calendar (Mon-Fri) if not found
        return cls._cache.get(name.upper(), BusinessCalendar(name=name.upper()))


def export_quantlib_holidays(ccy: str, years: int = 30) -> list[str]:
    """Utility to extract ad-hoc holidays from QuantLib for YAML config.
    
    This can be used to populate the 'ad_hoc' section of calendars.yaml.
    """
    try:
        import QuantLib as ql
    except ImportError:
        return []
        
    # Map ccy to QL calendar
    mapping = {
        "USD": ql.UnitedStates(ql.UnitedStates.Settlement),
        "EUR": ql.TARGET(),
        "GBP": ql.UnitedKingdom(),
        "JPY": ql.Japan(),
        "AUD": ql.Australia(),
        "CAD": ql.Canada(),
        "CHF": ql.Switzerland(),
        "SGD": ql.Singapore(),
        "SGP": ql.Singapore(),
    }
    
    qcal = mapping.get(ccy.upper())
    if not qcal: return []
    
    start_date = ql.Date(1, 1, datetime.date.today().year)
    end_date = ql.Date(1, 1, start_date.year() + years)
    
    ql_holidays = qcal.holidayList(qcal, start_date, end_date)
    
    # Filter out weekends and fixed dates already covered by deterministic rules
    # Fixed = Jan 1, Dec 25
    ad_hoc = []
    for qd in ql_holidays:
        py_d = datetime.date(qd.year(), qd.month(), qd.dayOfMonth())
        if py_d.weekday() >= 5: continue
        if (py_d.month == 1 and py_d.day == 1) or (py_d.month == 12 and py_d.day == 25):
            continue
        ad_hoc.append(py_d.isoformat())
        
    return sorted(ad_hoc)

if __name__ == "__main__":
    # Example usage:
    # print(export_quantlib_holidays("USD"))
    pass
