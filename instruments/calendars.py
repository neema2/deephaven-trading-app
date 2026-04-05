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
from enum import IntEnum

class BDConvention(IntEnum):
    Unadjusted = 0
    Following = 1
    ModifiedFollowing = 2
    Preceding = 3
    ModifiedPreceding = 4


def get_easter_sunday(year: int) -> datetime.date:
    """
    Calculates the date of Easter Sunday for a given Gregorian year.
    Based on the Meeus/Jones/Butcher algorithm.
    """
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    
    return datetime.date(year, month, day + 1)


class BusinessCalendar:
    """A calendar that defines business and non-business days."""
    
    def __init__(
        self, 
        name: str, 
        weekend: Sequence[int] = (5, 6), # 0=Mon, 5=Sat, 6=Sun
        fixed_holidays: Optional[Sequence[tuple[int, int]]] = None, # (month, day)
        ad_hoc_holidays: Optional[Sequence[datetime.date]] = None,
        easter_relative_rules: Optional[Sequence[tuple[int, str]]] = None, # (offset, name)
        nth_weekday_rules: Optional[Sequence[tuple[int, int, int, str]]] = None, # (month, weekday, n, name)
        fixed_adjusted_rules: Optional[Sequence[tuple[int, int, str, str]]] = None, # (month, day, rule, name)
        lunar_new_year_rules: Optional[Sequence[tuple[int, str]]] = None, # (offset, name)
        mid_autumn_rules: Optional[Sequence[tuple[int, str]]] = None # (offset, name)
    ):
        self._name = name
        self.weekend = set(weekend)
        self.fixed = set(fixed_holidays or [])
        self.ad_hoc = set(ad_hoc_holidays or [])
        self._easter_rel_rules = easter_relative_rules or []
        self._nth_rules = nth_weekday_rules or []
        self._fixed_adj_rules = fixed_adjusted_rules or []
        self._lunar_new_year_rules = lunar_new_year_rules or []
        self._mid_autumn_rules = mid_autumn_rules or []
        self._cached_holidays: dict[int, dict[datetime.date, str]] = {} # year -> {date: name}

    # Reference table for Lunar New Year dates (2020-2055)
    # Sourced from Han calendar calculations (Standard used in holidays.py)
    LUNAR_NEW_YEAR_DATES = {
        2020: (1, 25), 2021: (2, 12), 2022: (2, 1), 2023: (1, 22), 2024: (2, 10),
        2025: (1, 29), 2026: (2, 17), 2027: (2, 6), 2028: (1, 26), 2029: (2, 13),
        2030: (2, 3),  2031: (1, 23), 2032: (2, 11), 2033: (1, 31), 2034: (2, 19),
        2035: (2, 8),  2036: (1, 28), 2037: (2, 15), 2038: (2, 4),  2039: (1, 24),
        2040: (2, 12), 2041: (2, 1),  2042: (1, 22), 2043: (2, 10), 2044: (1, 30),
        2045: (2, 17), 2046: (2, 6),  2047: (1, 26), 2048: (2, 14), 2049: (2, 2),
        2050: (1, 23), 2051: (2, 11), 2052: (2, 1),  2053: (2, 19), 2054: (2, 8),
        2055: (1, 28)
    }

    # Mid-Autumn Festival dates (2020-2055)
    MID_AUTUMN_DATES = {
        2020: (10, 1), 2021: (9, 21), 2022: (9, 10), 2023: (9, 29), 2024: (9, 17),
        2025: (10, 6), 2026: (9, 25), 2027: (9, 15), 2028: (10, 3), 2029: (9, 22),
        2030: (9, 12), 2031: (10, 1), 2032: (9, 19), 2033: (9, 8),  2034: (9, 27),
        2035: (9, 16), 2036: (10, 4), 2037: (9, 24), 2038: (9, 13), 2039: (10, 2),
        2040: (9, 20), 2041: (9, 10), 2042: (9, 28), 2043: (9, 17), 2044: (10, 5),
        2045: (9, 25), 2046: (9, 15), 2047: (10, 4), 2048: (9, 22), 2049: (9, 11),
        2050: (9, 30), 2051: (9, 19), 2052: (9, 7),  2053: (9, 26), 2054: (9, 16),
        2055: (10, 5)
    }

    def name(self) -> str:
        return self._name

    def is_business_day(self, d: datetime.date) -> bool:
        """Check if a date is a business day (not weekend, not holiday)."""
        if d.weekday() in self.weekend:
            return False
            
        if d.year not in self._cached_holidays:
            self._precalculate_year(d.year)
            
        if d in self._cached_holidays[d.year]:
            return False
            
        if d in self.ad_hoc:
            return False
            
        return True

    def holiday_name(self, d: datetime.date) -> Optional[str]:
        """Return the name of the holiday if d is a holiday, else None."""
        if d.weekday() in self.weekend:
            import calendar
            return calendar.day_name[d.weekday()]
            
        if d.year not in self._cached_holidays:
            self._precalculate_year(d.year)
            
        return self._cached_holidays[d.year].get(d) or ("Ad-hoc Holiday" if d in self.ad_hoc else None)

    def _precalculate_year(self, year: int):
        """Build the cache of rule-based holidays for a given year."""
        hols: dict[datetime.date, str] = {}
        
        # 1. Fixed
        for m, dy, name in self.fixed:
            hols[datetime.date(year, m, dy)] = name
            
        # 2. Fixed Adjusted
        for m, dy, rule, name in self._fixed_adj_rules:
            # Shift a fixed date if it falls on a weekend
            original = datetime.date(year, m, dy)
            w = original.weekday()
            if rule == "MondayIfSunday":
                if w == 6: hols[original + datetime.timedelta(days=1)] = f"{name} (Observed)"
                else: hols[original] = name
            elif rule == "Following":
                curr = original
                while curr.weekday() in self.weekend:
                    curr += datetime.timedelta(days=1)
                hols[curr] = name if curr == original else f"{name} (Observed)"
            elif rule == "USRetail": # Sat->Fri, Sun->Mon
                if w == 5: hols[original - datetime.timedelta(days=1)] = f"{name} (Observed)"
                elif w == 6: hols[original + datetime.timedelta(days=1)] = f"{name} (Observed)"
                else: hols[original] = name
            elif rule == "PrecedingMonday": # Monday on or preceding (Victoria Day)
                hols[original - datetime.timedelta(days=original.weekday())] = name
            
        # 3. Easter Relative
        if self._easter_rel_rules:
            easter = get_easter_sunday(year)
            for offset, name in self._easter_rel_rules:
                hols[easter + datetime.timedelta(days=offset)] = name
                
        # 4. Lunar New Year Relative
        if self._lunar_new_year_rules and year in self.LUNAR_NEW_YEAR_DATES:
            m, d = self.LUNAR_NEW_YEAR_DATES[year]
            lny = datetime.date(year, m, d)
            for offset, name in self._lunar_new_year_rules:
                hols[lny + datetime.timedelta(days=offset)] = name
                
        # 5. Mid Autumn Relative
        if self._mid_autumn_rules and year in self.MID_AUTUMN_DATES:
            m, d = self.MID_AUTUMN_DATES[year]
            maf = datetime.date(year, m, d)
            for offset, name in self._mid_autumn_rules:
                hols[maf + datetime.timedelta(days=offset)] = name
            
        # 6. Nth Weekday
        import calendar
        for m, wd, n, name in self._nth_rules:
            # month m, weekday wd, n-th occurrence
            cal = calendar.monthcalendar(year, m)
            # cal is a list of weeks, each week is list of 7 days (0 if not in month)
            days_of_week = []
            for week in cal:
                if week[wd] != 0:
                    days_of_week.append(week[wd])
            
            if n == -1: # Last
                hols[datetime.date(year, m, days_of_week[-1])] = name
            elif 1 <= n <= len(days_of_week):
                hols[datetime.date(year, m, days_of_week[n-1])] = name
                
        self._cached_holidays[year] = hols

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
    def load_from_yaml(cls, path: Optional[str] = None) -> None:
        """Load all calendar definitions from a YAML file."""
        if path is None:
            # Default to the peer directory of this file
            path = str(Path(__file__).parent / "calendars.yaml")
            
        if not Path(path).exists():
            return
            
        with open(path, "r") as f:
            data = yaml.safe_load(f)
            
        for ccy, cfg in data.items():
            fixed = [tuple(f) for f in cfg.get("fixed", [])]
            ad_hoc = [datetime.date.fromisoformat(d) for d in cfg.get("ad_hoc", [])]
            nth = [tuple(rule) for rule in cfg.get("nth_weekday_rules", [])]
            adj = [tuple(rule) for rule in cfg.get("fixed_adjusted_rules", [])]
            easter_rel = [tuple(rule) for rule in cfg.get("easter_relative_rules", [])]
            lunar_rel = [tuple(rule) for rule in cfg.get("lunar_new_year_rules", [])]
            mid_autumn = [tuple(rule) for rule in cfg.get("mid_autumn_rules", [])]
            
            cls._cache[ccy.upper()] = BusinessCalendar(
                name=ccy.upper(),
                weekend=cfg.get("weekend", (5, 6)),
                fixed_holidays=fixed,
                ad_hoc_holidays=ad_hoc,
                easter_relative_rules=easter_rel,
                nth_weekday_rules=nth,
                fixed_adjusted_rules=adj,
                lunar_new_year_rules=lunar_rel,
                mid_autumn_rules=mid_autumn
            )

    @classmethod
    def get_calendar(cls, name: str) -> BusinessCalendar:
        """Get a calendar for a specific currency/name."""
        # Initial load if cache empty
        if not cls._cache:
            cls.load_from_yaml()
            
        # Fallback to a default calendar (Mon-Fri) if not found
        return cls._cache.get(name.upper(), BusinessCalendar(name=name.upper()))


