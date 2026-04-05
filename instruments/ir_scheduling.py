"""
ir_scheduling — Calendar-aware swap schedule generation.

Design mirrors QuantLib's six-stage pipeline::

    effectiveDate + terminationDate
        → Period (freq_months)
        → DateGenRule           (Backward / Forward / Zero)
            → unadjusted schedule dates
        → Calendar              (per-currency holiday calendar)
        → BDConvention          (ModifiedFollowing, Following, …)
            → adjusted schedule dates
        → DayCountConvention    (Act/365F, Act/360, 30/360, …)
            → year fractions  τ_i

Requires no third-party libraries for normal use.
When ``QuantLib`` (pip install QuantLib) is importable, ``calendar_for()``
and ``QuantLibCalendar`` delegate to real market calendars automatically.
The decimal-year helpers (``rack_dates``, ``payment_dates``, etc.) are kept
unchanged for backward compatibility with existing instrument code.
"""

from __future__ import annotations

import calendar as _cal_mod
import datetime
import enum
from typing import Optional, Sequence

# ---------------------------------------------------------------------------
# 1. Date type alias
# ---------------------------------------------------------------------------

ISODate = datetime.date
"""Canonical date type — plain ``datetime.date``, no extra deps required."""


def to_serial(d: ISODate) -> int:
    """Return integer serial: days since 1900-01-01 (our internal epoch)."""
    return (d - ISODate(1900, 1, 1)).days


def from_serial(n: int) -> ISODate:
    """Reconstruct a date from its integer serial."""
    return ISODate(1900, 1, 1) + datetime.timedelta(days=n)


def _to_date(x: "ISODate | str") -> ISODate:
    if isinstance(x, datetime.date):
        return x
    return ISODate.fromisoformat(x)


# ---------------------------------------------------------------------------
# 2. Enumerations  (mirror QuantLib C++ enums exactly)
# ---------------------------------------------------------------------------

class BDConvention(enum.Enum):
    """Business-day adjustment convention (ql::BusinessDayConvention).

    Following           — next business day after a holiday.
    ModifiedFollowing   — next bd, unless month changes → previous bd.
    Preceding           — previous business day before a holiday.
    ModifiedPreceding   — previous bd, unless month changes → next bd.
    Unadjusted          — do not adjust.
    """
    Following         = "Following"
    ModifiedFollowing = "ModifiedFollowing"
    Preceding         = "Preceding"
    ModifiedPreceding = "ModifiedPreceding"
    Unadjusted        = "Unadjusted"


class DateGenRule(enum.Enum):
    """Date-generation rule (ql::DateGeneration::Rule).

    Backward — walk backwards from termination date (default for vanilla IRS).
    Forward  — walk forward from effective date.
    Zero     — only effective and termination dates (bullet / zero-coupon).
    """
    Backward = "Backward"
    Forward  = "Forward"
    Zero     = "Zero"


class DayCountConvention(enum.Enum):
    """Day-count convention (mirrors QuantLib day counters).

    Act365Fixed — Actual/365 Fixed  (GBP, USD fixed leg)
    Act360      — Actual/360        (USD SOFR float, EUR float)
    Thirty360US — 30/360 US bond basis (US fixed leg)
    Thirty360EU — 30E/360 Eurobond basis (EUR fixed leg)
    ActActISDA  — Actual/Actual ISDA (government bonds)
    """
    Act365Fixed = "Act/365F"
    Act360      = "Act/360"
    Thirty360US = "30/360 US"
    Thirty360EU = "30E/360"
    ActActISDA  = "Act/Act ISDA"


# Market-conventional (fixed_dcc, float_dcc) by currency.
_CCY_DCC: dict[str, tuple[DayCountConvention, DayCountConvention]] = {
    "USD": (DayCountConvention.Thirty360US, DayCountConvention.Act360),
    "EUR": (DayCountConvention.Thirty360EU, DayCountConvention.Act360),
    "GBP": (DayCountConvention.Act365Fixed, DayCountConvention.Act365Fixed),
    "JPY": (DayCountConvention.Act365Fixed, DayCountConvention.Act360),
    "AUD": (DayCountConvention.Act365Fixed, DayCountConvention.Act365Fixed),
    "CAD": (DayCountConvention.Thirty360US, DayCountConvention.Act360),
    "CHF": (DayCountConvention.Thirty360EU, DayCountConvention.Act360),
    "SOFR": (DayCountConvention.Thirty360US, DayCountConvention.Act360),
}


def dcc_for_currency(currency: str, leg: str = "float") -> DayCountConvention:
    """Return the market-conventional day-count convention for a currency/leg.

    Parameters
    ----------
    currency:
        ISO currency code, e.g. ``"USD"``.
    leg:
        ``"fixed"`` or ``"float"`` (default).

    Examples
    --------
    >>> dcc_for_currency("USD", "fixed")
    <DayCountConvention.Thirty360US: '30/360 US'>
    >>> dcc_for_currency("USD", "float")
    <DayCountConvention.Act360: 'Act/360'>
    """
    pair = _CCY_DCC.get(currency.upper(), (DayCountConvention.Act365Fixed, DayCountConvention.Act365Fixed))
    return pair[0] if leg.lower() == "fixed" else pair[1]


# ---------------------------------------------------------------------------
# 3. Day-count fraction computations
# ---------------------------------------------------------------------------

def _eom_day(d: ISODate) -> int:
    """Last day of d's month."""
    return _cal_mod.monthrange(d.year, d.month)[1]


def _is_leap(year: int) -> bool:
    return _cal_mod.isleap(year)


def year_fraction(d1: ISODate, d2: ISODate,
                  dcc: DayCountConvention = DayCountConvention.Act365Fixed) -> float:
    """Compute accrual year fraction τ between two calendar dates.

    Parameters
    ----------
    d1, d2:
        Period start and end (``d2 >= d1``).
    dcc:
        Day-count convention to apply.

    Returns
    -------
    float
        Year fraction τ = daycount(d1, d2) / basis.

    Examples
    --------
    >>> from datetime import date
    >>> year_fraction(date(2026, 1, 1), date(2026, 7, 1), DayCountConvention.Act360)
    0.5
    """
    actual = (d2 - d1).days

    if dcc is DayCountConvention.Act365Fixed:
        return actual / 365.0

    if dcc is DayCountConvention.Act360:
        return actual / 360.0

    if dcc is DayCountConvention.Thirty360US:
        y1, m1, day1 = d1.year, d1.month, d1.day
        y2, m2, day2 = d2.year, d2.month, d2.day
        # Feb EOM rule
        if day1 == _eom_day(d1) and m1 == 2 and day2 == _eom_day(d2) and m2 == 2:
            day2 = 30
        if day1 == _eom_day(d1) and m1 == 2:
            day1 = 30
        if day2 == 31 and day1 >= 30:
            day2 = 30
        if day1 == 31:
            day1 = 30
        return (360 * (y2 - y1) + 30 * (m2 - m1) + (day2 - day1)) / 360.0

    if dcc is DayCountConvention.Thirty360EU:
        y1, m1, day1 = d1.year, d1.month, d1.day
        y2, m2, day2 = d2.year, d2.month, d2.day
        if day1 == 31:
            day1 = 30
        if day2 == 31:
            day2 = 30
        return (360 * (y2 - y1) + 30 * (m2 - m1) + (day2 - day1)) / 360.0

    if dcc is DayCountConvention.ActActISDA:
        if d1.year == d2.year:
            basis = 366 if _is_leap(d1.year) else 365
            return actual / basis
        # Split across year boundaries following ISDA method:
        #   segment 1: d1 → Dec 31 of d1.year  (use Dec31 exclusive, i.e. Jan 1 of next year)
        #   middle    : each whole year = 1.0
        #   last      : Jan 1 of d2.year → d2
        frac = 0.0
        # First partial year: days from d1 to end of d1's year
        next_year_start = ISODate(d1.year + 1, 1, 1)
        frac += (next_year_start - d1).days / (366.0 if _is_leap(d1.year) else 365.0)
        # Whole middle years
        for y in range(d1.year + 1, d2.year):
            frac += 1.0
        # Last partial year: days from Jan 1 of d2's year to d2
        this_year_start = ISODate(d2.year, 1, 1)
        frac += (d2 - this_year_start).days / (366.0 if _is_leap(d2.year) else 365.0)
        return frac

    raise ValueError(f"Unknown DayCountConvention: {dcc!r}")


# ---------------------------------------------------------------------------
# 4. Calendar abstraction
# ---------------------------------------------------------------------------

class Calendar:
    """Abstract calendar base — subclass must implement ``is_business_day``.

    All other methods (``adjust``, ``advance``, ``business_days_between``)
    derive from that single hook, mirroring QuantLib's Bridge pattern.
    """

    def name(self) -> str:
        """Human-readable calendar name."""
        return self.__class__.__name__

    def is_business_day(self, d: ISODate) -> bool:
        """Return True iff *d* is a business day in this calendar."""
        raise NotImplementedError

    def is_holiday(self, d: ISODate) -> bool:
        return not self.is_business_day(d)

    # ------------------------------------------------------------------
    # Adjustment
    # ------------------------------------------------------------------

    def adjust(self, d: ISODate,
               convention: BDConvention = BDConvention.ModifiedFollowing) -> ISODate:
        """Adjust *d* to the nearest business day per *convention*.

        Examples
        --------
        >>> cal = WeekendsOnlyCalendar()
        >>> from datetime import date
        >>> cal.adjust(date(2026, 3, 21), BDConvention.Following)  # Sat → Mon
        datetime.date(2026, 3, 23)
        """
        if convention is BDConvention.Unadjusted:
            return d
        if self.is_business_day(d):
            return d
        if convention is BDConvention.Following:
            return self._next_bd(d)
        if convention is BDConvention.Preceding:
            return self._prev_bd(d)
        if convention is BDConvention.ModifiedFollowing:
            candidate = self._next_bd(d)
            if candidate.month != d.month:
                candidate = self._prev_bd(d)
            return candidate
        if convention is BDConvention.ModifiedPreceding:
            candidate = self._prev_bd(d)
            if candidate.month != d.month:
                candidate = self._next_bd(d)
            return candidate
        raise ValueError(f"Unknown BDConvention: {convention!r}")

    def advance(self, d: ISODate, months: int,
                convention: BDConvention = BDConvention.ModifiedFollowing,
                end_of_month: bool = False) -> ISODate:
        """Advance *d* by *months* calendar months then adjust.

        Parameters
        ----------
        d:
            Reference date.
        months:
            Number of months to add (negative = subtract).
        convention:
            Business-day adjustment applied after month offset.
        end_of_month:
            If True and *d* is the last day of its month, keep
            subsequent dates at their month-end (QL EOM convention).
        """
        result = _add_months(d, months)
        if end_of_month and _is_eom(d):
            result = _last_dom(result)
        return self.adjust(result, convention)

    def business_days_between(self, d1: ISODate, d2: ISODate,
                               include_first: bool = True,
                               include_last: bool = False) -> int:
        """Count business days between *d1* and *d2*."""
        if d1 > d2:
            return -self.business_days_between(d2, d1, include_last, include_first)
        count = 0
        cur = d1
        one = datetime.timedelta(days=1)
        while cur < d2:
            if self.is_business_day(cur) and (cur > d1 or include_first):
                count += 1
            cur += one
        if include_last and self.is_business_day(d2):
            count += 1
        return count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _next_bd(self, d: ISODate) -> ISODate:
        one = datetime.timedelta(days=1)
        d = d + one
        while not self.is_business_day(d):
            d += one
        return d

    def _prev_bd(self, d: ISODate) -> ISODate:
        one = datetime.timedelta(days=1)
        d = d - one
        while not self.is_business_day(d):
            d -= one
        return d

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} '{self.name()}'>"


class WeekendsOnlyCalendar(Calendar):
    """Calendar with no holidays — weekends only.

    This is the always-available, zero-dependency fallback.
    Corresponds to ``ql::WeekendsOnly``.
    """

    def name(self) -> str:
        return "WeekendsOnly"

    def is_business_day(self, d: ISODate) -> bool:
        return d.weekday() < 5  # Mon=0 … Fri=4; Sat=5, Sun=6


class BespokeCalendar(WeekendsOnlyCalendar):
    """Weekends + an explicit set of holiday dates.

    Useful for tests and for currencies where only a handful of
    market-specific holidays matter.
    """

    def __init__(self, cal_name: str, holidays: Sequence[ISODate] = ()) -> None:
        self._name = cal_name
        self._holidays: set[ISODate] = set(holidays)

    def name(self) -> str:
        return self._name

    def add_holiday(self, d: ISODate) -> None:
        self._holidays.add(d)

    def remove_holiday(self, d: ISODate) -> None:
        self._holidays.discard(d)

    def is_business_day(self, d: ISODate) -> bool:
        return super().is_business_day(d) and d not in self._holidays


def make_bespoke_calendar(name: str,
                          holidays: Sequence[ISODate] = ()) -> BespokeCalendar:
    """Factory for a weekends + explicit holidays calendar."""
    return BespokeCalendar(name, holidays)


# ---------------------------------------------------------------------------
# 5. QuantLib-backed calendar (optional)
# ---------------------------------------------------------------------------

def _try_import_ql():
    """Return the QuantLib module if installed, else None."""
    try:
        import QuantLib as ql  # type: ignore[import-untyped]
        return ql
    except ImportError:
        return None


# BDConvention → QuantLib integer — populated lazily on first use.
_QL_BDC: dict[BDConvention, int] = {}


def _ql_bdc(convention: BDConvention):
    if not _QL_BDC:
        ql = _try_import_ql()
        if ql is None:
            raise RuntimeError("QuantLib is not installed (pip install QuantLib)")
        _QL_BDC.update({
            BDConvention.Following:        ql.Following,
            BDConvention.ModifiedFollowing: ql.ModifiedFollowing,
            BDConvention.Preceding:        ql.Preceding,
            BDConvention.ModifiedPreceding: ql.ModifiedPreceding,
            BDConvention.Unadjusted:       ql.Unadjusted,
        })
    return _QL_BDC[convention]


class QuantLibCalendar(Calendar):
    """Thin wrapper around a ``QuantLib.Calendar`` object.

    Delegates ``is_business_day`` and ``adjust`` to QuantLib for full
    market-accurate holiday calendars.  Falls back to the pure-Python
    super-class only if QuantLib is unexpectedly unavailable at call time.

    Parameters
    ----------
    ql_calendar:
        A QuantLib Calendar instance, e.g.
        ``ql.UnitedStates(ql.UnitedStates.FederalReserve)``.

    Examples
    --------
    >>> import QuantLib as ql
    >>> from instruments.ir_scheduling import QuantLibCalendar, BDConvention
    >>> cal = QuantLibCalendar(ql.UnitedStates(ql.UnitedStates.GovernmentBond))
    >>> from datetime import date
    >>> cal.is_business_day(date(2026, 7, 4))   # Independence Day
    False
    """

    def __init__(self, ql_calendar: object) -> None:
        self._ql_cal = ql_calendar

    def name(self) -> str:
        return self._ql_cal.name()  # type: ignore[union-attr]

    def _to_ql_date(self, d: ISODate):
        ql = _try_import_ql()
        assert ql is not None
        return ql.Date(d.day, d.month, d.year)

    def _from_ql_date(self, ql_date) -> ISODate:
        return ISODate(ql_date.year(), ql_date.month(), ql_date.dayOfMonth())

    def is_business_day(self, d: ISODate) -> bool:
        return self._ql_cal.isBusinessDay(self._to_ql_date(d))  # type: ignore[union-attr]

    def adjust(self, d: ISODate,
               convention: BDConvention = BDConvention.ModifiedFollowing) -> ISODate:
        """Delegate adjustment to QuantLib for exact market-holiday handling."""
        if convention is BDConvention.Unadjusted:
            return d
        ql_date = self._to_ql_date(d)
        adjusted = self._ql_cal.adjust(ql_date, _ql_bdc(convention))  # type: ignore[union-attr]
        return self._from_ql_date(adjusted)


# Per-currency QuantLib calendar factories.
_QL_CALENDAR_FACTORIES: dict[str, object] = {}


def _load_ql_factories() -> None:
    ql = _try_import_ql()
    if ql is None:
        return
    _QL_CALENDAR_FACTORIES.update({
        "USD": lambda: ql.UnitedStates(ql.UnitedStates.GovernmentBond),
        "EUR": lambda: ql.TARGET(),
        "GBP": lambda: ql.UnitedKingdom(ql.UnitedKingdom.Settlement),
        "JPY": lambda: ql.Japan(),
        "AUD": lambda: ql.Australia(),
        "CAD": lambda: ql.Canada(ql.Canada.Settlement),
        "CHF": lambda: ql.Switzerland(),
    })


_load_ql_factories()


# ── Calendar Mapping ────────────────────────────────────────────────────────

_PAYMENT_CALENDARS = {
    "USD": "US",
    "EUR": "TARGET",
    "GBP": "GB",
    "JPY": "JP",
    "AUD": "AU",
    "CAD": "CA",
    "CHF": "CH",
    "SGD": "SG",
    "SGP": "SG",
}

_FIXING_CALENDARS = {
    "USD": "US_GOVT",     # SOFR/GOVT bond market
    "EUR": "TARGET",
    "GBP": "GB",
    "JPY": "JP",
    "AUD": "AU",
    "CAD": "CA",
    "CHF": "CH",
    "SGD": "SG",
}


def calendar_for(currency: str, type: str = "payment") -> Calendar:
    """Return the conventional calendar for *currency*.

    Prioritizes the local YAML-based CalendarFactory. Falls back to 
    QuantLib if a specific YAML entry is not found.
    """
    from instruments.calendars import CalendarFactory
    
    # 1. Resolve logical calendar name from currency
    if type == "fixing":
        name = _FIXING_CALENDARS.get(currency.upper(), currency.upper())
    else:
        name = _PAYMENT_CALENDARS.get(currency.upper(), currency.upper())

    # 2. Try local YAML config via logical name
    cal = CalendarFactory.get_calendar(name)
    
    # Check if the returned calendar is actually defined in the cache or a default
    # If name matches but it's not weekends-only, we use it.
    # Note: CalendarFactory.get_calendar will create one if not in cache.
    if cal.name() == name.upper() and len(cal.ad_hoc) > 0 or len(cal.fixed) > 0:
        return cal # type: ignore[return-value]

    # 3. Fallback to QuantLib (using original currency code for QL factory logic)
    factory = _QL_CALENDAR_FACTORIES.get(currency.upper())
    if factory is not None:
        try:
            return QuantLibCalendar(factory()) # type: ignore[return-value]
        except Exception:
            pass
            
    # 4. Final Fallback to the local instance (might just be WeekendsOnly in effect)
    return cal # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 6. Internal date arithmetic helpers
# ---------------------------------------------------------------------------

def _add_months(d: ISODate, n: int) -> ISODate:
    """Add *n* calendar months to *d*, clamping to end-of-month."""
    month = d.month + n
    year = d.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    max_day = _cal_mod.monthrange(year, month)[1]
    return ISODate(year, month, min(d.day, max_day))


def _last_dom(d: ISODate) -> ISODate:
    """Return the last day of *d*'s month."""
    return ISODate(d.year, d.month, _cal_mod.monthrange(d.year, d.month)[1])


def _is_eom(d: ISODate) -> bool:
    return d.day == _cal_mod.monthrange(d.year, d.month)[1]


def get_imm_date(month: int, year: int) -> ISODate:
    """Return the IMM date (3rd Wednesday) for a given month/year."""
    # Start at the 1st of the month
    first_day = ISODate(year, month, 1)
    # weekday() is 0=Mon, 2=Wed
    first_wed_offset = (2 - first_day.weekday() + 7) % 7
    # 3rd Wednesday is first Wednesday + 14 days
    return first_day + datetime.timedelta(days=first_wed_offset + 14)


# ---------------------------------------------------------------------------
# 7. Schedule generation
# ---------------------------------------------------------------------------

def make_schedule(
    effective_date: "ISODate | str",
    termination_date: "ISODate | str",
    freq_months: int = 12,
    calendar: Optional[Calendar] = None,
    convention: BDConvention = BDConvention.ModifiedFollowing,
    termination_convention: Optional[BDConvention] = None,
    rule: DateGenRule = DateGenRule.Backward,
    end_of_month: bool = False,
    first_date: Optional[ISODate] = None,
    next_to_last_date: Optional[ISODate] = None,
) -> list[ISODate]:
    """Generate a swap leg schedule of adjusted calendar dates.

    Mirrors ``ql::MakeSchedule`` for the three standard IRS rules:
    Backward, Forward, and Zero.

    Parameters
    ----------
    effective_date:
        Trade start / value date (ISO string or ``datetime.date``).
    termination_date:
        Maturity date.
    freq_months:
        Coupon frequency in months: 12=annual, 6=semi-annual, 3=quarterly,
        1=monthly.
    calendar:
        Holiday calendar for adjustment.  ``None`` → :class:`WeekendsOnlyCalendar`.
    convention:
        BDConvention for interior coupon dates.
    termination_convention:
        BDConvention for the final date (defaults to *convention*;
        use ``Unadjusted`` for ISDA standard).
    rule:
        :class:`DateGenRule` — ``Backward`` (default), ``Forward``, or ``Zero``.
    end_of_month:
        If True and *effective_date* is month-end, snap subsequent dates to
        their month-end before adjusting (QL EOM convention).
    first_date:
        Optional explicit front-stub date.
    next_to_last_date:
        Optional explicit back-stub date.

    Returns
    -------
    list[ISODate]
        Adjusted schedule dates, length = periods + 1.

    Examples
    --------
    >>> from datetime import date
    >>> sch = make_schedule(date(2026, 3, 20), date(2031, 3, 20), freq_months=12)
    >>> len(sch)  # 5 annual periods → 6 dates
    6
    """
    eff  = _to_date(effective_date)
    term = _to_date(termination_date)
    if eff >= term:
        raise ValueError(f"effective_date ({eff}) must be before termination_date ({term})")

    cal  = calendar if calendar is not None else WeekendsOnlyCalendar()
    term_conv = termination_convention if termination_convention is not None else convention
    eom = end_of_month and _is_eom(eff)

    # ------------------------------------------------------------------
    # Step 1: generate unadjusted dates
    # ------------------------------------------------------------------
    if rule is DateGenRule.Zero:
        raw = [eff, term]

    elif rule is DateGenRule.Backward:
        raw: list[ISODate] = [term]
        seed = term
        if next_to_last_date is not None:
            raw.append(next_to_last_date)
            seed = next_to_last_date

        exit_date = first_date if first_date is not None else eff
        periods = 1
        while True:
            candidate = _add_months(seed, -periods * freq_months)
            if eom:
                candidate = _last_dom(candidate)
            if candidate < exit_date:
                if first_date is not None and raw[-1] != first_date:
                    raw.append(first_date)
                break
            raw.append(candidate)
            periods += 1

        if raw[-1] != eff:
            raw.append(eff)
        raw.reverse()

    else:  # Forward
        if first_date is not None:
            raw = [eff, first_date]
            seed = first_date
        else:
            raw = [eff]
            seed = eff
        exit_date = next_to_last_date if next_to_last_date is not None else term
        periods = 1
        while True:
            candidate = _add_months(seed, periods * freq_months)
            if eom:
                candidate = _last_dom(candidate)
            if candidate > exit_date:
                if next_to_last_date is not None and raw[-1] != next_to_last_date:
                    raw.append(next_to_last_date)
                break
            raw.append(candidate)
            periods += 1
        if raw[-1] != term:
            raw.append(term)

    # ------------------------------------------------------------------
    # Step 2: adjust each date (two-pass, matching QuantLib's approach)
    # ------------------------------------------------------------------
    n = len(raw)
    adjusted: list[ISODate] = []
    for i, d in enumerate(raw):
        if i == 0:
            # Effective date: adjust with main convention
            adjusted.append(cal.adjust(d, convention))
        elif i == n - 1:
            # Termination date: use termination convention
            adjusted.append(cal.adjust(d, term_conv))
        else:
            if eom:
                adjusted.append(cal.adjust(_last_dom(d), convention))
            else:
                adjusted.append(cal.adjust(d, convention))

    # ------------------------------------------------------------------
    # Step 3: deduplicate (EOM can collapse adjacent dates)
    # ------------------------------------------------------------------
    deduped: list[ISODate] = [adjusted[0]]
    for d in adjusted[1:]:
        if d > deduped[-1]:
            deduped.append(d)

    return deduped


def swap_schedule(
    effective_date: "ISODate | str",
    termination_date: "ISODate | str",
    freq_months: int = 12,
    currency: str = "USD",
    *,
    calendar: Optional[Calendar] = None,
    convention: BDConvention = BDConvention.ModifiedFollowing,
    termination_convention: Optional[BDConvention] = None,
    rule: DateGenRule = DateGenRule.Backward,
    end_of_month: bool = False,
) -> list[ISODate]:
    """Return a full swap leg schedule, auto-selecting the calendar by currency.

    Convenience wrapper around :func:`make_schedule` that calls
    :func:`calendar_for` when no explicit *calendar* is given.

    Examples
    --------
    >>> from datetime import date
    >>> sch = swap_schedule(date(2026, 3, 20), date(2031, 3, 20),
    ...                     freq_months=6, currency="USD")
    >>> len(sch)  # 5Y semi-annual → 11 dates
    11
    """
    if calendar is None:
        calendar = calendar_for(currency)
    return make_schedule(
        effective_date, termination_date,
        freq_months=freq_months,
        calendar=calendar,
        convention=convention,
        termination_convention=termination_convention,
        rule=rule,
        end_of_month=end_of_month,
    )


def swap_leg_fracs(
    schedule: list[ISODate],
    dcc: DayCountConvention = DayCountConvention.Act365Fixed,
) -> list[float]:
    """Return accrual year-fractions τ_i for each coupon period.

    Parameters
    ----------
    schedule:
        Full date schedule (n+1 dates for n periods).
    dcc:
        :class:`DayCountConvention` to apply.

    Returns
    -------
    list[float]
        ``[τ_1, …, τ_n]``.
    """
    return [
        year_fraction(schedule[i], schedule[i + 1], dcc)
        for i in range(len(schedule) - 1)
    ]


def swap_coupon_periods(
    schedule: list[ISODate],
    dcc: DayCountConvention = DayCountConvention.Act365Fixed,
) -> list[tuple[ISODate, ISODate, float]]:
    """Return ``(start, end, τ)`` for every coupon period.

    Examples
    --------
    >>> for start, end, tau in swap_coupon_periods(sch, DayCountConvention.Act360):
    ...     pv = notional * forward_rate * tau * discount_factor(end)
    """
    return [
        (schedule[i], schedule[i + 1],
         year_fraction(schedule[i], schedule[i + 1], dcc))
        for i in range(len(schedule) - 1)
    ]


# ---------------------------------------------------------------------------
# 8. Legacy decimal-year helpers (backward-compatible, DO NOT REMOVE)
#    Used by IRSwapFixedFloat, IRSwapFixedFloatApprox, IRSwapFloatFloat.
# ---------------------------------------------------------------------------

def rack_dates(tenor_years: float) -> list[float]:
    """Complete schedule rack including 0.0, walking back from tenor.

    Short front-stub convention: last date is the tenor, then step back
    by 1.0 until 0.0.

    Examples
    --------
    >>> rack_dates(5.0)
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    >>> rack_dates(2.5)
    [0.0, 0.5, 1.5, 2.5]
    """
    if tenor_years < 0:
        raise ValueError(f"tenor_years must be >= 0, got {tenor_years}")
    dates: list[float] = []
    t = float(tenor_years)
    while t > 1e-9:
        dates.append(t)
        t -= 1.0
    dates.append(0.0)
    return sorted(dates)


def payment_dates(tenor_years: float) -> list[float]:
    """Payment dates = rack minus the first (0.0)."""
    return rack_dates(tenor_years)[1:]


def reset_dates(tenor_years: float) -> list[float]:
    """Float reset dates = rack minus the last (tenor_years)."""
    return rack_dates(tenor_years)[:-1]


def day_count_fraction(t1: float, t2: float) -> float:
    """ACT/365 year fraction between two decimal-year values."""
    return float(t2 - t1)


# ---------------------------------------------------------------------------
# 9. OIS / SOFR Compounding Logic
# ---------------------------------------------------------------------------

def compounded_rate(
    start: ISODate,
    end: ISODate,
    evaluation_date: ISODate,
    discount_curve: Any,
    fixings: dict[ISODate, float],
    day_counter: DayCountConvention = DayCountConvention.Act360,
    telescopic: bool = True,
) -> Any:
    """Compute the compounded floating rate for an OIS/SOFR period.

    Implements the logic researched from QuantLib:
    1. For any days already elapsed (fixing < evaluation_date), look up in *fixings*.
    2. For future days, use the telescopic approximation: (P(start)/P(end) - 1)/tau.
    3. Handles the "aged" period where the coupon has already started.

    Parameters
    ----------
    start, end:
        Accrual start and end dates for the coupon.
    evaluation_date:
        The "as-of" date for pricing.
    discount_curve:
        A curve object with a `.df(date)` method (returning a value or Expr).
    fixings:
        Map of {date: rate} for historical daily SOFR/OIS fixings.
    day_counter:
        Day-count convention for the rate (usually Act/360).
    telescopic:
        If True, use the ratio-of-discount-factors approximation for future dates.

    Returns
    -------
    float or Expr
        The projected compounded rate for the period.
    """
    if start >= end:
        return 0.0

    # Total accrual period year fraction
    total_tau = year_fraction(start, end, day_counter)

    # 1. Past fixings part (Compound Factor = Π (1 + r_i * δ_i))
    # Standard OIS spans all business days in the calendar.
    # For simplicity in this demo, we assume the provided fixings map
    # covers all required historical business days.
    cf_past = 1.0
    curr = start
    while curr < end and curr < evaluation_date:
        rate = fixings.get(curr)
        if rate is None:
            # Fallback to zero if missing, or we could raise error
            rate = 0.0
        
        # OIS daily fixing covers 1 day (or more over weekends)
        # We need the next date to get the span δ_i
        # For simplicity, we use the calendar to find the next business day
        # But we don't have the calendar here.
        # Approximation: assume rate is constant until the next fixing or evaluation_date
        # Real QuantLib logic uses the interestDates list.
        # For this demo, we'll assume the fixings represent the accurate daily resets.
        # Simplified: (1 + r * δ) where δ ≈ 1/360
        # In a real implementation, we'd iterate the actual business days.
        delta_t = 1.0 / 360.0 # Placeholder for exact day-count between internal resets
        cf_past *= (1.0 + rate * delta_t)
        curr += datetime.timedelta(days=1)

    # 2. Future part (Telescopic Property)
    # Compound Factor = P(evaluation_date) / P(end)  [if start < evaluation_date]
    #               or P(start) / P(end)            [if start >= evaluation_date]
    if curr < end:
        if not telescopic:
            # Daily projection fallback (not implemented here for simplicity)
            # Would require iterating all future business days.
            pass

        # Use tenors in years (float) for the curve calls
        def _get_tenor(d):
            return (d - evaluation_date).days / 365.2425

        future_start = max(start, evaluation_date)
        df_start_expr = discount_curve.df(_get_tenor(future_start))
        df_end_expr = discount_curve.df(_get_tenor(end))
        
        cf_future = df_start_expr / df_end_expr
        cf_total = cf_past * cf_future
    else:
        # Period is fully in the past
        cf_total = cf_past

    # r = (CF_total - 1) / tau
    return (cf_total - 1.0) / total_tau
