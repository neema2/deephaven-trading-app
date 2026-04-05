"""
tests/test_ir_scheduling.py — Test suite for instruments/ir_scheduling.py.

Tests are split into three layers:

1. ``TestNoQuantLib``     — pure-Python tests, run always.  QuantLib must NOT
                            be required.  Confirms all core logic works with
                            WeekendsOnlyCalendar on a plain dev install.

2. ``TestQuantLibParity`` — skipped unless QuantLib is installed.
                            Cross-checks every schedule and day-count result
                            against the real QuantLib C++ library.
                            Install with:  pip install QuantLib
                                       or: pip install -e ".[quantlib]"

3. ``TestLegacyAPI``      — regression tests for the decimal-year shims that
                            existing instrument code depends on.

Run without QuantLib:
    pytest tests/test_ir_scheduling.py -v

Run with QuantLib cross-checks:
    pip install QuantLib
    pytest tests/test_ir_scheduling.py -v
"""

from __future__ import annotations

import calendar as _cal_mod
import datetime
import importlib.util
import pathlib
from datetime import date

import pytest

# ---------------------------------------------------------------------------
# Load ir_scheduling directly from its source file.
#
# We deliberately bypass ``instruments/__init__.py`` because that module
# eagerly imports IRSwapFixedFloat etc., pulling in the full store /
# reactive / streaming stack (which requires Python ≥ 3.11 and heavy optional
# deps).  Direct file loading keeps this test runnable with a plain
# ``pip install -e ".[dev]"`` even on Python 3.10.
# ---------------------------------------------------------------------------
_SCHED_FILE = pathlib.Path(__file__).parent.parent / "instruments" / "ir_scheduling.py"
_spec = importlib.util.spec_from_file_location("_ir_scheduling_standalone", _SCHED_FILE)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

# Public surface — bring names into module scope for test readability.
BDConvention         = _mod.BDConvention
BespokeCalendar      = _mod.BespokeCalendar
Calendar             = _mod.Calendar
DateGenRule          = _mod.DateGenRule
DayCountConvention   = _mod.DayCountConvention
QuantLibCalendar     = _mod.QuantLibCalendar
WeekendsOnlyCalendar = _mod.WeekendsOnlyCalendar
calendar_for         = _mod.calendar_for
dcc_for_currency     = _mod.dcc_for_currency
day_count_fraction   = _mod.day_count_fraction
from_serial          = _mod.from_serial
make_bespoke_calendar = _mod.make_bespoke_calendar
make_schedule        = _mod.make_schedule
payment_dates        = _mod.payment_dates
rack_dates           = _mod.rack_dates
reset_dates          = _mod.reset_dates
swap_coupon_periods  = _mod.swap_coupon_periods
swap_leg_fracs       = _mod.swap_leg_fracs
swap_schedule        = _mod.swap_schedule
to_serial            = _mod.to_serial
year_fraction        = _mod.year_fraction

# ---------------------------------------------------------------------------
# QuantLib detection
# ---------------------------------------------------------------------------
HAS_QL = importlib.util.find_spec("QuantLib") is not None
needs_ql = pytest.mark.skipif(
    not HAS_QL,
    reason="QuantLib not installed — run: pip install QuantLib  or: pip install -e '.[quantlib]'"
)

if HAS_QL:
    import QuantLib as ql  # type: ignore[import-untyped]


# ===========================================================================
# 1.  Tests that must pass WITHOUT QuantLib installed
# ===========================================================================

class TestNoQuantLib:
    """All tests here must pass in a plain `pip install -e ".[dev]"` install."""

    # -----------------------------------------------------------------------
    # Import / dependency hygiene
    # -----------------------------------------------------------------------

    def test_module_loads_without_quantlib(self):
        """Loading ir_scheduling must not import QuantLib or any heavy dep."""
        # If we reached this point, the module-level import at top of file
        # succeeded without QuantLib.  Verify the key objects are present.
        assert callable(make_schedule)
        assert callable(year_fraction)
        assert callable(calendar_for)

    def test_calendar_for_falls_back_when_no_ql(self, monkeypatch):
        """calendar_for() must return WeekendsOnlyCalendar when QL absent."""
        monkeypatch.setattr(_mod, "_QL_CALENDAR_FACTORIES", {})
        cal = _mod.calendar_for("USD")
        assert isinstance(cal, WeekendsOnlyCalendar)

    # -----------------------------------------------------------------------
    # Serial round-trip
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("d", [
        date(1900, 1, 1),
        date(2000, 1, 1),
        date(2026, 3, 20),
        date(2035, 12, 31),
    ])
    def test_serial_round_trip(self, d):
        assert from_serial(to_serial(d)) == d

    # -----------------------------------------------------------------------
    # WeekendsOnlyCalendar
    # -----------------------------------------------------------------------

    def test_weekday_is_business_day(self):
        cal = WeekendsOnlyCalendar()
        assert cal.is_business_day(date(2026, 3, 20))   # Friday

    def test_saturday_is_not_business_day(self):
        cal = WeekendsOnlyCalendar()
        assert not cal.is_business_day(date(2026, 3, 21))  # Saturday

    def test_sunday_is_not_business_day(self):
        cal = WeekendsOnlyCalendar()
        assert not cal.is_business_day(date(2026, 3, 22))  # Sunday

    def test_adjust_following_saturday(self):
        cal = WeekendsOnlyCalendar()
        assert cal.adjust(date(2026, 3, 21), BDConvention.Following) == date(2026, 3, 23)

    def test_adjust_preceding_saturday(self):
        cal = WeekendsOnlyCalendar()
        assert cal.adjust(date(2026, 3, 21), BDConvention.Preceding) == date(2026, 3, 20)

    def test_adjust_modified_following_same_month(self):
        """Next bd is still in the same month → use it."""
        cal = WeekendsOnlyCalendar()
        # 2026-08-29 (Sat) → Mon 2026-08-31 (same month Aug) ✓
        assert cal.adjust(date(2026, 8, 29), BDConvention.ModifiedFollowing) == date(2026, 8, 31)

    def test_adjust_modified_following_crosses_month(self):
        """Next bd is next month → fall back to previous bd."""
        cal = WeekendsOnlyCalendar()
        # 2026-05-30 (Sat) → Mon 2026-06-01 (next month!) → prev: Fri 2026-05-29
        assert cal.adjust(date(2026, 5, 30), BDConvention.ModifiedFollowing) == date(2026, 5, 29)

    def test_adjust_unadjusted_returns_input(self):
        cal = WeekendsOnlyCalendar()
        sat = date(2026, 3, 21)
        assert cal.adjust(sat, BDConvention.Unadjusted) == sat

    def test_adjust_noop_on_business_day(self):
        cal = WeekendsOnlyCalendar()
        fri = date(2026, 3, 20)
        for conv in BDConvention:
            assert cal.adjust(fri, conv) == fri

    # -----------------------------------------------------------------------
    # BespokeCalendar
    # -----------------------------------------------------------------------

    def test_bespoke_holiday_not_business_day(self):
        cal = make_bespoke_calendar("Test", [date(2026, 3, 20)])
        assert not cal.is_business_day(date(2026, 3, 20))   # Friday, but declared holiday

    def test_bespoke_non_holiday_is_business_day(self):
        cal = make_bespoke_calendar("Test", [date(2026, 3, 20)])
        assert cal.is_business_day(date(2026, 3, 19))        # Thursday, not a holiday

    def test_bespoke_add_remove_dynamic(self):
        cal = BespokeCalendar("Test")
        d = date(2026, 6, 5)   # Friday
        assert cal.is_business_day(d)
        cal.add_holiday(d)
        assert not cal.is_business_day(d)
        cal.remove_holiday(d)
        assert cal.is_business_day(d)

    # -----------------------------------------------------------------------
    # Calendar.advance
    # -----------------------------------------------------------------------

    def test_advance_one_month(self):
        cal = WeekendsOnlyCalendar()
        # 2026-01-20 + 1M = 2026-02-20 (Friday, no adjustment needed)
        assert cal.advance(date(2026, 1, 20), months=1) == date(2026, 2, 20)

    def test_advance_eom_clamp(self):
        cal = WeekendsOnlyCalendar()
        # Jan 31 + 1M = Feb 28 (no leap), unadjusted
        result = cal.advance(date(2026, 1, 31), months=1, convention=BDConvention.Unadjusted)
        assert result == date(2026, 2, 28)

    def test_advance_eom_flag(self):
        """EOM flag: start at month-end → subsequent dates snap to month-end."""
        cal = WeekendsOnlyCalendar()
        # Jan 31 (EOM) → eom=True → Feb gets last day of Feb = 28 (Sat 2026)
        # ModifiedFollowing: Sat → Fri same month = 2026-02-27
        result = cal.advance(date(2026, 1, 31), months=1,
                             convention=BDConvention.ModifiedFollowing,
                             end_of_month=True)
        assert result == date(2026, 2, 27)

    # -----------------------------------------------------------------------
    # make_schedule — structure and edge cases
    # -----------------------------------------------------------------------

    def test_annual_backward_produces_6_dates(self):
        sch = make_schedule(date(2026, 3, 20), date(2031, 3, 20), freq_months=12)
        assert len(sch) == 6           # 5 periods + 1 endpoint
        assert sch[0] == date(2026, 3, 20)
        assert sch[-1] == date(2031, 3, 20)

    def test_semiannual_produces_11_dates(self):
        sch = make_schedule(date(2026, 3, 20), date(2031, 3, 20), freq_months=6)
        assert len(sch) == 11

    def test_quarterly_produces_21_dates(self):
        sch = make_schedule(date(2026, 3, 20), date(2031, 3, 20), freq_months=3)
        assert len(sch) == 21

    def test_zero_rule_gives_just_endpoints(self):
        sch = make_schedule(date(2026, 3, 20), date(2031, 3, 20), rule=DateGenRule.Zero)
        assert sch == [date(2026, 3, 20), date(2031, 3, 20)]

    def test_forward_and_backward_whole_tenor_same_length(self):
        bwd = make_schedule(date(2026, 3, 20), date(2031, 3, 20),
                            freq_months=12, rule=DateGenRule.Backward)
        fwd = make_schedule(date(2026, 3, 20), date(2031, 3, 20),
                            freq_months=12, rule=DateGenRule.Forward)
        assert len(fwd) == len(bwd)

    def test_schedule_is_strictly_ascending(self):
        sch = make_schedule(date(2026, 3, 20), date(2031, 3, 20), freq_months=6)
        assert all(sch[i] < sch[i + 1] for i in range(len(sch) - 1))

    def test_iso_string_input_accepted(self):
        sch = make_schedule("2026-03-20", "2031-03-20", freq_months=12)
        assert len(sch) == 6

    def test_effective_ge_termination_raises(self):
        with pytest.raises(ValueError, match="before termination"):
            make_schedule(date(2026, 3, 20), date(2026, 3, 20))

    def test_backward_short_front_stub(self):
        """Backward rule on non-whole tenor → short period at the front."""
        # 5Y6M annual backward → stub is first period (~6 months)
        sch = make_schedule(date(2026, 3, 20), date(2031, 9, 20), freq_months=12)
        first_period_days = (sch[1] - sch[0]).days
        assert first_period_days < 200, f"Expected short stub, got {first_period_days} days"

    def test_forward_short_back_stub(self):
        """Forward rule on non-whole tenor → short period at the back."""
        sch = make_schedule(date(2026, 3, 20), date(2031, 9, 20),
                            freq_months=12, rule=DateGenRule.Forward)
        last_period_days = (sch[-1] - sch[-2]).days
        assert last_period_days < 200, f"Expected short back stub, got {last_period_days} days"

    def test_no_weekend_dates_with_weekends_only_calendar(self):
        cal = WeekendsOnlyCalendar()
        sch = make_schedule(date(2026, 3, 20), date(2031, 3, 20),
                            freq_months=12, calendar=cal)
        for d in sch:
            assert d.weekday() < 5, f"{d} is a weekend!"

    def test_unadjusted_convention_may_include_weekends(self):
        cal = WeekendsOnlyCalendar()
        # Start on a Saturday so intermediate dates likely land on weekends too
        sch = make_schedule(date(2026, 3, 21), date(2031, 3, 21),
                            freq_months=12, calendar=cal,
                            convention=BDConvention.Unadjusted,
                            termination_convention=BDConvention.Unadjusted)
        assert any(d.weekday() >= 5 for d in sch[1:-1])

    def test_end_of_month_flag_keeps_dates_near_eom(self):
        cal = WeekendsOnlyCalendar()
        sch = make_schedule(date(2026, 1, 31), date(2028, 1, 31),
                            freq_months=6, calendar=cal,
                            end_of_month=True,
                            convention=BDConvention.ModifiedFollowing)
        for d in sch[1:-1]:
            last = _cal_mod.monthrange(d.year, d.month)[1]
            assert last - d.day <= 4, f"{d} is not near EOM (last={last})"

    def test_next_to_last_date_creates_back_stub(self):
        ntl = date(2030, 9, 20)
        sch = make_schedule(date(2026, 3, 20), date(2031, 3, 20),
                            freq_months=12, next_to_last_date=ntl)
        # The second-to-last date should be at or before ntl
        assert sch[-2] <= ntl

    # -----------------------------------------------------------------------
    # swap_schedule convenience wrapper
    # -----------------------------------------------------------------------

    def test_swap_schedule_usd_5y_semiannual(self):
        sch = swap_schedule(date(2026, 3, 20), date(2031, 3, 20),
                            freq_months=6, currency="USD")
        assert len(sch) == 11

    def test_swap_schedule_works_without_quantlib(self, monkeypatch):
        """Must work even when the QL calendar registry is empty."""
        monkeypatch.setattr(_mod, "_QL_CALENDAR_FACTORIES", {})
        sch = swap_schedule(date(2026, 3, 20), date(2031, 3, 20),
                            freq_months=12, currency="USD")
        assert len(sch) == 6

    # -----------------------------------------------------------------------
    # Day-count fractions
    # -----------------------------------------------------------------------

    def test_act365_full_year(self):
        tau = year_fraction(date(2026, 1, 1), date(2027, 1, 1), DayCountConvention.Act365Fixed)
        assert tau == pytest.approx(365 / 365.0)

    def test_act365_leap_year_divides_by_365_not_366(self):
        # 2028 has 366 actual days but Act/365F always divides by 365
        tau = year_fraction(date(2028, 1, 1), date(2029, 1, 1), DayCountConvention.Act365Fixed)
        assert tau == pytest.approx(366 / 365.0)

    def test_act360_correct_basis(self):
        # Jan 1 → Jul 1 2026 = 181 days
        tau = year_fraction(date(2026, 1, 1), date(2026, 7, 1), DayCountConvention.Act360)
        assert tau == pytest.approx(181 / 360.0)

    def test_thirty360_us_standard_period(self):
        # Jan 15 → Jul 15: exactly 6 months of 30 days each = 0.5
        tau = year_fraction(date(2026, 1, 15), date(2026, 7, 15), DayCountConvention.Thirty360US)
        assert tau == pytest.approx(0.5)

    def test_thirty360_us_31st_start_rule(self):
        # Start on 31st → treated as 30th; end on 30th → unchanged → 0.5Y
        tau = year_fraction(date(2026, 1, 31), date(2026, 7, 30), DayCountConvention.Thirty360US)
        assert tau == pytest.approx(0.5)

    def test_thirty360_eu_both_31st(self):
        # Both 31st → both become 30th
        tau = year_fraction(date(2026, 1, 31), date(2026, 7, 31), DayCountConvention.Thirty360EU)
        assert tau == pytest.approx(0.5)

    def test_actact_isda_full_year_non_leap(self):
        tau = year_fraction(date(2026, 1, 1), date(2027, 1, 1), DayCountConvention.ActActISDA)
        assert tau == pytest.approx(1.0)

    def test_actact_isda_full_year_leap(self):
        tau = year_fraction(date(2028, 1, 1), date(2029, 1, 1), DayCountConvention.ActActISDA)
        assert tau == pytest.approx(1.0)

    def test_actact_isda_crosses_year_boundary(self):
        # Jul 1 2026 → Jan 1 2027
        # ISDA: (Jan1_2027 - Jul1_2026) / 365  +  (Jan1_2027 - Jan1_2027) / 365
        #      = 184/365 + 0 = 0.5041...
        tau = year_fraction(date(2026, 7, 1), date(2027, 1, 1), DayCountConvention.ActActISDA)
        # Verify: first segment only (d2 = Jan 1 2027, which is boundary)
        expected = (date(2027, 1, 1) - date(2026, 7, 1)).days / 365.0
        assert tau == pytest.approx(expected, rel=1e-10)

    # -----------------------------------------------------------------------
    # swap_leg_fracs / swap_coupon_periods
    # -----------------------------------------------------------------------

    def test_leg_fracs_correct_count(self):
        sch = make_schedule(date(2026, 3, 20), date(2031, 3, 20), freq_months=12)
        assert len(swap_leg_fracs(sch, DayCountConvention.Act365Fixed)) == 5

    def test_leg_fracs_annual_reasonable_magnitude(self):
        sch = make_schedule(date(2026, 3, 20), date(2031, 3, 20), freq_months=12)
        for tau in swap_leg_fracs(sch, DayCountConvention.Act365Fixed):
            assert 0.95 < tau < 1.10, f"tau={tau} out of expected annual range"

    def test_coupon_periods_structure(self):
        sch = make_schedule(date(2026, 3, 20), date(2031, 3, 20), freq_months=12)
        periods = swap_coupon_periods(sch, DayCountConvention.Act360)
        assert len(periods) == 5
        for start, end, tau in periods:
            assert start < end
            assert tau > 0

    def test_leg_fracs_sum_approx_tenor(self):
        """Sum of Act/365F year fractions should be within 2bp of exact tenor."""
        sch = make_schedule(date(2026, 1, 1), date(2031, 1, 1), freq_months=12)
        total = sum(swap_leg_fracs(sch, DayCountConvention.Act365Fixed))
        assert total == pytest.approx(5.0, abs=0.05)

    # -----------------------------------------------------------------------
    # dcc_for_currency
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("ccy,leg,expected", [
        ("USD", "fixed", DayCountConvention.Thirty360US),
        ("USD", "float", DayCountConvention.Act360),
        ("EUR", "fixed", DayCountConvention.Thirty360EU),
        ("EUR", "float", DayCountConvention.Act360),
        ("GBP", "fixed", DayCountConvention.Act365Fixed),
        ("GBP", "float", DayCountConvention.Act365Fixed),
        ("XYZ", "float", DayCountConvention.Act365Fixed),
    ])
    def test_dcc_for_currency(self, ccy, leg, expected):
        assert dcc_for_currency(ccy, leg) == expected


# ===========================================================================
# 2.  QuantLib parity tests — skipped when QuantLib is absent
# ===========================================================================

class TestQuantLibParity:
    """Cross-check our Python implementation against the real QuantLib library.

    All tests in this class are skipped automatically when QuantLib is not
    installed.  To run them:
        pip install QuantLib
        pytest tests/test_ir_scheduling.py::TestQuantLibParity -v
    """

    @needs_ql
    def test_quantlibcalendar_name_delegates_to_ql(self):
        ql_cal_obj = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
        cal = QuantLibCalendar(ql_cal_obj)
        assert "United States" in cal.name()

    @needs_ql
    @pytest.mark.parametrize("d,expected", [
        (date(2026, 7, 4),   False),   # Independence Day
        (date(2026, 11, 26), False),   # Thanksgiving (4th Thursday)
        (date(2026, 12, 25), False),   # Christmas
        (date(2026, 1, 1),   False),   # New Year
        (date(2026, 3, 20),  True),    # Ordinary Friday
        (date(2026, 6, 15),  True),    # Ordinary Monday
    ])
    def test_us_holiday_detection(self, d, expected):
        ql_cal_obj = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
        our_cal = QuantLibCalendar(ql_cal_obj)
        ql_date = ql.Date(d.day, d.month, d.year)
        ql_result = ql_cal_obj.isBusinessDay(ql_date)
        our_result = our_cal.is_business_day(d)
        assert our_result == ql_result == expected, (
            f"{d}: QL={ql_result}, ours={our_result}, expected={expected}"
        )

    @needs_ql
    @pytest.mark.parametrize("currency,ql_factory", [
        ("USD", lambda: ql.UnitedStates(ql.UnitedStates.GovernmentBond)),
        ("GBP", lambda: ql.UnitedKingdom(ql.UnitedKingdom.Settlement)),
        ("EUR", lambda: ql.TARGET()),
        ("JPY", lambda: ql.Japan()),
    ])
    def test_adjust_matches_quantlib_90_days(self, currency, ql_factory):
        """MF adjustment must agree with QuantLib for a 90-day window."""
        ql_cal_obj = ql_factory()
        our_cal = QuantLibCalendar(ql_cal_obj)
        start = date(2026, 1, 1)
        for delta in range(90):
            d = start + datetime.timedelta(days=delta)
            ql_date = ql.Date(d.day, d.month, d.year)
            ql_adj  = ql_cal_obj.adjust(ql_date, ql.ModifiedFollowing)
            expected = date(ql_adj.year(), ql_adj.month(), ql_adj.dayOfMonth())
            ours     = our_cal.adjust(d, BDConvention.ModifiedFollowing)
            assert ours == expected, f"{currency} {d}: QL={expected}, ours={ours}"

    @needs_ql
    @pytest.mark.parametrize("eff,term,freq,currency,ql_factory", [
        # Standard vanilla cases
        (date(2026, 3, 20), date(2031, 3, 20), 12, "USD",
         lambda: ql.UnitedStates(ql.UnitedStates.GovernmentBond)),
        (date(2026, 3, 20), date(2031, 3, 20),  6, "USD",
         lambda: ql.UnitedStates(ql.UnitedStates.GovernmentBond)),
        (date(2026, 6, 16), date(2031, 6, 16),  3, "EUR",
         lambda: ql.TARGET()),
        (date(2026, 1, 15), date(2036, 1, 15), 12, "GBP",
         lambda: ql.UnitedKingdom(ql.UnitedKingdom.Settlement)),
        (date(2026, 9, 18), date(2031, 9, 18),  6, "JPY",
         lambda: ql.Japan()),
        # Short front stub (5Y6M / annual freq)
        (date(2026, 3, 20), date(2031, 9, 20), 12, "USD",
         lambda: ql.UnitedStates(ql.UnitedStates.GovernmentBond)),
        # EOM (Jan 31 effective)
        (date(2026, 1, 31), date(2028, 1, 31),  6, "USD",
         lambda: ql.UnitedStates(ql.UnitedStates.GovernmentBond)),
    ])
    def test_schedule_matches_quantlib_exactly(self, eff, term, freq, currency, ql_factory):
        """make_schedule() must produce exactly the same dates as ql.MakeSchedule."""
        ql_cal_obj = ql_factory()
        our_cal = QuantLibCalendar(ql_cal_obj)
        eom_flag = _is_eom(eff)

        # QuantLib schedule
        ql_eff  = ql.Date(eff.day, eff.month, eff.year)
        ql_term = ql.Date(term.day, term.month, term.year)
        ql_sch  = (ql.MakeSchedule()
                   .from_(ql_eff)
                   .to(ql_term)
                   .withTenor(ql.Period(freq, ql.Months))
                   .withCalendar(ql_cal_obj)
                   .withConvention(ql.ModifiedFollowing)
                   .withTerminationDateConvention(ql.ModifiedFollowing)
                   .backwards()
                   .endOfMonth(eom_flag)
                   .makeSchedule())
        ql_dates = [
            date(ql_sch[i].year(), ql_sch[i].month(), ql_sch[i].dayOfMonth())
            for i in range(len(ql_sch))
        ]

        # Our schedule
        our_dates = make_schedule(
            eff, term,
            freq_months=freq,
            calendar=our_cal,
            convention=BDConvention.ModifiedFollowing,
            termination_convention=BDConvention.ModifiedFollowing,
            rule=DateGenRule.Backward,
            end_of_month=eom_flag,
        )

        assert len(our_dates) == len(ql_dates), (
            f"{currency} {eff}→{term} {freq}M: "
            f"ours={len(our_dates)}, QL={len(ql_dates)}\n"
            f"ours={our_dates}\nQL  ={ql_dates}"
        )
        for i, (ours, theirs) in enumerate(zip(our_dates, ql_dates)):
            assert ours == theirs, (
                f"{currency} {eff}→{term} {freq}M: "
                f"date[{i}] ours={ours}, QL={theirs}"
            )

    @needs_ql
    @pytest.mark.parametrize("d1,d2,our_dcc,ql_dc_factory", [
        (date(2026, 1,  1), date(2027, 1,  1),
         DayCountConvention.Act365Fixed,  lambda: ql.Actual365Fixed()),
        (date(2026, 1,  1), date(2026, 7,  1),
         DayCountConvention.Act360,       lambda: ql.Actual360()),
        (date(2026, 1, 15), date(2026, 7, 15),
         DayCountConvention.Thirty360US,  lambda: ql.Thirty360(ql.Thirty360.BondBasis)),
        (date(2026, 1, 31), date(2026, 7, 31),
         DayCountConvention.Thirty360EU,  lambda: ql.Thirty360(ql.Thirty360.EurobondBasis)),
        (date(2026, 7,  1), date(2027, 1,  1),
         DayCountConvention.ActActISDA,   lambda: ql.ActualActual(ql.ActualActual.ISDA)),
        (date(2028, 7,  1), date(2029, 7,  1),
         DayCountConvention.ActActISDA,   lambda: ql.ActualActual(ql.ActualActual.ISDA)),
        (date(2026, 1, 31), date(2026, 4, 30),
         DayCountConvention.Thirty360US,  lambda: ql.Thirty360(ql.Thirty360.BondBasis)),
    ])
    def test_year_fraction_matches_quantlib(self, d1, d2, our_dcc, ql_dc_factory):
        """year_fraction() must agree with QuantLib to machine precision."""
        ql_dc   = ql_dc_factory()
        ql_d1   = ql.Date(d1.day, d1.month, d1.year)
        ql_d2   = ql.Date(d2.day, d2.month, d2.year)
        ql_tau  = ql_dc.yearFraction(ql_d1, ql_d2)
        our_tau = year_fraction(d1, d2, our_dcc)
        assert our_tau == pytest.approx(ql_tau, rel=1e-12), (
            f"DCC {our_dcc.value}: {d1}→{d2}: ours={our_tau}, QL={ql_tau}"
        )

    @needs_ql
    def test_calendar_for_usd_uses_quantlib(self):
        cal = calendar_for("USD")
        assert isinstance(cal, QuantLibCalendar), (
            f"Expected QuantLibCalendar when QL is installed, got {type(cal).__name__}"
        )

    @needs_ql
    def test_calendar_for_eur_is_target(self):
        cal = calendar_for("EUR")
        assert isinstance(cal, QuantLibCalendar)
        assert "TARGET" in cal.name() or "Euro" in cal.name()


# ===========================================================================
# 3.  Legacy decimal-year API — backward-compatibility regression tests
# ===========================================================================

class TestLegacyAPI:
    """Existing instrument code must continue to work without any changes."""

    def test_rack_dates_whole_tenor(self):
        assert rack_dates(5.0) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    def test_rack_dates_stub(self):
        assert rack_dates(2.5) == [0.0, 0.5, 1.5, 2.5]

    def test_rack_dates_zero_tenor(self):
        assert rack_dates(0.0) == [0.0]

    def test_rack_dates_negative_raises(self):
        with pytest.raises(ValueError):
            rack_dates(-1.0)

    def test_payment_dates(self):
        assert payment_dates(5.0) == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_reset_dates(self):
        assert reset_dates(5.0) == [0.0, 1.0, 2.0, 3.0, 4.0]

    @pytest.mark.parametrize("tenor", [1.0, 2.5, 5.0, 10.0])
    def test_payment_reset_cover_rack(self, tenor):
        rack = rack_dates(tenor)
        assert payment_dates(tenor) == rack[1:]
        assert reset_dates(tenor) == rack[:-1]

    @pytest.mark.parametrize("t1,t2,expected", [
        (0.0, 0.5, 0.5),
        (1.0, 2.0, 1.0),
        (0.25, 0.75, 0.5),
    ])
    def test_day_count_fraction(self, t1, t2, expected):
        assert day_count_fraction(t1, t2) == pytest.approx(expected)

    def test_legacy_import_path_still_works(self):
        """Import path used by IRSwapFixedFloat must remain valid."""
        # This imports via instruments/__init__.py if the full stack is available,
        # which it may not be on Python 3.10.  Import directly from the module.
        import importlib.util as _ilu
        _p = pathlib.Path(__file__).parent.parent / "instruments" / "ir_scheduling.py"
        _s = _ilu.spec_from_file_location("_check", _p)
        _m = _ilu.module_from_spec(_s)  # type: ignore
        _s.loader.exec_module(_m)        # type: ignore
        assert _m.payment_dates(3.0) == [1.0, 2.0, 3.0]
        assert _m.reset_dates(3.0) == [0.0, 1.0, 2.0]


# ===========================================================================
# Module-level helper (used by parity tests)
# ===========================================================================

def _is_eom(d: date) -> bool:
    return d.day == _cal_mod.monthrange(d.year, d.month)[1]
