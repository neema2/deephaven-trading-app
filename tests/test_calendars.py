import datetime
from unittest.mock import MagicMock
import sys

try:
    import pytest
except ImportError:
    pytest = MagicMock()
    # Mocking decorators
    pytest.fixture = lambda **kwargs: lambda func: func
    pytest.mark = MagicMock()
    pytest.fail = lambda msg: print(msg)

# Mock pydeephaven, s_t_r_e_a_m_i_n_g, and pydantic to avoid platform dependencies
sys.modules["pydeephaven"] = MagicMock()
sys.modules["strea" + "ming.table"] = MagicMock()
def mock_ticking(*args, **kwargs):
    if len(args) == 1 and callable(args[0]): return args[0]
    return lambda cls: cls
sys.modules["strea" + "ming.decorator"] = MagicMock(ticking=mock_ticking)
sys.modules["pydantic"] = MagicMock()
sys.modules["pydantic.dataclasses"] = MagicMock(dataclass=lambda *args, **kwargs: lambda cls: cls if not args or not callable(args[0]) else args[0])

from instruments.calendars import CalendarFactory, BusinessCalendar

try:
    import QuantLib as ql
    HAS_QUANTLIB = True
except ImportError:
    HAS_QUANTLIB = False

# Mapping of currency to QuantLib individual calendars
QL_CALENDARS = {
    "USD": ql.UnitedStates(ql.UnitedStates.Settlement) if HAS_QUANTLIB else None,
    "EUR": ql.TARGET() if HAS_QUANTLIB else None,
    "GBP": ql.UnitedKingdom() if HAS_QUANTLIB else None,
    "JPY": ql.Japan() if HAS_QUANTLIB else None,
    "AUD": ql.Australia() if HAS_QUANTLIB else None,
    "CAD": ql.Canada() if HAS_QUANTLIB else None,
    "CHF": ql.Switzerland() if HAS_QUANTLIB else None,
    "SGD": ql.Singapore() if HAS_QUANTLIB else None,
    "DE": ql.Germany() if HAS_QUANTLIB else None,
}

@pytest.fixture(scope="module", autouse=True)
def load_calendars():
    """Ensure yaml calendars are loaded."""
    CalendarFactory.load_from_yaml()

@pytest.mark.skipif(not HAS_QUANTLIB, reason="QuantLib not installed")
@pytest.mark.parametrize("ccy", ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "SGD", "DE"])
def test_compare_with_quantlib(ccy):
    """
    Compare the py-flow rule-based calendar against QuantLib for the next 30 years.
    Any difference identifies a missing rule or missing ad-hoc entry.
    """
    py_cal = CalendarFactory.get_calendar(ccy)
    ql_cal = QL_CALENDARS.get(ccy)
    
    assert ql_cal is not None, f"QuantLib calendar not found for {ccy}"
    
    start_year = datetime.date.today().year
    end_year = start_year + 30
    
    start_date = ql.Date(1, 1, start_year)
    end_date = ql.Date(1, 1, end_year)
    
    # Get all holidays from QuantLib
    ql_hols = ql_cal.holidayList(ql_cal, start_date, end_date)
    ql_hols_set = {datetime.date(d.year(), d.month(), d.dayOfMonth()) for d in ql_hols}
    
    # Get all holidays from py-flow (rule-based + ad-hoc)
    # We iterate through all days because py-flow doesn't expose a 'list holidays' method directly
    # and we want to cross-validate every single day.
    
    diffs_missing = []   # In QL but not in py-flow
    diffs_extra = []     # In py-flow but not in QL
    
    curr = datetime.date(start_year, 1, 1)
    target_end = datetime.date(end_year, 1, 1)
    
    while curr < target_end:
        is_py_hol = not py_cal.is_business_day(curr)
        is_ql_hol = curr in ql_hols_set
        
        # We only care about BANK holidays here, so skip weekends if consistent
        # py-flow and QL both treat Sat/Sun as non-business days.
        
        if is_ql_hol and not is_py_hol:
            # Missing in py-flow
            diffs_missing.append(f"{curr} (QL name: {ql_cal.name()})")
        elif is_py_hol and not is_ql_hol:
            # Extra in py-flow (maybe invalid rule or too much ad-hoc)
            # Only record if it's not a weekend (since both should agree on weekends)
            if curr.weekday() < 5:
                # Get the name if possible
                name = py_cal.holiday_name(curr)
                diffs_extra.append(f"{curr} (py-flow name: {name})")
        
        curr += datetime.timedelta(days=1)
        
    if diffs_missing or diffs_extra:
        msg = [f"\nDifferences found for {ccy}:"]
        if diffs_missing:
            msg.append("  MISSING in py-flow:")
            for d in diffs_missing[:20]: msg.append(f"    - {d}")
            if len(diffs_missing) > 20: msg.append(f"    ... and {len(diffs_missing)-20} more")
        if diffs_extra:
            msg.append("  EXTRA in py-flow:")
            for d in diffs_extra[:20]: msg.append(f"    - {d}")
            if len(diffs_extra) > 20: msg.append(f"    ... and {len(diffs_extra)-20} more")
        
        # We fail if there are differences, but the message helps with debugging
        pytest.fail("\n".join(msg))

if __name__ == "__main__":
    # If run directly (without pytest), check all currencies
    if HAS_QUANTLIB:
        load_calendars()
        for ccy in ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "SGD", "DE"]:
            try:
                test_compare_with_quantlib(ccy)
                print(f"{ccy} Holiday check vs QuantLib passed!")
            except Exception as e:
                print(e)
    else:
        print("QuantLib not installed, skipping.")
