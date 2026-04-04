import datetime
from instruments.calendars import CalendarFactory
from instruments.ir_scheduling import calendar_for, BDConvention

def test_yaml_calendar_loading():
    """Verify that YAML-based calendars are loaded and correctly identify holidays."""
    # Ensure calendars.yaml is loaded
    CalendarFactory.load_from_yaml("calendars.yaml")
    
    # USD Calendar (July 4 is fixed holiday)
    usd_cal = calendar_for("USD")
    ind_day = datetime.date(2026, 7, 4)
    assert not usd_cal.is_business_day(ind_day), "July 4 should be a holiday in USD"
    
    # Check weekend (July 5, 2026 is Sunday)
    sunday = datetime.date(2026, 7, 5)
    assert not usd_cal.is_business_day(sunday), "Sunday should not be a business day"
    
    # Check business day (July 6, 2026 is Monday)
    monday = datetime.date(2026, 7, 6)
    assert usd_cal.is_business_day(monday), "Monday July 6 should be a business day"

def test_bd_adjustment():
    """Verify business day adjustment logic (ModifiedFollowing)."""
    CalendarFactory.load_from_yaml("calendars.yaml")
    usd_cal = calendar_for("USD")
    
    # July 4, 2026 is a Saturday. 
    # Following should be Monday July 6.
    ind_day = datetime.date(2026, 7, 4)
    adj_following = usd_cal.adjust(ind_day, BDConvention.Following)
    assert adj_following == datetime.date(2026, 7, 6)
    
    # Dec 25, 2026 is a Friday.
    # Dec 26 is Saturday.
    # Following for Dec 25 is Dec 28 (Monday).
    christmas = datetime.date(2026, 12, 25)
    adj_following_christmas = usd_cal.adjust(christmas, BDConvention.Following)
    # Wait, Dec 25 is Friday, so it is a holiday. 
    # Dec 26 Sat, Dec 27 Sun. 
    # Dec 28 Mon. Correct.
    assert adj_following_christmas == datetime.date(2026, 12, 28)

def test_ad_hoc_holidays_from_yaml():
    """Verify that ad-hoc holidays from YAML are correctly honored."""
    CalendarFactory.load_from_yaml("calendars.yaml")
    usd_cal = calendar_for("USD")
    
    # From calendars.yaml: MLK Day 2024-01-15 is ad-hoc
    mlk_day = datetime.date(2024, 1, 15)
    assert not usd_cal.is_business_day(mlk_day), "MLK Day 2024-01-15 should be an ad-hoc holiday"

if __name__ == "__main__":
    test_yaml_calendar_loading()
    test_bd_adjustment()
    test_ad_hoc_holidays_from_yaml()
    print("Holiday system tests passed!")
