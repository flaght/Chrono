# -*- coding: utf-8 -*-
import re, pdb
from collections import defaultdict
from ..DateUtilities.Calendar import Calendar
from ..DateUtilities.Date import Date
from ..DateUtilities.Date import check_date
from ..DateUtilities.Period import Period
from ..DateUtilities.Period import check_period
from ..DateUtilities.Schedule import Schedule
from ..Enums import BizDayConventions
from ..Enums import DateGeneration
from ..Enums import TimeUnits


def _generate_halfyearly_calendar(dates):
    calendar = defaultdict(list)

    for date in dates:
        year = date.year
        halfyear = 1 if date.month <= 6 else 2  

        calendar[(year, halfyear)].append(date)

    return calendar

def _generate_weekly_calendar(dates):
    calendar = defaultdict(list)

    for date in dates:
        year, week_num, _ = date.isocalendar()

        calendar[(year, week_num)].append(date)

    return calendar

def _generate_monthly_calendar(dates):
    calendar = defaultdict(list)

    for date in dates:
        year = date.year
        month = date.month

        calendar[(year, month)].append(date)

    return calendar


def _generate_quarterly_calendar(dates):
    calendar = defaultdict(list)

    for date in dates:
        year = date.year
        quarter = (date.month - 1) // 3 + 1

        calendar[(year, quarter)].append(date)

    return calendar

def _get_all_nth_dates(calendar, n):
    nth_dates = []

    for dates in calendar.values():
        if n <= len(dates):
            nth_dates.append(dates[n - 1]) if n > 0 else nth_dates.append(dates[n])
    return nth_dates



calendar_mapping = {'w':_generate_weekly_calendar,'m':_generate_monthly_calendar,
                    'q':_generate_quarterly_calendar,'y':_generate_halfyearly_calendar}



def isBizDay(holidayCenter, ref):
    cal = Calendar(holidayCenter)
    ref = check_date(ref)
    return cal.isBizDay(ref)


def datesList(fromDate, toDate):
    fromDate = check_date(fromDate)
    toDate = check_date(toDate)
    return [
        Date.fromExcelSerialNumber(serial).toDateTime()
        for serial in range(fromDate.serialNumber, toDate.serialNumber + 1)
    ]


def bizDatesList(holidayCenter, fromDate, toDate):
    cal = Calendar(holidayCenter)
    fromDate = check_date(fromDate)
    toDate = check_date(toDate)
    return [d.toDateTime() for d in cal.bizDatesList(fromDate, toDate)]


def holDatesList(holidayCenter, fromDate, toDate, includeWeekend=True):
    cal = Calendar(holidayCenter)
    fromDate = check_date(fromDate)
    toDate = check_date(toDate)
    return [
        d.toDateTime()
        for d in cal.holDatesList(fromDate, toDate, includeWeekend)
    ]


def advanceDate(referenceDate, period):
    d = check_date(referenceDate) + period
    return d.toDateTime()


def adjustDateByCalendar(holidayCenter,
                         referenceDate,
                         convention=BizDayConventions.Following):
    cal = Calendar(holidayCenter)
    refer = check_date(referenceDate)
    return cal.adjustDate(refer, convention).toDateTime()


def advanceDateByCalendar(holidayCenter,
                          referenceDate,
                          period,
                          convention=BizDayConventions.Following):
    cal = Calendar(holidayCenter)
    refer = check_date(referenceDate)
    period = check_period(period)
    return cal.advanceDate(refer, period, convention).toDateTime()


def nthWeekDay(nth, dayOfWeek, month, year):
    date = Date.nthWeekday(nth, dayOfWeek, month, year)
    return date.toDateTime()



def makeSchedule(firstDate,
                 endDate,
                 tenor,
                 calendar='NullCalendar',
                 dateRule=BizDayConventions.Following,
                 dateGenerationRule=DateGeneration.Forward):
    cal = Calendar(calendar)
    firstDate = check_date(firstDate)
    endDate = check_date(endDate)
    tenor = check_period(tenor)

    if tenor.units() == TimeUnits.BDays:
        schedule = []
        if dateGenerationRule == DateGeneration.Forward:
            d = cal.adjustDate(firstDate, dateRule)
            while d <= endDate:
                schedule.append(d)
                d = cal.advanceDate(d, tenor, dateRule)
        elif dateGenerationRule == DateGeneration.Backward:
            d = cal.adjustDate(endDate, dateRule)
            while d >= firstDate:
                schedule.append(d)
                d = cal.advanceDate(d, -tenor, dateRule)
            schedule = sorted(schedule)
    else:
        schedule = Schedule(firstDate,
                            endDate,
                            tenor,
                            cal,
                            convention=dateRule,
                            dateGenerationRule=dateGenerationRule)
    return [d.toDateTime() for d in schedule]



def silceSchedule(firstDate,
                 endDate,
                 tenor,
                 calendar='NullCalendar',
                 dateRule=BizDayConventions.Following,
                 dateGenerationRule=DateGeneration.Forward):

    dates = makeSchedule(firstDate=firstDate, endDate=advanceDateByCalendar('china.sse', endDate, '10b'),
                 tenor='1b', calendar=calendar,dateRule=dateRule,
                 dateGenerationRule=dateGenerationRule)
    parts = re.findall(r'(-?\d+|[a-zA-Z])', tenor)
    calendar_func = calendar_mapping[parts[1]]
    calendar_dates = calendar_func(dates)

    return _get_all_nth_dates(calendar_dates, int(parts[0]))[:-1]

def carveSchedule(firstDate,
                 endDate,
                 tenor,
                 calendar='NullCalendar',
                 dateRule=BizDayConventions.Following,
                 dateGenerationRule=DateGeneration.Forward):
    parts = re.findall(r'(-?\d+|[a-zA-Z])', tenor)
    if len(parts)==3:
        return silceSchedule(firstDate=firstDate,
                 endDate=endDate,
                 tenor=tenor,
                 calendar=calendar,
                 dateRule=dateRule,
                 dateGenerationRule=dateGenerationRule)
    elif len(parts) ==2:
        return makeSchedule(firstDate=firstDate,
                 endDate=endDate,
                 tenor=tenor,
                 calendar=calendar,
                 dateRule=dateRule,
                 dateGenerationRule=dateGenerationRule)


__all__ = [
    "datesList", "bizDatesList", "holDatesList", "isBizDay", "advanceDate",
    "BizDayConventions", "DateGeneration", "adjustDateByCalendar",
    "advanceDateByCalendar", "nthWeekDay", "makeSchedule","silceSchedule","carveSchedule"
]
