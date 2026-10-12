"""Export compiled calendar rules for a reproducible file-based backtest."""
import argparse
from datetime import date
from pathlib import Path
from bomber.framework.calendars import NativeCalendarProvider, export_calendar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar-id", default="china.cffex")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_calendar(NativeCalendarProvider(args.calendar_id).snapshot(args.start, args.end), args.output)
    print(f"日历导出: {args.output}\n{result}")


if __name__ == "__main__":
    main()
