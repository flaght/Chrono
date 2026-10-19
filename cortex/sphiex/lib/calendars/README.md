# Calendar Cython build

This library is independent of the Bomber trading engine and NumPy. Its eleven
Cython extensions must be built with the Python interpreter, operating system
and CPU architecture used by the application. The existing Python 3.8 Linux
extension files cannot serve a Python 3.12 application.

## Build in the running environment

From the `pro` project root, with the target virtual environment activated:

```bash
uv pip install -r 3rd/calendars/requirements-build.txt
python 3rd/calendars/setup.py build_ext --inplace
PYTHONPATH=3rd python -u -m unittest discover -s 3rd/calendars/tests -v
```

Without uv, use `python -m pip install -r
3rd/calendars/requirements-build.txt` for the first step. A C compiler
and Python headers are required. Linux uses a working GCC/Clang toolchain; macOS
uses the Command Line Tools; Windows uses a compatible MSVC toolchain.

`setup.py` normalizes the working directory to the parent of this package,
discovers the `.pyx` modules, and derives extension names such as
`calendars.DateUtilities.Date` from the package directory. Python imports and
Cython cimports inside the package use relative references. It generates new C
sources and build intermediates under `3rd/calendars/.build/`,
leaving the historical `.c` files unchanged. `--inplace` places the extensions
beside their sources. Use `build_ext --inplace --force` to force recompilation
when compiler options change.

For manual transfer to another machine, copy the `.py`, `.pyx`, `.pxd` files,
`requirements-build.txt` and the `tests/` directory, including `setup.py`.
Rebuild on that machine. Do not transfer the
local machine's `.so`/`.pyd`, `__pycache__` or `.build` artifacts as runtime inputs.
The package has no dependency on the parent trading engine. After moving it,
rebuild the extensions and put its parent directory on `PYTHONPATH`. Historical
`.c` files are not used by this build; C sources are regenerated from `.pyx`.

## Interface

With `3rd` on `PYTHONPATH` (for example, `PYTHONPATH=3rd python`):

```python
from datetime import date
from calendars.api import isBizDay, bizDatesList
from calendars.DateUtilities import Calendar, Date

isBizDay("china.cffex", date(2026, 1, 12))
days = bizDatesList("china.cffex", date(2026, 1, 12), date(2026, 1, 16))
remaining = Calendar("china.cffex").bizDaysBetween(
    Date(2026, 1, 12), Date(2026, 1, 16), False, True,
)
```

The public date-list API returns Python `datetime` objects. `bizDaysBetween`
accepts explicit inclusion flags; the example excludes today and includes the
last trading day, matching the option-vega strategy's counting convention.

The public `NativeCalendarProvider` in `trading.py` now wraps the compiled rules
and returns an immutable `CalendarSnapshot` with explicit identity, version,
coverage and content fingerprint. It supports the bundled `china.cffex` and
`china.sse` rules for 1991–2026 and rejects requests outside that range. The
existing CFFEX implementation uses the SSE holiday rules. Holiday data is
unchanged; the boundary is declared for the bundled table, not inferred from
whether an unknown future weekday happens to be absent from its holiday set.

08 Live now consumes a native snapshot by default, without a calendar file.
`export_calendar()` writes a new CSV and completion manifest;
`ExportedCalendarProvider` verifies coverage, file SHA256 and snapshot identity.
The public dataprep calendar loader uses this provider for manifested exports,
so a backtest consumes the same rules version from a file.

```bash
PYTHONPATH=3rd python -u -m unittest discover -s 3rd/calendars/tests -v
python -m scripts.integration.reference_data.calendar_export \
  --calendar-id china.cffex --start 2026-01-01 --end 2026-12-31 \
  --output tests/results/cffex-2026.csv
```

Intraday sessions and night-session trading-day attribution remain separate
from this date calendar.
