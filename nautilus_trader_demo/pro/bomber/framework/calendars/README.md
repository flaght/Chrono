# Calendar Cython build

This library is independent of the Bomber trading engine and NumPy. Its eleven
Cython extensions must be built with the Python interpreter, operating system
and CPU architecture used by the application. The existing Python 3.8 Linux
extension files cannot serve a Python 3.12 application.

## Build in the running environment

From the `pro` project root, with the target virtual environment activated:

```bash
uv pip install -r bomber/framework/calendars/requirements-build.txt
python bomber/framework/calendars/setup.py build_ext --inplace
python -u -m unittest tests.test_calendars_compiled -v
```

Without uv, use `python -m pip install -r
bomber/framework/calendars/requirements-build.txt` for the first step. A C compiler
and Python headers are required. Linux uses a working GCC/Clang toolchain; macOS
uses the Command Line Tools; Windows uses a compatible MSVC toolchain.

`setup.py` normalizes the working directory to the project root, discovers the
`.pyx` modules, and uses their full `bomber.framework.calendars.*` module names.
It generates new C sources and build intermediates under `.build/calendars/`,
leaving the historical `.c` files unchanged. `--inplace` places the extensions
beside their sources. Use `build_ext --inplace --force` to force recompilation
when compiler options change.

For manual transfer to another machine, copy the `.py`, `.pyx`, `.pxd` files and
`requirements-build.txt` in this directory and its subdirectories, plus
`tests/test_calendars_compiled.py`. Rebuild on that machine. Do not transfer the
local machine's `.so`/`.pyd`, `__pycache__` or `.build` artifacts as runtime inputs.
The source imports and Cython cimports now use the Bomber namespace; an external
`alphacopilot` installation is not needed.

## Interface

```python
from datetime import date
from bomber.framework.calendars.api import isBizDay, bizDatesList
from bomber.framework.calendars.DateUtilities import Calendar, Date

isBizDay("china.cffex", date(2026, 1, 12))
days = bizDatesList("china.cffex", date(2026, 1, 12), date(2026, 1, 16))
remaining = Calendar("china.cffex").bizDaysBetween(
    Date(2026, 1, 12), Date(2026, 1, 16), False, True,
)
```

The public date-list API returns Python `datetime` objects. `bizDaysBetween`
accepts explicit inclusion flags; the example excludes today and includes the
last trading day, matching the option-vega strategy's counting convention.

This change restores compilation and imports; it does not change holiday rules,
certify their completeness, or add an out-of-coverage guard. The existing CFFEX
implementation uses the SSE holiday rules and includes entries for 2026. A
public calendar provider still needs explicit coverage and version metadata
before replacing 08's required Live calendar file. Live can consume a snapshot
from that provider; backtests can consume an export of the same snapshot.
Intraday sessions and night-session trading-day attribution remain separate
from this date calendar.
