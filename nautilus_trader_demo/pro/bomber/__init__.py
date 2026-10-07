"""Development overlay: local framework plus the installed Bomber engine.

Only ``framework/`` is published. Keep pro first on PYTHONPATH during development
so framework imports use this checkout while engine modules use the installed
Bomber package. Without an engine, the independent data utilities remain usable.
"""

from pathlib import Path as _FrameworkPath
from pkgutil import extend_path as _framework_extend_path
from importlib.util import spec_from_file_location as _framework_engine_spec

__path__ = _framework_extend_path(__path__, __name__)
_framework_local = _FrameworkPath(__file__).resolve().parent

for _framework_location in __path__:
    _framework_engine_init = _FrameworkPath(_framework_location) / "__init__.py"
    if (_framework_engine_init.parent.resolve() != _framework_local
            and _framework_engine_init.is_file()):
        # Preserve engine metadata (e.g. PACKAGE_ROOT) relative to its own file,
        # while retaining the local-first package search path for framework.
        __file__ = str(_framework_engine_init)
        __spec__ = _framework_engine_spec(__name__, __file__,
                                         submodule_search_locations=list(__path__))
        __loader__ = __spec__.loader
        __cached__ = __spec__.cached
        exec(compile(_framework_engine_init.read_bytes(), __file__, "exec"), globals())
        break

del _FrameworkPath, _framework_extend_path, _framework_engine_spec
