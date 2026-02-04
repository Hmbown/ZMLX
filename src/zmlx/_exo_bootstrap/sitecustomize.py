"""Auto-hook for exo subprocesses.

Python's ``site.py`` imports ``sitecustomize`` during interpreter startup.
The ZMLX exo launcher prepends this directory to ``PYTHONPATH``, so every
spawned subprocess (``multiprocessing.spawn``) picks up this module and
installs the load_mlx_items hook before exo code is imported.

The ``_ZMLX_EXO_HOOK`` env guard ensures zero impact on non-exo processes.
"""

import os

if os.environ.get("_ZMLX_EXO_HOOK") == "1":
    try:
        from zmlx.exo import install_hook

        install_hook()
    except Exception:
        pass
