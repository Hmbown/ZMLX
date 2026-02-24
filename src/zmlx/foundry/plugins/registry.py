"""Plugin discovery and loading for the ZMLX foundry.

Supports two loading mechanisms:

1. **Entry points** (``zmlx.foundry.plugins`` group): external packages
   declare plugins via ``[project.entry-points."zmlx.foundry.plugins"]``
   in their pyproject.toml.

2. **Local module paths**: for development or in-tree plugins, load by
   dotted module path plus optional attribute name.
"""
from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LoadedPlugin:
    """Container for a loaded plugin reference."""
    name: str
    obj: Any


# Entry-point group name.  External packages register under this group
# to be auto-discovered by the foundry.
_EP_GROUP = "zmlx.foundry.plugins"


def load_entrypoint_plugins(group: str = _EP_GROUP) -> list[LoadedPlugin]:
    """Load all plugins registered under the given entry-point group.

    Silently skips plugins that fail to load (broken install, missing
    dependency, etc.) to avoid crashing the entire foundry when one
    plugin is broken.
    """
    plugins: list[LoadedPlugin] = []
    try:
        eps = importlib.metadata.entry_points()
        # Python 3.12+ has eps.select(); 3.9/3.10 compat via dict get
        if hasattr(eps, "select"):
            group_eps = eps.select(group=group)
        else:
            group_eps = eps.get(group, [])  # type: ignore[union-attr]
        for ep in group_eps:
            try:
                obj = ep.load()
                plugins.append(LoadedPlugin(name=ep.name, obj=obj))
            except Exception:
                # Log at debug level in production; here we silently skip.
                continue
    except Exception:
        pass
    return plugins


def load_local_plugin(
    module_path: str,
    attr: str | None = None,
) -> LoadedPlugin:
    """Load a plugin from a local dotted module path.

    Parameters
    ----------
    module_path : str
        Fully qualified module path, e.g. ``"zmlx.foundry.myplugin"``.
    attr : str, optional
        Attribute name within the module.  If omitted, the module itself
        is returned as the plugin object.

    Returns
    -------
    LoadedPlugin
    """
    mod = importlib.import_module(module_path)
    obj = getattr(mod, attr) if attr else mod
    name = f"{module_path}:{attr}" if attr else module_path
    return LoadedPlugin(name=name, obj=obj)
