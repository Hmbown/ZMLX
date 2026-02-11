"""Plugin system for the ZMLX foundry.

Provides Protocol classes defining the plugin contracts, plus entry-point
and local-module loaders for extensibility.
"""
from __future__ import annotations

from .protocols import (
    DiscoveryContext,
    DiscoveryPlugin,
    ExportArtifacts,
    ExportPlugin,
    FoundryContext,
    FoundryPlugin,
    MoEPlugin,
)
from .registry import LoadedPlugin, load_entrypoint_plugins, load_local_plugin

__all__ = [
    "DiscoveryPlugin",
    "FoundryPlugin",
    "MoEPlugin",
    "ExportPlugin",
    "DiscoveryContext",
    "FoundryContext",
    "ExportArtifacts",
    "LoadedPlugin",
    "load_entrypoint_plugins",
    "load_local_plugin",
]
