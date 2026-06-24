"""Backwards-compat shim. The canonical tiled orchestrator now lives in
``tracer.pipeline_tiled``; see ``plan/refactor_benchmark_workflow.md``.

Forwarding-module pattern: reads and writes are routed to
``tracer.pipeline_tiled`` so any caller that does
``import tests._pipeline_runner_tiled as runner_tiled`` keeps working
without modification.
"""
from __future__ import annotations

import sys as _sys
import types as _types

from tracer import pipeline_tiled as _canonical


class _ForwardingShim(_types.ModuleType):
    def __getattr__(self, name: str):
        return getattr(_canonical, name)

    def __setattr__(self, name: str, value) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(_canonical, name, value)

    def __delattr__(self, name: str) -> None:
        if name.startswith("_") and name in self.__dict__:
            object.__delattr__(self, name)
        else:
            delattr(_canonical, name)

    def __dir__(self):
        return sorted(set(list(self.__dict__) + dir(_canonical)))


_sys.modules[__name__].__class__ = _ForwardingShim
