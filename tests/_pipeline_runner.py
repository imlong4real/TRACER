"""Backwards-compat shim. The canonical pipeline now lives in
``tracer.pipeline``; see ``plan/refactor_benchmark_workflow.md``.

This shim is a *forwarding module*: every attribute read and every
attribute write is routed to ``tracer.pipeline``. That means existing
callers that do::

    import tests._pipeline_runner as runner
    runner.PHASE1_RERANK_ENABLED = True

correctly mutate ``tracer.pipeline.PHASE1_RERANK_ENABLED``, which is the
global the pipeline reads from. New code should import from
``tracer.pipeline`` directly.

Why the forwarding (not a plain ``from tracer.pipeline import *``):
plain re-export would let test/benchmark code mutate *this module's*
attributes only; ``run_segmented_pipeline`` resolves its globals from
``tracer.pipeline``, so the monkeypatch would be silently dropped.
"""
from __future__ import annotations

import sys as _sys
import types as _types

from tracer import pipeline as _canonical


class _ForwardingShim(_types.ModuleType):
    """Module subclass that forwards getattr/setattr to ``tracer.pipeline``.

    Reads not satisfied from this shim's own ``__dict__`` (which only
    holds shim bookkeeping like the canonical ref and this class) fall
    through to the canonical module via ``__getattr__``. Writes always
    go to the canonical module, so ``runner.X = Y`` semantics are
    identical to ``tracer.pipeline.X = Y``.
    """

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
