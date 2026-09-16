"""A stale config must say what to write INSTEAD, not just that it is wrong.

Removing `nuclear_only_admit` / `nuclear_seed_only` / `seed_coherence_floor`
is a deliberate hard break (see prune_scope). The loader already rejects
unknown keys, but a bare "unknown keys: [...] (valid: [40 names])" leaves the
reader to reverse-engineer the migration. These keys have a known replacement,
so the error should name it.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tracer.config import load_config  # noqa: E402


def _load(body: str):
    p = Path(tempfile.mkdtemp()) / "user.toml"
    p.write_text(body)
    return load_config(path=p)


@pytest.mark.parametrize("key,value,must_mention", [
    ("nuclear_only_admit", "false", 'prune_scope = "cell"'),
    ("nuclear_only_admit", "true", 'prune_scope = "nuclear"'),
    ("nuclear_seed_only", "false", "prune_scope"),
    ("seed_coherence_floor", "0.1", "Mid-QC"),
])
def test_retired_key_error_names_the_replacement(key, value, must_mention):
    with pytest.raises(ValueError) as ei:
        _load(f"[phase1]\n{key} = {value}\n")
    msg = str(ei.value)
    assert key in msg, f"error should name the offending key; got: {msg}"
    assert must_mention in msg, (
        f"error for retired key {key!r} must point at its replacement "
        f"({must_mention!r}); got: {msg}")


def test_unknown_key_without_a_known_replacement_still_errors_plainly():
    """A genuine typo has no migration hint — keep the valid-key listing."""
    with pytest.raises(ValueError) as ei:
        _load("[phase1]\nprune_scoop = 'cell'\n")
    msg = str(ei.value)
    assert "prune_scoop" in msg
    assert "valid:" in msg, "plain unknown keys should still list valid names"


def test_valid_config_is_unaffected():
    cfg = _load('[phase1]\nprune_scope = "nuclear"\n')
    assert cfg.phase1.resolve_scope() == (True, True)
