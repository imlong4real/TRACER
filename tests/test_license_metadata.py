"""The project's license declarations must agree with each other.

Before the move to MIT they did not: LICENSE and the pyproject classifier said
Apache-2.0 while ``tracer.__license__`` already said MIT. This pins every
project-owned declaration to the same SPDX identifier.
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_license_file_is_mit_with_the_existing_holder():
    text = (ROOT / "LICENSE").read_text()
    assert text.startswith("MIT License")
    assert "Copyright (c) 2025 imlong4real" in text
    assert "Apache" not in text


def test_pyproject_declares_mit_as_an_spdx_expression():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]
    # PEP 639: a license expression replaces the license classifiers, and
    # setuptools refuses to build when both are present.
    assert not [c for c in project["classifiers"] if c.startswith("License ::")]


def test_package_dunder_matches():
    import tracer

    assert tracer.__license__ == "MIT"


def test_readme_badge_and_section_say_mit():
    readme = (ROOT / "README.md").read_text()
    assert "badge/license-MIT-" in readme
    assert "Apache" not in readme
