#!/usr/bin/env python3
"""Compare local and Cirro TRACER outputs using canonical data fingerprints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_fingerprints(root: Path) -> dict:
    path = root / "provenance" / "output_fingerprints.json"
    if not path.exists() and (root / "tracer_results").is_dir():
        path = root / "tracer_results" / "provenance" / "output_fingerprints.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("local", type=Path)
    parser.add_argument("cirro", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    local = load_fingerprints(args.local)
    cirro = load_fingerprints(args.cirro)
    keys = ["refined_transcripts", "cell_scores", "cell_by_gene"]
    comparisons = {
        key: {
            "local": local[key],
            "cirro": cirro[key],
            "concordant": local[key] == cirro[key],
        }
        for key in keys
    }
    result = {
        "algorithm": local.get("algorithm"),
        "concordant": all(value["concordant"] for value in comparisons.values()),
        "comparisons": comparisons,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["concordant"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
