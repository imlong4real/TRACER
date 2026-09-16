#!/usr/bin/env python3
"""Run TRACER on one downsampled transcript parquet."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transcripts", required=True, type=Path)
    panel = p.add_mutually_exclusive_group(required=True)
    panel.add_argument(
        "--npmi",
        type=Path,
        help="Legacy name for the TRACER gene-pair panel.",
    )
    panel.add_argument(
        "--panel",
        type=Path,
        help="TRACER gene-pair panel. PMI is the operational edge-weight column.",
    )
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--sample-name", required=True)
    p.add_argument("--platform", default="xenium")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--runner", default="scripts/run_tracer.py")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    panel = args.panel if args.panel is not None else args.npmi
    cmd = [
        sys.executable,
        args.runner,
        "--transcripts",
        str(args.transcripts),
        "--npmi",
        str(panel),
        "--outdir",
        str(args.outdir),
        "--sample-name",
        args.sample_name,
        "--platform",
        args.platform,
        "--seed",
        str(args.seed),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
