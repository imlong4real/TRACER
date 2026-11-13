#!/usr/bin/env python
"""Per-stage runtime and peak-memory benchmark for the TRACER pipeline.

Built for the unified-prune migration (``docs/design/unified_prune.md``):
record a baseline before a change, rerun after it, and compare stage by
stage. The regression tests pin *what* each stage outputs; this script
measures *how long and how much memory* each stage takes on real data.

Each repeat runs in a fresh subprocess, so every repeat pays the same
import and numba-JIT cost and the peak-RSS reading is not inherited from a
previous run. Memory is sampled by a background thread every
``--sample-ms`` over the whole process tree (parent plus any workers), and
the peak is attributed to the stage that was running when it occurred.
It is a true peak, not an end-of-stage snapshot.

Usage
-----
Record::

    python scripts/bench_stages.py record \
        --input transcripts.parquet --panel panel.csv.gz \
        --roi 548 1048 1144 1644 --repeats 3 --out baseline.csv

Compare a new run with a baseline (exit code 1 if anything is flagged)::

    python scripts/bench_stages.py record ... --out after.csv
    python scripts/bench_stages.py compare baseline.csv after.csv

Input columns: ``x/y/z`` (or Xenium ``x_location``/...), ``feature_name``,
``transcript_id``, ``cell_id`` (or ``input_cell_id``), optional
``overlaps_nucleus``. The panel is deduplicated to one row per unordered
gene pair before use, as production runners do.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STAGE_COLS = ["stage", "seconds", "peak_rss_gb"]


# --------------------------------------------------------------------------
# Environment checks
# --------------------------------------------------------------------------
def _git(*args) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), *args],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _check_kernels() -> list[str]:
    """Every compiled Cython kernel must be newer than its .pyx.

    A stale .so silently changes results (and timings) without any error.
    """
    stale = []
    for pyx in (REPO / "src" / "tracer").glob("_cy_*.pyx"):
        sos = list(pyx.parent.glob(pyx.stem + ".*.so"))
        if not sos:
            stale.append(f"{pyx.name}: not built")
        elif max(s.stat().st_mtime for s in sos) < pyx.stat().st_mtime:
            stale.append(f"{pyx.name}: .so older than .pyx")
    return stale


# --------------------------------------------------------------------------
# Worker: one pipeline run in this process
# --------------------------------------------------------------------------
class _RssSampler(threading.Thread):
    def __init__(self, interval_s: float):
        super().__init__(daemon=True)
        import psutil
        self._proc = psutil.Process()
        self._interval = interval_s
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.current = 0
        self.peak = 0

    def _tree_rss(self) -> int:
        total = 0
        try:
            total = self._proc.memory_info().rss
            for ch in self._proc.children(recursive=True):
                try:
                    total += ch.memory_info().rss
                except Exception:
                    pass
        except Exception:
            pass
        return total

    def run(self):
        while not self._stop.is_set():
            r = self._tree_rss()
            with self._lock:
                self.current = r
                self.peak = max(self.peak, r)
            self._stop.wait(self._interval)

    def take_peak(self) -> int:
        """Return the peak since the last call and reset it."""
        r = self._tree_rss()
        with self._lock:
            p = max(self.peak, r)
            self.peak = r
        return p

    def stop(self):
        self._stop.set()


def _load(args):
    import numpy as np
    import pandas as pd

    t = pd.read_parquet(args.input)
    t = t.rename(columns={"x_location": "x", "y_location": "y", "z_location": "z"})
    if "cell_id" not in t.columns and "input_cell_id" in t.columns:
        t = t.rename(columns={"input_cell_id": "cell_id"})
    keep = ["x", "y", "z", "feature_name", "cell_id", "transcript_id"]
    if "overlaps_nucleus" in t.columns and not args.no_nucleus:
        keep.append("overlaps_nucleus")
    t = t[keep]
    if args.roi:
        x0, x1, y0, y1 = args.roi
        t = t[(t.x >= x0) & (t.x < x1) & (t.y >= y0) & (t.y < y1)]
    t = t.reset_index(drop=True)
    for c in ("feature_name", "cell_id", "transcript_id"):
        t[c] = t[c].astype(str)

    p = pd.read_csv(args.panel)
    key = np.where(p.gene_i < p.gene_j, p.gene_i + "|" + p.gene_j, p.gene_j + "|" + p.gene_i)
    p = p.loc[~pd.Series(key).duplicated().values].reset_index(drop=True)
    return t, p


def _worker(args) -> None:
    sys.path.insert(0, str(REPO / "src"))
    import tracer.pipeline as P

    sampler = _RssSampler(args.sample_ms / 1000.0)
    sampler.start()

    t0 = time.perf_counter()
    df, panel = _load(args)
    load_s = time.perf_counter() - t0
    rows = [{"stage": "_load", "seconds": round(load_s, 3),
             "peak_rss_gb": round(sampler.take_peak() / 1e9, 3)}]

    last = [time.perf_counter()]
    orig = P._record_stage

    def timed(progression, stage_name, frame, col):
        now = time.perf_counter()
        rows.append({"stage": stage_name, "seconds": round(now - last[0], 3),
                     "peak_rss_gb": round(sampler.take_peak() / 1e9, 3)})
        out = orig(progression, stage_name, frame, col)
        last[0] = time.perf_counter()  # exclude the snapshot's own cost
        return out

    P._record_stage = timed
    run = P.run_noseg_pipeline if args.mode == "noseg" else P.run_segmented_pipeline
    t1 = time.perf_counter()
    out, _ = run(df, panel)
    total = time.perf_counter() - t1
    sampler.stop()

    # Output fingerprint: a timing comparison is only meaningful between
    # runs that produce the same labels (or between runs where you expect
    # them to differ, and know it).
    lab = out[["transcript_id", "tracer_id"]].astype(str).sort_values("transcript_id")
    sha = hashlib.sha256("\n".join(lab.transcript_id + "\t" + lab.tracer_id).encode()).hexdigest()

    rows.append({"stage": "_total_pipeline", "seconds": round(total, 3),
                 "peak_rss_gb": round(max(r["peak_rss_gb"] for r in rows), 3)})
    json.dump({"rows": rows, "n_tx": len(df), "output_sha": sha}, sys.stdout)


# --------------------------------------------------------------------------
# record / compare
# --------------------------------------------------------------------------
def _record(args) -> None:
    import pandas as pd

    stale = _check_kernels()
    if stale and not args.allow_stale:
        sys.exit("Stale or missing Cython kernels (rebuild with "
                 "`python setup.py build_ext --inplace`):\n  " + "\n  ".join(stale))

    worker_argv = [sys.executable, __file__, "_worker",
                   "--input", args.input, "--panel", args.panel,
                   "--mode", args.mode, "--sample-ms", str(args.sample_ms)]
    if args.roi:
        worker_argv += ["--roi", *map(str, args.roi)]
    if args.no_nucleus:
        worker_argv.append("--no-nucleus")

    runs, shas, n_tx = [], set(), None
    for i in range(args.repeats):
        res = subprocess.run(worker_argv, capture_output=True, text=True)
        if res.returncode != 0:
            sys.exit(f"repeat {i + 1} failed:\n{res.stderr[-3000:]}")
        payload = json.loads(res.stdout.strip().splitlines()[-1])
        n_tx, sha = payload["n_tx"], payload["output_sha"]
        shas.add(sha)
        for r in payload["rows"]:
            runs.append({**r, "repeat": i + 1})
        tot = [r for r in payload["rows"] if r["stage"] == "_total_pipeline"][0]
        print(f"repeat {i + 1}/{args.repeats}: {tot['seconds']:.1f} s, "
              f"peak {tot['peak_rss_gb']:.2f} GB", file=sys.stderr)
    if len(shas) != 1:
        print("WARNING: output differed between repeats", file=sys.stderr)

    raw = pd.DataFrame(runs)
    order = list(dict.fromkeys(raw.stage))
    g = raw.groupby("stage", sort=False)
    summ = pd.DataFrame({
        "stage": order,
        "seconds": [g.get_group(s).seconds.median() for s in order],
        "seconds_min": [g.get_group(s).seconds.min() for s in order],
        "seconds_max": [g.get_group(s).seconds.max() for s in order],
        "peak_rss_gb": [g.get_group(s).peak_rss_gb.max() for s in order],
    })
    meta = {
        "commit": _git("rev-parse", "--short", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "input": Path(args.input).name, "roi": args.roi, "mode": args.mode,
        "nucleus": not args.no_nucleus, "n_tx": n_tx, "repeats": args.repeats,
        "output_sha": sorted(shas)[0][:16] if len(shas) == 1 else "VARIES",
        "python": platform.python_version(), "machine": platform.machine(),
        "cpus": os.cpu_count(), "date": time.strftime("%Y-%m-%d"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        fh.write("# " + json.dumps(meta) + "\n")
        summ.to_csv(fh, index=False)
    raw.to_csv(out.with_suffix(".raw.csv"), index=False)
    print(json.dumps(meta), file=sys.stderr)
    print(summ.to_string(index=False))


def _read(path):
    import pandas as pd
    with open(path) as fh:
        meta = json.loads(fh.readline().lstrip("# "))
        return meta, pd.read_csv(fh)


def _compare(args) -> int:
    mb, b = _read(args.baseline)
    mc, c = _read(args.current)
    for k in ("input", "roi", "mode", "nucleus", "n_tx"):
        if mb.get(k) != mc.get(k):
            sys.exit(f"Not comparable: {k} differs ({mb.get(k)} vs {mc.get(k)})")
    m = b.merge(c, on="stage", how="outer", suffixes=("_base", "_cur"), sort=False)
    order = list(dict.fromkeys(list(b.stage) + list(c.stage)))
    m = m.set_index("stage").loc[order].reset_index()
    m["dt_s"] = m.seconds_cur - m.seconds_base
    m["ratio"] = m.seconds_cur / m.seconds_base
    m["drss_gb"] = m.peak_rss_gb_cur - m.peak_rss_gb_base
    # A stage is flagged only if it is slower by BOTH the ratio and the
    # absolute floor, so sub-second stages cannot trip on jitter.
    slow = (m.ratio > 1 + args.time_tol) & (m.dt_s > args.min_abs_s)
    fat = m.drss_gb > args.rss_tol * m.peak_rss_gb_base
    m["flag"] = ""
    m.loc[slow, "flag"] += "SLOWER "
    m.loc[fat, "flag"] += "MORE-RAM"
    cols = ["stage", "seconds_base", "seconds_cur", "dt_s", "ratio",
            "peak_rss_gb_base", "peak_rss_gb_cur", "flag"]
    print(f"baseline {mb['commit']}  vs  current {mc['commit']}"
          f"{' (dirty)' if mc.get('dirty') else ''}")
    same = mb["output_sha"] == mc["output_sha"]
    print(f"output: {'identical' if same else 'DIFFERENT'} "
          f"({mb['output_sha']} vs {mc['output_sha']})")
    print(m[cols].round(3).to_string(index=False))
    flagged = m[m.flag != ""]
    if len(flagged):
        print(f"\n{len(flagged)} stage(s) flagged "
              f"(time > +{args.time_tol:.0%} and > {args.min_abs_s}s, "
              f"or peak RSS > +{args.rss_tol:.0%})")
        return 1
    print("\nNo stage flagged.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def run_args(p):
        p.add_argument("--input", required=True)
        p.add_argument("--panel", required=True)
        p.add_argument("--roi", type=float, nargs=4, metavar=("X0", "X1", "Y0", "Y1"))
        p.add_argument("--mode", choices=["seg", "noseg"], default="seg")
        p.add_argument("--no-nucleus", action="store_true",
                       help="drop overlaps_nucleus (whole-cell Prune path)")
        p.add_argument("--sample-ms", type=float, default=50.0)

    rec = sub.add_parser("record")
    run_args(rec)
    rec.add_argument("--repeats", type=int, default=3)
    rec.add_argument("--out", required=True)
    rec.add_argument("--allow-stale", action="store_true")

    w = sub.add_parser("_worker")
    run_args(w)

    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("baseline")
    cmp_.add_argument("current")
    cmp_.add_argument("--time-tol", type=float, default=0.10)
    cmp_.add_argument("--min-abs-s", type=float, default=1.0)
    cmp_.add_argument("--rss-tol", type=float, default=0.05)

    args = ap.parse_args(argv)
    if args.cmd == "_worker":
        _worker(args)
        return 0
    if args.cmd == "record":
        _record(args)
        return 0
    return _compare(args)


if __name__ == "__main__":
    sys.exit(main())
