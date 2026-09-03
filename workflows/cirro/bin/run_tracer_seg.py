#!/usr/bin/env python3
"""Run the pinned TRACER Seg CLI and add Cirro-oriented provenance."""

from __future__ import annotations

import argparse
import base64
import dataclasses
import gzip
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TRACER_ROOT = Path("/app")
TRACER_RUNNER = TRACER_ROOT / "scripts" / "run_tracer.py"
TRACER_PREPROCESSOR = TRACER_ROOT / "scripts" / "preprocess_xenium.py"
REQUIRED_OUTPUTS = (
    "outputs/transcripts_tracer_refined.parquet",
    "outputs/cell_by_gene_tracer.h5ad",
    "outputs/cell_scores.tsv.gz",
    "config_receipt.json",
    "runtime_memory.json",
    "run_summary.md",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def decode_source(value: str) -> str | None:
    if not value:
        return None
    return base64.b64decode(value.encode("ascii")).decode("utf-8")


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def run_logged(
    command: list[str], log_path: Path, *, env: dict[str, str] | None = None
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("argv: " + json.dumps(command) + "\n")
        log_handle.flush()
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_handle.write(line)
        return_code = proc.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _first_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    return next((name for name in candidates if name in available), None)


def _geojson_records(path: Path) -> list[dict[str, Any]]:
    from shapely.geometry import shape

    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = []
    for index, feature in enumerate(payload.get("features", [])):
        props = dict(feature.get("properties") or {})
        props["geometry"] = shape(feature["geometry"])
        props.setdefault("_feature_index", index)
        records.append(props)
    return records


def load_vector_boundaries(path: Path, kind: str):
    """Load Xenium vertex tables, GeoParquet, or GeoJSON boundaries."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Polygon

    lower = path.name.lower()
    frame = None
    if lower.endswith((".geojson", ".json", ".geojson.gz", ".json.gz")):
        frame = pd.DataFrame(_geojson_records(path))
    elif lower.endswith(".parquet"):
        try:
            geo = gpd.read_parquet(path)
            if "geometry" in geo.columns:
                frame = geo
        except Exception:
            frame = None
        if frame is None:
            frame = pd.read_parquet(path)
    elif lower.endswith((".csv", ".csv.gz")):
        frame = pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported {kind} boundary format for {path}; use Parquet, CSV(.gz), or GeoJSON(.gz)"
        )

    id_candidates = (
        ["cell_id", "cell", "label", "object_id", "EntityID", "id"]
        if kind == "cell"
        else ["nucleus_id", "cell_id", "nucleus", "label", "object_id", "EntityID", "id"]
    )
    id_column = _first_column(frame.columns, id_candidates)

    if "geometry" not in frame.columns:
        x_column = _first_column(frame.columns, ["vertex_x", "x", "x_location"])
        y_column = _first_column(frame.columns, ["vertex_y", "y", "y_location"])
        if not id_column or not x_column or not y_column:
            raise ValueError(
                f"{path} must be a geometry table or contain an ID plus vertex_x/vertex_y; "
                f"found {list(frame.columns)}"
            )
        order_column = _first_column(
            frame.columns, ["vertex_index", "vertex_id", "point_index", "index"]
        )
        if order_column:
            frame = frame.sort_values([id_column, order_column], kind="mergesort")
        records = []
        for mask_id, group in frame.groupby(id_column, sort=True, observed=True):
            polygon = Polygon(list(zip(group[x_column], group[y_column])))
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if not polygon.is_empty:
                records.append({"_mask_id": str(mask_id), "geometry": polygon})
        geo = gpd.GeoDataFrame(records, geometry="geometry")
    else:
        geo = gpd.GeoDataFrame(frame, geometry="geometry")
        if id_column:
            geo["_mask_id"] = geo[id_column].astype(str)
        else:
            geo["_mask_id"] = [str(index) for index in range(len(geo))]
        geo = geo.loc[geo.geometry.notna() & ~geo.geometry.is_empty, ["_mask_id", "geometry"]]

    if geo.empty:
        raise ValueError(f"No usable polygons found in {path}")
    geo = geo.reset_index(drop=True)
    geo["_mask_id"] = geo["_mask_id"].astype(str)
    return geo


def _polygon_matches(x_values, y_values, polygons):
    import geopandas as gpd

    points = gpd.GeoDataFrame(
        {"_row": range(len(x_values))},
        geometry=gpd.points_from_xy(x_values, y_values),
    )
    joined = gpd.sjoin(
        points,
        polygons[["_mask_id", "geometry"]],
        how="left",
        predicate="intersects",
    )
    return joined.loc[joined["index_right"].notna(), ["_row", "_mask_id"]]


def apply_vector_masks(
    transcripts: Path,
    output: Path,
    cell_boundaries: Path | None,
    nucleus_boundaries: Path | None,
    summary_path: Path,
) -> Path:
    """Overlay transcript coordinates on optional vector masks in batches."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    cell_polygons = load_vector_boundaries(cell_boundaries, "cell") if cell_boundaries else None
    nucleus_polygons = (
        load_vector_boundaries(nucleus_boundaries, "nucleus") if nucleus_boundaries else None
    )

    parquet = pq.ParquetFile(transcripts)
    writer = None
    rows_total = 0
    rows_cell_assigned = 0
    rows_nuclear = 0
    try:
        for batch in parquet.iter_batches(batch_size=250_000):
            frame = batch.to_pandas()
            x_column = _first_column(frame.columns, ["x", "x_location"])
            y_column = _first_column(frame.columns, ["y", "y_location"])
            if not x_column or not y_column:
                raise ValueError("Transcript table has no x/y or x_location/y_location columns")

            if cell_polygons is not None:
                matches = _polygon_matches(frame[x_column], frame[y_column], cell_polygons)
                labels = np.full(len(frame), "-1", dtype=object)
                if not matches.empty:
                    best = (
                        matches.assign(_mask_id=matches["_mask_id"].astype(str))
                        .sort_values(["_row", "_mask_id"], kind="mergesort")
                        .drop_duplicates("_row", keep="first")
                    )
                    labels[best["_row"].to_numpy(dtype=int)] = best["_mask_id"].to_numpy()
                frame["cell_id"] = labels

            if nucleus_polygons is not None:
                matches = _polygon_matches(frame[x_column], frame[y_column], nucleus_polygons)
                overlaps = np.zeros(len(frame), dtype=np.uint8)
                if not matches.empty:
                    overlaps[np.unique(matches["_row"].to_numpy(dtype=int))] = 1
                frame["overlaps_nucleus"] = overlaps

            rows_total += len(frame)
            if "cell_id" in frame.columns:
                rows_cell_assigned += int((frame["cell_id"].astype(str) != "-1").sum())
            if "overlaps_nucleus" in frame.columns:
                rows_nuclear += int(frame["overlaps_nucleus"].astype(bool).sum())

            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    summary = {
        "cell_boundaries": str(cell_boundaries) if cell_boundaries else None,
        "nucleus_boundaries": str(nucleus_boundaries) if nucleus_boundaries else None,
        "cell_polygon_count": len(cell_polygons) if cell_polygons is not None else 0,
        "nucleus_polygon_count": len(nucleus_polygons) if nucleus_polygons is not None else 0,
        "rows_total": rows_total,
        "rows_cell_assigned": rows_cell_assigned,
        "rows_overlapping_nucleus": rows_nuclear,
        "multiple_polygon_rule": "lexicographically-smallest-mask-id",
        "point_predicate": "intersects",
    }
    write_json(summary_path, summary)
    return output


def resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    from tracer.config import load_config, to_dict

    config_path = Path(args.user_config) if args.user_config else None
    config = load_config(path=config_path, platform=args.platform)
    if args.g_z_um is not None:
        value: str | float = args.g_z_um
        if value != "auto":
            value = float(value)
        config = dataclasses.replace(
            config,
            stitch=dataclasses.replace(config.stitch, g_z_um=value),
        )
    return {
        "platform": args.platform,
        "config": to_dict(config),
        "cli_overrides": {
            "pmi_threshold": args.pmi_threshold,
            "g_z_um": args.g_z_um,
            "tau": args.tau,
            "min_tx_per_cell_for_scores": args.min_tx_per_cell_for_scores,
            "score_mode": args.score_mode,
            "seed": args.seed,
        },
    }


def canonical_frame_hash(path: Path, sort_columns: list[str]) -> tuple[str, int]:
    import pandas as pd

    if path.name.endswith(".parquet"):
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, sep="\t")
    ordered_columns = sorted(frame.columns)
    sort_by = [column for column in sort_columns if column in frame.columns]
    if sort_by:
        frame = frame.sort_values(sort_by, kind="mergesort", na_position="last")
    payload = frame[ordered_columns].to_csv(
        index=False,
        lineterminator="\n",
        na_rep="<NA>",
        float_format="%.17g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(frame)


def canonical_h5ad_hash(path: Path) -> tuple[str, list[int]]:
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

    adata = ad.read_h5ad(path)
    obs_order = np.argsort(adata.obs_names.astype(str), kind="mergesort")
    var_order = np.argsort(adata.var_names.astype(str), kind="mergesort")
    matrix = adata.X[obs_order, :][:, var_order]
    matrix = sp.csr_matrix(matrix)

    digest = hashlib.sha256()
    digest.update(json.dumps(list(matrix.shape)).encode("ascii"))
    for values in (matrix.indptr, matrix.indices, matrix.data):
        contiguous = np.ascontiguousarray(values)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    digest.update("\n".join(adata.obs_names[obs_order].astype(str)).encode("utf-8"))
    digest.update("\n".join(adata.var_names[var_order].astype(str)).encode("utf-8"))

    obs = adata.obs.iloc[obs_order].copy()
    obs.index = adata.obs_names[obs_order].astype(str)
    digest.update(
        obs.sort_index(axis=1).to_csv(
            lineterminator="\n", na_rep="<NA>", float_format="%.17g"
        ).encode("utf-8")
    )
    return digest.hexdigest(), [int(matrix.shape[0]), int(matrix.shape[1])]


def output_fingerprints(outdir: Path) -> dict[str, Any]:
    transcripts_hash, transcript_rows = canonical_frame_hash(
        outdir / "outputs" / "transcripts_tracer_refined.parquet",
        ["transcript_id", "feature_name", "x", "y", "z"],
    )
    scores_hash, score_rows = canonical_frame_hash(
        outdir / "outputs" / "cell_scores.tsv.gz", ["cell_id"]
    )
    matrix_hash, matrix_shape = canonical_h5ad_hash(
        outdir / "outputs" / "cell_by_gene_tracer.h5ad"
    )
    return {
        "algorithm": "sha256-canonical-v1",
        "refined_transcripts": {"sha256": transcripts_hash, "rows": transcript_rows},
        "cell_scores": {"sha256": scores_hash, "rows": score_rows},
        "cell_by_gene": {"sha256": matrix_hash, "shape": matrix_shape},
    }


def write_checksums(outdir: Path) -> None:
    checksum_path = outdir / "provenance" / "checksums.sha256"
    rows = []
    for path in sorted(outdir.rglob("*")):
        if path.is_file() and path != checksum_path:
            rows.append(f"{sha256_file(path)}  {path.relative_to(outdir).as_posix()}")
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--transcripts", required=True, type=Path)
    result.add_argument("--pmi", required=True, type=Path)
    result.add_argument("--outdir", required=True, type=Path)
    result.add_argument("--sample-name", required=True)
    result.add_argument("--platform", choices=["xenium", "atera"], required=True)
    result.add_argument("--qv-min", type=float)
    result.add_argument("--remove-control-probes", action="store_true")
    result.add_argument("--drop-unassigned", action="store_true")
    result.add_argument("--pmi-threshold", type=float)
    result.add_argument("--g-z-um")
    result.add_argument("--tau", type=float)
    result.add_argument("--min-tx-per-cell-for-scores", type=int, default=5)
    result.add_argument("--score-mode", choices=["count", "magnitude"], default="count")
    result.add_argument("--seed", type=int, default=1)
    result.add_argument("--user-config", type=Path)
    result.add_argument("--cell-boundaries", type=Path)
    result.add_argument("--nucleus-boundaries", type=Path)
    result.add_argument("--fastparquet-shim", required=True, type=Path)
    for name in (
        "transcripts-source-b64",
        "pmi-source-b64",
        "user-config-source-b64",
        "cell-boundaries-source-b64",
        "nucleus-boundaries-source-b64",
    ):
        result.add_argument(f"--{name}", default="")
    result.add_argument("--container-image", required=True)
    result.add_argument("--execution-container", required=True)
    result.add_argument("--tracer-source-commit", required=True)
    result.add_argument("--tracer-version", required=True)
    result.add_argument("--workflow-commit", required=True)
    result.add_argument("--workflow-revision", required=True)
    result.add_argument("--task-attempt", type=int, required=True)
    result.add_argument("--task-cpus", type=int, required=True)
    result.add_argument("--task-memory-b64", required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    outdir = args.outdir.resolve()
    workdir = Path("adapter_work").resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    workdir.mkdir(parents=True, exist_ok=False)
    (outdir / "logs").mkdir()
    (outdir / "preprocessing" / "qc").mkdir(parents=True)
    (outdir / "provenance").mkdir()

    input_paths = {
        "transcripts": args.transcripts.resolve(),
        "pmi": args.pmi.resolve(),
        "user_config": args.user_config.resolve() if args.user_config else None,
        "cell_boundaries": args.cell_boundaries.resolve() if args.cell_boundaries else None,
        "nucleus_boundaries": args.nucleus_boundaries.resolve() if args.nucleus_boundaries else None,
    }
    input_sources = {
        "transcripts": decode_source(args.transcripts_source_b64),
        "pmi": decode_source(args.pmi_source_b64),
        "user_config": decode_source(args.user_config_source_b64),
        "cell_boundaries": decode_source(args.cell_boundaries_source_b64),
        "nucleus_boundaries": decode_source(args.nucleus_boundaries_source_b64),
    }
    manifest = {
        "status": "started",
        "started_at_utc": utc_now(),
        "sample_name": args.sample_name,
        "tracer": {
            "version": args.tracer_version,
            "source_commit": args.tracer_source_commit,
            "container": args.container_image,
        },
        "workflow": {
            "commit": args.workflow_commit,
            "revision": args.workflow_revision,
        },
        "execution": {
            "task_attempt": args.task_attempt,
            "cpus": args.task_cpus,
            "memory": decode_source(args.task_memory_b64),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "seed": args.seed,
            "container_runtime_source": args.execution_container,
        },
        "inputs": {
            name: (
                {
                    "source": input_sources[name],
                    "effective_staged_path": str(path),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                if path is not None
                else None
            )
            for name, path in input_paths.items()
        },
    }
    write_json(outdir / "provenance" / "run_manifest.json", manifest)

    resolved = resolved_config(args)
    write_json(outdir / "provenance" / "resolved_tracer_config.json", resolved)

    try:
        preprocessing_input = args.transcripts.resolve()
        if args.cell_boundaries or args.nucleus_boundaries:
            preprocessing_input = apply_vector_masks(
                transcripts=args.transcripts.resolve(),
                output=workdir / "mask_assigned_transcripts.parquet",
                cell_boundaries=input_paths["cell_boundaries"],
                nucleus_boundaries=input_paths["nucleus_boundaries"],
                summary_path=outdir / "preprocessing" / "mask_assignment_summary.json",
            )

        standardized = workdir / "transcripts_standardized.parquet"
        preprocess_command = [
            sys.executable,
            str(TRACER_PREPROCESSOR),
            "--input",
            str(preprocessing_input),
            "--out",
            str(standardized),
            "--summary-dir",
            str(outdir / "preprocessing" / "qc"),
            "--platform",
            args.platform,
        ]
        if args.qv_min is not None:
            preprocess_command += ["--qv-min", str(args.qv_min)]
        if args.remove_control_probes:
            preprocess_command.append("--remove-control-probes")
        if args.drop_unassigned:
            preprocess_command.append("--drop-unassigned")
        preprocess_env = os.environ.copy()
        adapter_bin = str(args.fastparquet_shim.resolve().parent)
        preprocess_env["PYTHONPATH"] = os.pathsep.join(
            value
            for value in (adapter_bin, preprocess_env.get("PYTHONPATH", ""))
            if value
        )
        run_logged(
            preprocess_command,
            outdir / "logs" / "preprocess.log",
            env=preprocess_env,
        )

        manifest["inputs"]["standardized_transcripts"] = {
            "effective_staged_path": str(standardized),
            "sha256": sha256_file(standardized),
            "size_bytes": standardized.stat().st_size,
        }
        write_json(outdir / "provenance" / "run_manifest.json", manifest)

        tracer_command = [
            sys.executable,
            str(TRACER_RUNNER),
            "--transcripts",
            str(standardized),
            "--npmi",
            str(args.pmi.resolve()),
            "--outdir",
            str(outdir),
            "--sample-name",
            args.sample_name,
            "--platform",
            args.platform,
            "--seed",
            str(args.seed),
            "--min-tx-per-cell-for-scores",
            str(args.min_tx_per_cell_for_scores),
            "--score-mode",
            args.score_mode,
            "--overwrite",
        ]
        if args.user_config:
            tracer_command += ["--user-config", str(args.user_config.resolve())]
        if args.pmi_threshold is not None:
            tracer_command += ["--pmi-threshold", str(args.pmi_threshold)]
        if args.g_z_um is not None:
            tracer_command += ["--g-z-um", args.g_z_um]
        if args.tau is not None:
            tracer_command += ["--tau", str(args.tau)]
        run_logged(tracer_command, outdir / "logs" / "tracer_stdout_stderr.log")

        for relative_path in REQUIRED_OUTPUTS:
            path = outdir / relative_path
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"TRACER did not create required output: {relative_path}")

        if (outdir / "run.log").exists():
            shutil.copy2(outdir / "run.log", outdir / "logs" / "tracer.log")

        receipt_path = outdir / "config_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["cirro_adapter"] = {
            "tracer_version": args.tracer_version,
            "tracer_source_commit": args.tracer_source_commit,
            "container": args.container_image,
            "workflow_commit": args.workflow_commit,
            "workflow_revision": args.workflow_revision,
            "resolved_config": "provenance/resolved_tracer_config.json",
        }
        receipt["inputs"]["effective_pmi"] = manifest["inputs"]["pmi"]
        receipt["inputs"]["effective_transcripts"] = manifest["inputs"]["transcripts"]
        receipt["inputs"]["cell_boundaries"] = manifest["inputs"]["cell_boundaries"]
        receipt["inputs"]["nucleus_boundaries"] = manifest["inputs"]["nucleus_boundaries"]
        write_json(receipt_path, receipt)

        versions = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import sys,tracer; print('python=' + sys.version.split()[0]); "
                "print('tracer=' + tracer.__version__)",
            ],
            text=True,
        )
        (outdir / "provenance" / "software_versions.txt").write_text(
            versions
            + f"tracer_source_commit={args.tracer_source_commit}\n"
            + f"container={args.container_image}\n",
            encoding="utf-8",
        )

        fingerprints = output_fingerprints(outdir)
        write_json(outdir / "provenance" / "output_fingerprints.json", fingerprints)
        manifest["status"] = "complete"
        manifest["completed_at_utc"] = utc_now()
        manifest["output_fingerprints"] = fingerprints
        write_json(outdir / "provenance" / "run_manifest.json", manifest)
        write_checksums(outdir)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["completed_at_utc"] = utc_now()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_json(outdir / "provenance" / "run_manifest.json", manifest)
        write_checksums(outdir)
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
