#!/usr/bin/env python3
"""
Prepare Slide 3 tissue-piece QC artifacts from Xenium cell centroids.

The default workflow is QC-first: detect major tissue components from
cells.csv.gz, write a summary CSV and review plot, then stop. Per-piece
transcript parquet generation is available only through --write-pieces and
requires an explicit approved manifest.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SLIDE_TISSUE_IDS = {
    1: "DFCI4.S1.C4.L1",
    2: "DFCI4.S2.C4.L1",
    3: "DFCI4.S3.C4.L1",
    4: "DFCI4.S3.C4.L2",
    5: "DFCI4.S4.C4.L1",
    6: "DFCI4.S4.C4.L2",
    7: "DFCI4.S5.C4.L1",
    8: "DFCI4.S5.C4.L2",
    9: "DFCI16.S1.C4.L1",
    10: "DFCI16.S2.C4.L1",
    11: "DFCI16.S3.C4.L1",
    12: "DFCI16.S4.C4.L1",
}
SAMPLE_LABELS = {
    "Patient4": list(range(1, 9)),
    "Patient6": list(range(9, 13)),
}
SAMPLE_LABEL_ASSIGNMENT_ORDER = {
    "Patient4": [1, 2, 4, 3, 6, 5, 7, 8],
    "Patient6": SAMPLE_LABELS["Patient6"],
}
DEFAULT_BIN_SIZES_UM = (50.0, 75.0)
COORD_ALIASES = {
    "x_centroid": "x_centroid",
    "y_centroid": "y_centroid",
    "cell_id": "cell_id",
    "x": "x_centroid",
    "y": "y_centroid",
    "centroid_x": "x_centroid",
    "centroid_y": "y_centroid",
}
TRANSCRIPT_COORD_ALIASES = {
    "x_location": "x",
    "y_location": "y",
    "z_location": "z",
}
TRUE_VALUES = {"1", "true", "t", "yes", "y", "approved"}
APPROVAL_TEMPLATE_COLUMNS = [
    "sample_name",
    "component_id",
    "assigned_label",
    "slide_tissue_id",
    "approved",
]
PIECE_RUN_MANIFEST_COLUMNS = [
    "task_id",
    "sample_name",
    "component_id",
    "assigned_label",
    "slide_tissue_id",
    "input_piece_parquet",
    "output_tracer_parquet",
    "rows_written",
]


@dataclass(frozen=True)
class SampleInput:
    name: str
    xenium_dir: Path

    @property
    def cells_path(self) -> Path:
        return self.xenium_dir / "cells.csv.gz"

    @property
    def transcripts_path(self) -> Path:
        return self.xenium_dir / "transcripts.parquet"


@dataclass(frozen=True)
class Component:
    sample_name: str
    raw_component_id: int
    component_id: str
    bin_size_um: float
    cell_count: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    x_centroid: float
    y_centroid: float
    assigned_label: int | None
    slide_tissue_id: str
    qc_status: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect Slide 3 tissue components and prepare QC artifacts."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--qc-only",
        action="store_true",
        help="Write component_summary.csv and component plot, then exit.",
    )
    mode.add_argument(
        "--write-pieces",
        action="store_true",
        help="Write per-piece transcript parquets using an approved manifest.",
    )
    parser.add_argument(
        "--xenium-output",
        action="append",
        default=[],
        metavar="SAMPLE=PATH",
        help=(
            "Xenium output directory containing cells.csv.gz and transcripts.parquet. "
            "May be repeated for Patient4 and Patient6."
        ),
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Defaults to tutorials/gbm/output/slide3_qc.",
    )
    parser.add_argument(
        "--bin-sizes-um",
        default="50,75",
        help="Comma-separated occupancy bin sizes to try, in microns.",
    )
    parser.add_argument(
        "--min-component-cells",
        type=int,
        default=500,
        help="Minimum cells required for a component to be considered a tissue piece.",
    )
    parser.add_argument(
        "--component-sort",
        choices=("yx", "xy"),
        default="yx",
        help="Spatial ordering used to assign expected labels to detected components.",
    )
    parser.add_argument(
        "--max-plot-cells-per-component",
        type=int,
        default=25000,
        help="Maximum centroid points plotted per component.",
    )
    parser.add_argument(
        "--no-invert-y",
        action="store_true",
        help="Do not invert the y axis in the review plot.",
    )
    parser.add_argument(
        "--approved-manifest",
        help=(
            "CSV manifest required by --write-pieces. Must include sample_name, "
            "component_id, assigned_label, and approved columns."
        ),
    )
    parser.add_argument(
        "--pieces-outdir",
        default=None,
        help=(
            "Output directory for --write-pieces. "
            "Defaults to tutorials/gbm/output/slide3_pieces."
        ),
    )
    parser.add_argument(
        "--hull-buffer-um",
        type=float,
        default=100.0,
        help="Buffer applied to component hulls when assigning transcripts to pieces.",
    )
    parser.add_argument(
        "--transcript-batch-size",
        type=int,
        default=250000,
        help="Parquet batch size for streaming transcripts in --write-pieces mode.",
    )
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_qc_outdir() -> Path:
    return _repo_root() / "tutorials" / "gbm" / "output" / "slide3_qc"


def _default_pieces_outdir() -> Path:
    return _repo_root() / "tutorials" / "gbm" / "output" / "slide3_pieces"


def _parse_bin_sizes(value: str) -> tuple[float, ...]:
    sizes = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not sizes:
        raise ValueError("At least one occupancy bin size is required.")
    if any(size <= 0 for size in sizes):
        raise ValueError("Occupancy bin sizes must be positive.")
    return sizes


def _parse_sample_spec(value: str) -> SampleInput:
    if "=" not in value:
        raise ValueError(f"Expected SAMPLE=PATH for --xenium-output, got: {value}")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Missing sample name in --xenium-output: {value}")
    if name not in SAMPLE_LABELS:
        expected = ", ".join(SAMPLE_LABELS)
        raise ValueError(f"Unknown sample {name!r}. Expected one of: {expected}")
    return SampleInput(name=name, xenium_dir=Path(raw_path).expanduser().resolve())


def _discover_default_samples() -> list[SampleInput]:
    gbm_dir = _repo_root() / "tutorials" / "gbm"
    roots = [
        gbm_dir / "data",
        gbm_dir / "data" / "xenium",
        gbm_dir / "data" / "slide3",
        _repo_root().parents[1] / "data" / "xenium",
    ]
    suffixes = ("", "_xenium", "_xenium_output", "_outs")
    samples: list[SampleInput] = []
    seen: set[str] = set()

    for sample_name in SAMPLE_LABELS:
        found = False
        for root in roots:
            for suffix in suffixes:
                candidate = root / f"{sample_name}{suffix}"
                if (candidate / "cells.csv.gz").exists():
                    key = str(candidate.resolve())
                    if key not in seen:
                        samples.append(SampleInput(sample_name, candidate.resolve()))
                        seen.add(key)
                    found = True
                    break
            if found:
                break
            if root.exists():
                for candidate in sorted(root.glob(f"*{sample_name}*")):
                    if (candidate / "cells.csv.gz").exists():
                        key = str(candidate.resolve())
                        if key not in seen:
                            samples.append(
                                SampleInput(sample_name, candidate.resolve())
                            )
                            seen.add(key)
                        found = True
                        break
            if found:
                break

    return samples


def resolve_samples(sample_specs: Iterable[str]) -> list[SampleInput]:
    samples = [_parse_sample_spec(spec) for spec in sample_specs]
    if not samples:
        samples = _discover_default_samples()
    if not samples:
        raise SystemExit(
            "No Xenium outputs found. Pass "
            "--xenium-output SAMPLE=/path/to/xenium_output."
        )

    seen_names: set[str] = set()
    for sample in samples:
        if sample.name in seen_names:
            raise ValueError(f"Duplicate Xenium output for sample {sample.name!r}.")
        seen_names.add(sample.name)
        if not sample.cells_path.exists():
            raise FileNotFoundError(
                f"Missing cells.csv.gz for {sample.name}: {sample.cells_path}"
            )
    return samples


def _normalize_cell_columns(df):
    import pandas as pd

    rename_map = {}
    for col in df.columns:
        canonical = COORD_ALIASES.get(col)
        if canonical and canonical not in df.columns:
            rename_map[col] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)
    missing = {"x_centroid", "y_centroid"} - set(df.columns)
    if missing:
        raise ValueError(
            "cells.csv.gz is missing required centroid columns: "
            f"{', '.join(sorted(missing))}"
        )

    columns = [
        col for col in ("cell_id", "x_centroid", "y_centroid") if col in df.columns
    ]
    df = df[columns].copy()
    df["x_centroid"] = pd.to_numeric(df["x_centroid"], errors="coerce")
    df["y_centroid"] = pd.to_numeric(df["y_centroid"], errors="coerce")
    df = df.dropna(subset=["x_centroid", "y_centroid"]).reset_index(drop=True)
    if df.empty:
        raise ValueError(
            "No valid cell centroids remain after dropping missing coordinates."
        )
    return df


def load_cells(cells_path: Path):
    import pandas as pd

    df = pd.read_csv(cells_path)
    return _normalize_cell_columns(df)


def _occupied_bin_components(bin_xy) -> dict[tuple[int, int], int]:
    occupied = {tuple(item) for item in bin_xy.tolist()}
    labels: dict[tuple[int, int], int] = {}
    component_id = 0
    neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    for start in sorted(occupied):
        if start in labels:
            continue
        component_id += 1
        stack = [start]
        labels[start] = component_id
        while stack:
            x_bin, y_bin = stack.pop()
            for dx, dy in neighbors:
                candidate = (x_bin + dx, y_bin + dy)
                if candidate in occupied and candidate not in labels:
                    labels[candidate] = component_id
                    stack.append(candidate)

    return labels


def assign_occupancy_components(cells, bin_size_um: float):
    import numpy as np

    coords = cells[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
    mins = coords.min(axis=0)
    bin_xy = np.floor((coords - mins) / bin_size_um).astype(int)
    bin_labels = _occupied_bin_components(bin_xy)
    return np.array([bin_labels[tuple(item)] for item in bin_xy], dtype=int)


def _component_sort_key(component: Component, order: str) -> tuple[float, float, int]:
    if order == "xy":
        return (component.x_centroid, component.y_centroid, component.raw_component_id)
    return (component.y_centroid, component.x_centroid, component.raw_component_id)


def summarize_components(
    cells,
    labels,
    *,
    sample_name: str,
    bin_size_um: float,
    min_component_cells: int,
    label_order: str,
) -> list[Component]:
    components: list[Component] = []
    for raw_component_id in sorted(set(labels.tolist())):
        mask = labels == raw_component_id
        comp_cells = cells.loc[mask, ["x_centroid", "y_centroid"]]
        cell_count = int(mask.sum())
        qc_status = "major" if cell_count >= min_component_cells else "tiny"
        components.append(
            Component(
                sample_name=sample_name,
                raw_component_id=int(raw_component_id),
                component_id="",
                bin_size_um=float(bin_size_um),
                cell_count=cell_count,
                x_min=float(comp_cells["x_centroid"].min()),
                x_max=float(comp_cells["x_centroid"].max()),
                y_min=float(comp_cells["y_centroid"].min()),
                y_max=float(comp_cells["y_centroid"].max()),
                x_centroid=float(comp_cells["x_centroid"].mean()),
                y_centroid=float(comp_cells["y_centroid"].mean()),
                assigned_label=None,
                slide_tissue_id="",
                qc_status=qc_status,
            )
        )

    major = [component for component in components if component.qc_status == "major"]
    expected_labels = SAMPLE_LABELS[sample_name]
    assignment_labels = SAMPLE_LABEL_ASSIGNMENT_ORDER[sample_name]
    ordered_major = sorted(
        major,
        key=lambda component: _component_sort_key(component, label_order),
    )
    label_by_raw = {}
    if len(ordered_major) == len(expected_labels):
        label_by_raw = {
            component.raw_component_id: label
            for component, label in zip(ordered_major, assignment_labels)
        }

    ordered_all = sorted(
        components,
        key=lambda component: _component_sort_key(component, label_order),
    )
    major_rank_by_raw = {
        component.raw_component_id: rank
        for rank, component in enumerate(ordered_major, start=1)
    }
    tiny_rank = 0
    finalized: list[Component] = []
    for component in ordered_all:
        assigned_label = label_by_raw.get(component.raw_component_id)
        if component.qc_status == "major":
            if assigned_label is None:
                component_id = (
                    f"{component.sample_name}_major"
                    f"{major_rank_by_raw[component.raw_component_id]:02d}"
                )
            else:
                component_id = f"{component.sample_name}_piece{assigned_label:02d}"
        else:
            tiny_rank += 1
            component_id = f"{component.sample_name}_tiny{tiny_rank:03d}"
        finalized.append(
            Component(
                sample_name=component.sample_name,
                raw_component_id=component.raw_component_id,
                component_id=component_id,
                bin_size_um=component.bin_size_um,
                cell_count=component.cell_count,
                x_min=component.x_min,
                x_max=component.x_max,
                y_min=component.y_min,
                y_max=component.y_max,
                x_centroid=component.x_centroid,
                y_centroid=component.y_centroid,
                assigned_label=assigned_label,
                slide_tissue_id=(
                    "" if assigned_label is None else SLIDE_TISSUE_IDS[assigned_label]
                ),
                qc_status=component.qc_status,
            )
        )
    return finalized


def choose_component_resolution(
    cells,
    *,
    sample_name: str,
    bin_sizes_um: Iterable[float],
    min_component_cells: int,
    label_order: str,
) -> tuple[list[Component], list[str]]:
    attempts: list[tuple[float, list[Component], int]] = []
    expected_count = len(SAMPLE_LABELS[sample_name])
    messages: list[str] = []

    for bin_size_um in bin_sizes_um:
        labels = assign_occupancy_components(cells, bin_size_um)
        components = summarize_components(
            cells,
            labels,
            sample_name=sample_name,
            bin_size_um=bin_size_um,
            min_component_cells=min_component_cells,
            label_order=label_order,
        )
        major_count = sum(component.qc_status == "major" for component in components)
        attempts.append((bin_size_um, components, major_count))
        messages.append(
            f"{sample_name}: {major_count} major components at {bin_size_um:g} um "
            f"(expected {expected_count})"
        )

    for _, components, major_count in attempts:
        if major_count == expected_count:
            return components, messages

    return attempts[0][1], messages


def _component_rows(components: Iterable[Component]) -> list[dict[str, object]]:
    rows = []
    for component in components:
        rows.append(
            {
                "sample_name": component.sample_name,
                "component_id": component.component_id,
                "raw_component_id": component.raw_component_id,
                "bin_size_um": component.bin_size_um,
                "qc_status": component.qc_status,
                "cell_count": component.cell_count,
                "x_min": component.x_min,
                "x_max": component.x_max,
                "y_min": component.y_min,
                "y_max": component.y_max,
                "x_centroid": component.x_centroid,
                "y_centroid": component.y_centroid,
                "assigned_label": (
                    "" if component.assigned_label is None else component.assigned_label
                ),
                "slide_tissue_id": component.slide_tissue_id,
            }
        )
    return rows


def write_component_summary(
    components: Iterable[Component],
    path: Path,
    *,
    include_tiny: bool = False,
) -> None:
    if include_tiny:
        summary_components = list(components)
    else:
        summary_components = [
            component for component in components if component.qc_status == "major"
        ]
    rows = _component_rows(summary_components)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_name",
        "component_id",
        "raw_component_id",
        "bin_size_um",
        "qc_status",
        "cell_count",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "x_centroid",
        "y_centroid",
        "assigned_label",
        "slide_tissue_id",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_component_approval_template(
    components: Iterable[Component],
    path: Path,
) -> None:
    rows = []
    for component in components:
        if component.qc_status != "major":
            continue
        rows.append(
            {
                "sample_name": component.sample_name,
                "component_id": component.component_id,
                "assigned_label": (
                    "" if component.assigned_label is None else component.assigned_label
                ),
                "slide_tissue_id": component.slide_tissue_id,
                "approved": "",
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=APPROVAL_TEMPLATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _convex_hull_points(points):
    import numpy as np
    from scipy.spatial import ConvexHull, QhullError

    points = np.unique(points, axis=0)
    if points.shape[0] < 3:
        return points
    try:
        hull = ConvexHull(points)
    except QhullError:
        return points
    return points[hull.vertices]


def _component_lookup(
    components: Iterable[Component],
) -> dict[tuple[str, int], Component]:
    return {
        (component.sample_name, component.raw_component_id): component
        for component in components
    }


def write_component_plot(
    sample_cells_and_labels: Iterable[tuple[SampleInput, object, object]],
    components: list[Component],
    path: Path,
    *,
    max_plot_cells_per_component: int,
    invert_y: bool,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    major_components = [
        component for component in components if component.qc_status == "major"
    ]
    if not major_components:
        raise ValueError("No major components to plot.")

    labels = [component.assigned_label for component in major_components]
    color_values = sorted(label for label in labels if label is not None)
    cmap = plt.get_cmap("tab20", max(len(color_values), 1))
    color_by_label = {
        label: cmap(index % cmap.N)
        for index, label in enumerate(color_values)
    }
    component_by_raw = _component_lookup(components)

    fig, ax = plt.subplots(figsize=(12, 10), dpi=180)
    rng = np.random.default_rng(0)

    for sample, cells, raw_labels in sample_cells_and_labels:
        for raw_component_id in sorted(set(raw_labels.tolist())):
            component = component_by_raw[(sample.name, int(raw_component_id))]
            mask = raw_labels == raw_component_id
            coords = cells.loc[mask, ["x_centroid", "y_centroid"]].to_numpy(dtype=float)

            if component.qc_status == "tiny":
                ax.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    s=0.15,
                    c="#bdbdbd",
                    alpha=0.25,
                    linewidths=0,
                )
                continue

            if coords.shape[0] > max_plot_cells_per_component:
                keep = rng.choice(
                    coords.shape[0],
                    size=max_plot_cells_per_component,
                    replace=False,
                )
                plot_coords = coords[keep]
            else:
                plot_coords = coords

            color = color_by_label.get(component.assigned_label, "#4c78a8")
            ax.scatter(
                plot_coords[:, 0],
                plot_coords[:, 1],
                s=0.18,
                color=color,
                alpha=0.55,
                linewidths=0,
            )

            hull = _convex_hull_points(coords)
            if hull.shape[0] >= 3:
                hull = np.vstack([hull, hull[0]])
                ax.plot(hull[:, 0], hull[:, 1], color=color, linewidth=1.4)

            text = (
                f"{component.assigned_label}\n{component.slide_tissue_id}"
                if component.assigned_label is not None
                else component.component_id
            )
            ax.text(
                component.x_centroid,
                component.y_centroid,
                text,
                ha="center",
                va="center",
                fontsize=7,
                color="black",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "alpha": 0.75,
                    "edgecolor": "none",
                },
            )

    ax.set_xlabel("x centroid (um)")
    ax.set_ylabel("y centroid (um)")
    ax.set_title("Slide 3 tissue component QC: Patient4 and Patient6 run targets")
    ax.set_aspect("equal", adjustable="box")
    if invert_y:
        ax.invert_yaxis()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def detect_all_components(
    samples: Iterable[SampleInput],
    *,
    bin_sizes_um: Iterable[float],
    min_component_cells: int,
    label_order: str,
):
    all_components: list[Component] = []
    sample_cells_and_labels = []
    messages: list[str] = []

    for sample in samples:
        cells = load_cells(sample.cells_path)
        components, sample_messages = choose_component_resolution(
            cells,
            sample_name=sample.name,
            bin_sizes_um=bin_sizes_um,
            min_component_cells=min_component_cells,
            label_order=label_order,
        )
        selected_bin_size = (
            components[0].bin_size_um if components else DEFAULT_BIN_SIZES_UM[0]
        )
        labels = assign_occupancy_components(cells, selected_bin_size)
        all_components.extend(components)
        sample_cells_and_labels.append((sample, cells, labels))
        messages.extend(sample_messages)

    return all_components, sample_cells_and_labels, messages


def validate_expected_counts(components: Iterable[Component]) -> list[str]:
    by_sample: dict[str, list[Component]] = {}
    for component in components:
        by_sample.setdefault(component.sample_name, []).append(component)

    errors: list[str] = []
    for sample_name, sample_components in by_sample.items():
        expected_labels = SAMPLE_LABELS[sample_name]
        major = [
            component
            for component in sample_components
            if component.qc_status == "major"
        ]
        assigned = sorted(
            component.assigned_label
            for component in major
            if component.assigned_label is not None
        )
        if len(major) != len(expected_labels):
            errors.append(
                f"{sample_name}: detected {len(major)} major components; "
                f"expected {len(expected_labels)}."
            )
        elif assigned != expected_labels:
            errors.append(
                f"{sample_name}: assigned labels {assigned}; "
                f"expected {expected_labels}."
            )
    return errors


def _read_manifest(path: Path) -> dict[tuple[str, str], int]:
    import pandas as pd

    df = pd.read_csv(path)
    required = {"sample_name", "component_id", "assigned_label", "approved"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Approved manifest is missing columns: {', '.join(sorted(missing))}"
        )

    manifest: dict[tuple[str, str], int] = {}
    for _, row in df.iterrows():
        approved = str(row["approved"]).strip().lower()
        if approved not in TRUE_VALUES:
            continue
        sample_name = str(row["sample_name"]).strip()
        component_id = str(row["component_id"]).strip()
        label = int(row["assigned_label"])
        manifest[(sample_name, component_id)] = label
    return manifest


def validate_manifest(
    path: Path,
    components: Iterable[Component],
) -> dict[tuple[str, str], int]:
    manifest = _read_manifest(path)
    if not manifest:
        raise ValueError("Approved manifest has no rows with approved=yes.")

    component_by_key = {
        (component.sample_name, component.component_id): component
        for component in components
        if component.qc_status == "major"
    }
    unknown = [
        f"{sample_name}/{component_id}"
        for sample_name, component_id in manifest
        if (sample_name, component_id) not in component_by_key
    ]
    mismatched = [
        f"{sample_name}/{component_id}"
        for (sample_name, component_id), assigned_label in manifest.items()
        if (sample_name, component_id) in component_by_key
        and assigned_label
        != component_by_key[(sample_name, component_id)].assigned_label
    ]

    if unknown or mismatched:
        parts = []
        if unknown:
            parts.append(f"unknown approved components: {', '.join(unknown)}")
        if mismatched:
            parts.append(f"label mismatches for: {', '.join(mismatched)}")
        raise ValueError(
            "Approved manifest does not match current QC components: "
            + "; ".join(parts)
        )
    return manifest


def _repo_relative_path(path: Path, *, repo_root: Path | None = None) -> str:
    root = (repo_root or _repo_root()).resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Path must be inside the repository for SGE /app binding: {resolved}"
        ) from exc


def _component_by_manifest_key(
    components: Iterable[Component],
) -> dict[tuple[str, str], Component]:
    return {
        (component.sample_name, component.component_id): component
        for component in components
        if component.qc_status == "major"
    }


def write_piece_run_manifest(
    outdir: Path,
    components: Iterable[Component],
    manifest: dict[tuple[str, str], int],
    counts: dict[tuple[str, str], int],
    *,
    repo_root: Path | None = None,
) -> Path:
    root = repo_root or _repo_root()
    component_by_key = _component_by_manifest_key(components)
    run_manifest_path = outdir / "piece_run_manifest.csv"
    tracer_outdir = root / "tutorials" / "gbm" / "output" / "slide3_tracer"
    run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tracer_outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    task_id = 1
    for key, label in sorted(manifest.items(), key=lambda item: item[1]):
        rows_written = counts.get(key, 0)
        if rows_written <= 0:
            continue
        sample_name, component_id = key
        component = component_by_key[key]
        input_path = outdir / f"slide3_piece_{label:02d}_{sample_name}.parquet"
        output_path = (
            tracer_outdir / f"slide3_piece_{label:02d}_{sample_name}_tracer.parquet"
        )
        rows.append(
            {
                "task_id": task_id,
                "sample_name": sample_name,
                "component_id": component_id,
                "assigned_label": label,
                "slide_tissue_id": component.slide_tissue_id,
                "input_piece_parquet": _repo_relative_path(
                    input_path,
                    repo_root=repo_root,
                ),
                "output_tracer_parquet": _repo_relative_path(
                    output_path,
                    repo_root=repo_root,
                ),
                "rows_written": rows_written,
            }
        )
        task_id += 1

    with run_manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PIECE_RUN_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return run_manifest_path


def _component_hulls(
    sample_cells_and_labels,
    components: Iterable[Component],
    manifest,
):
    from matplotlib.path import Path as MplPath
    import numpy as np
    from shapely.geometry import Polygon

    component_by_raw = _component_lookup(components)
    hulls = {}
    for sample, cells, raw_labels in sample_cells_and_labels:
        for raw_component_id in sorted(set(raw_labels.tolist())):
            component = component_by_raw[(sample.name, int(raw_component_id))]
            key = (component.sample_name, component.component_id)
            if key not in manifest:
                continue
            coords = cells.loc[
                raw_labels == raw_component_id,
                ["x_centroid", "y_centroid"],
            ].to_numpy(dtype=float)
            hull = _convex_hull_points(coords)
            if hull.shape[0] < 3:
                xmin, ymin = coords.min(axis=0)
                xmax, ymax = coords.max(axis=0)
                hull = np.array(
                    [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
                )
            polygon = Polygon(hull).buffer(0)
            hulls[key] = MplPath(np.asarray(polygon.exterior.coords, dtype=float))
    return hulls


def _normalize_transcript_batch(table):
    import pyarrow as pa

    rename_pairs = []
    names = table.schema.names
    for src, dst in TRANSCRIPT_COORD_ALIASES.items():
        if src in names and dst not in names:
            rename_pairs.append((src, dst))
    if not rename_pairs:
        return table

    arrays = []
    fields = []
    rename_map = dict(rename_pairs)
    for field, column in zip(table.schema, table.itercolumns()):
        new_name = rename_map.get(field.name, field.name)
        fields.append(pa.field(new_name, field.type, field.nullable, field.metadata))
        arrays.append(column)
    return pa.Table.from_arrays(
        arrays,
        schema=pa.schema(fields, metadata=table.schema.metadata),
    )


def write_piece_parquets(
    samples: Iterable[SampleInput],
    sample_cells_and_labels,
    components: list[Component],
    manifest: dict[tuple[str, str], int],
    outdir: Path,
    *,
    hull_buffer_um: float,
    batch_size: int,
) -> Path:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from shapely.geometry import Polygon

    outdir.mkdir(parents=True, exist_ok=True)
    hull_paths = _component_hulls(sample_cells_and_labels, components, manifest)

    # Apply the requested buffer, then use matplotlib Path for fast batch tests.
    buffered_paths = {}
    for key, path in hull_paths.items():
        vertices = np.asarray(path.vertices, dtype=float)
        polygon = Polygon(vertices).buffer(hull_buffer_um)
        from matplotlib.path import Path as MplPath

        buffered_paths[key] = MplPath(
            np.asarray(polygon.exterior.coords, dtype=float)
        )

    sample_by_name = {sample.name: sample for sample in samples}
    writers: dict[tuple[str, str], pq.ParquetWriter] = {}
    counts: dict[tuple[str, str], int] = {}

    try:
        for sample_name, sample in sample_by_name.items():
            if not sample.transcripts_path.exists():
                raise FileNotFoundError(
                    f"Missing transcripts.parquet for {sample.name}: "
                    f"{sample.transcripts_path}"
                )

            parquet_file = pq.ParquetFile(sample.transcripts_path)
            sample_keys = [key for key in manifest if key[0] == sample_name]
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                table = _normalize_transcript_batch(pa.Table.from_batches([batch]))
                names = table.schema.names
                if "x" not in names or "y" not in names:
                    raise ValueError(
                        f"{sample.transcripts_path} must contain x/y or "
                        "x_location/y_location columns."
                    )
                x = table["x"].to_numpy(zero_copy_only=False)
                y = table["y"].to_numpy(zero_copy_only=False)
                points = np.column_stack([x, y])

                for key in sample_keys:
                    keep = buffered_paths[key].contains_points(points)
                    if not keep.any():
                        continue
                    piece_table = table.filter(pa.array(keep))
                    label = manifest[key]
                    output_path = (
                        outdir / f"slide3_piece_{label:02d}_{sample_name}.parquet"
                    )
                    if key not in writers:
                        writers[key] = pq.ParquetWriter(output_path, piece_table.schema)
                        counts[key] = 0
                    writers[key].write_table(piece_table)
                    counts[key] += int(keep.sum())
    finally:
        for writer in writers.values():
            writer.close()

    manifest_path = outdir / "piece_write_summary.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_name",
                "component_id",
                "assigned_label",
                "rows_written",
            ],
        )
        writer.writeheader()
        for key, label in sorted(manifest.items(), key=lambda item: item[1]):
            writer.writerow(
                {
                    "sample_name": key[0],
                    "component_id": key[1],
                    "assigned_label": label,
                    "rows_written": counts.get(key, 0),
                }
            )
    run_manifest_path = write_piece_run_manifest(outdir, components, manifest, counts)
    return run_manifest_path


def main() -> None:
    args = _parse_args()
    bin_sizes_um = _parse_bin_sizes(args.bin_sizes_um)
    samples = resolve_samples(args.xenium_output)
    outdir = (
        Path(args.outdir).expanduser().resolve()
        if args.outdir
        else _default_qc_outdir()
    )

    components, sample_cells_and_labels, messages = detect_all_components(
        samples,
        bin_sizes_um=bin_sizes_um,
        min_component_cells=args.min_component_cells,
        label_order=args.component_sort,
    )

    summary_path = outdir / "component_summary.csv"
    diagnostics_path = outdir / "component_diagnostics.csv"
    approval_path = outdir / "component_approval_template.csv"
    plot_path = outdir / "component_plot.png"
    write_component_summary(components, summary_path)
    write_component_summary(components, diagnostics_path, include_tiny=True)
    write_component_approval_template(components, approval_path)
    write_component_plot(
        sample_cells_and_labels,
        components,
        plot_path,
        max_plot_cells_per_component=args.max_plot_cells_per_component,
        invert_y=not args.no_invert_y,
    )

    print("\n".join(messages))
    print(f"Saved component summary to: {summary_path}")
    print(f"Saved component diagnostics to: {diagnostics_path}")
    print(f"Saved component approval template to: {approval_path}")
    print(f"Saved component plot to: {plot_path}")

    errors = validate_expected_counts(components)
    if errors:
        raise SystemExit("QC failed:\n" + "\n".join(errors))

    if args.qc_only:
        print("QC passed. No transcript parquets were written.")
        return

    if not args.approved_manifest:
        raise SystemExit("--write-pieces requires --approved-manifest.")

    manifest = validate_manifest(Path(args.approved_manifest), components)
    pieces_outdir = (
        Path(args.pieces_outdir).expanduser().resolve()
        if args.pieces_outdir
        else _default_pieces_outdir()
    )
    run_manifest_path = write_piece_parquets(
        samples,
        sample_cells_and_labels,
        components,
        manifest,
        pieces_outdir,
        hull_buffer_um=args.hull_buffer_um,
        batch_size=args.transcript_batch_size,
    )
    print(f"Saved per-piece transcript parquets to: {pieces_outdir}")
    print(f"Saved SGE task manifest to: {run_manifest_path}")


if __name__ == "__main__":
    main()
