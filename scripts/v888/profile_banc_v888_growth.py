#!/usr/bin/env python3
"""Phase-0 BANC v888 growth-oriented dataset profiler.

This script is intentionally exploratory. It characterizes the adult connectome
and lineage-associated morphology/connectivity proxies; it does not infer a
causal mechanism of brain growth from a single adult snapshot.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

METRIC_COLUMNS = (
    "l2_nodes",
    "l2_cable_length_um",
    "volume_nm3",
    "input_connections",
    "output_connections",
    "mitochondria",
    "mitochondria_volume",
    "pd_width",
    "segregation_index",
)

SUM_METRICS = (
    "l2_nodes",
    "l2_cable_length_um",
    "volume_nm3",
    "input_connections",
    "output_connections",
    "mitochondria",
    "mitochondria_volume",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "datasets" / "banc_v888",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def feather_info(path: Path) -> dict[str, object]:
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)
        rows = 0
        for index in range(reader.num_record_batches):
            rows += reader.get_batch(index).num_rows
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "rows": rows,
            "record_batches": reader.num_record_batches,
            "schema": [(field.name, str(field.type)) for field in reader.schema],
        }


def parquet_info(path: Path) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "columns": parquet.schema.names,
        "schema": str(parquet.schema_arrow),
    }


def find_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    by_lower = {name.lower(): name for name in columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [
            "__".join(str(part) for part in column if str(part))
            for column in frame.columns.to_flat_index()
        ]
    return frame


def compute_scaling(
    grouped: pd.DataFrame,
    count_column: str,
    total_columns: list[str],
) -> dict[str, object]:
    result: dict[str, object] = {}
    n = pd.to_numeric(grouped[count_column], errors="coerce").to_numpy(dtype=float)
    for column in total_columns:
        y = pd.to_numeric(grouped[column], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(n) & np.isfinite(y) & (n > 0.0) & (y > 0.0)
        if int(mask.sum()) < 3:
            continue
        slope, intercept = np.polyfit(np.log(n[mask]), np.log(y[mask]), 1)
        predicted = slope * np.log(n[mask]) + intercept
        residual = np.log(y[mask]) - predicted
        ss_res = float(np.sum(residual * residual))
        centered = np.log(y[mask]) - float(np.mean(np.log(y[mask])))
        ss_tot = float(np.sum(centered * centered))
        result[column] = {
            "groups": int(mask.sum()),
            "log_log_slope": float(slope),
            "log_log_intercept": float(intercept),
            "r2": None if ss_tot <= 0.0 else float(1.0 - ss_res / ss_tot),
        }
    return result


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else data_root / "analysis" / "brain_growth_phase0"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    files = {
        "meta": data_root / "banc_888_meta.feather",
        "metrics": data_root / "banc_888_metrics.feather",
        "edgelist_v3": data_root / "banc_888_edgelist_simple_v3.feather",
        "edgelist_split_v2": data_root / "banc_888_edgelist_split_v2.feather",
        "synapses_v3": data_root / "banc_888_synapses_v3_enriched.parquet",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required BANC v888 files: {missing}")

    profile: dict[str, object] = {
        "dataset": "BANC",
        "snapshot": "v888",
        "analysis_scope": "adult_snapshot_exploratory_growth_proxies_not_causal_growth",
        "files": {
            "meta": feather_info(files["meta"]),
            "metrics": feather_info(files["metrics"]),
            "edgelist_v3": feather_info(files["edgelist_v3"]),
            "edgelist_split_v2": feather_info(files["edgelist_split_v2"]),
            "synapses_v3": parquet_info(files["synapses_v3"]),
        },
    }

    meta = feather.read_feather(files["meta"])
    metrics = feather.read_feather(files["metrics"])

    meta_columns = [str(column) for column in meta.columns]
    metrics_columns = [str(column) for column in metrics.columns]
    id_meta = find_column(meta_columns, ("banc_888_id", "root_id", "root_888"))
    id_metrics = find_column(metrics_columns, ("banc_888_id", "root_id", "root_888"))
    hemilineage = find_column(meta_columns, ("hemilineage",))
    region = find_column(meta_columns, ("region",))

    profile["resolved_columns"] = {
        "meta_id": id_meta,
        "metrics_id": id_metrics,
        "hemilineage": hemilineage,
        "region": region,
        "growth_metrics_present": [
            name for name in METRIC_COLUMNS if name in metrics.columns
        ],
    }

    if id_meta is None or id_metrics is None:
        raise ValueError("could not resolve BANC v888 primary key in meta/metrics tables")
    if hemilineage is None:
        raise ValueError("BANC v888 metadata does not expose a hemilineage column")

    meta_subset_columns = [id_meta, hemilineage]
    if region is not None:
        meta_subset_columns.append(region)
    metric_payload_columns = [
        name for name in METRIC_COLUMNS if name in metrics.columns
    ]
    metrics_subset = metrics[[id_metrics, *metric_payload_columns]].copy()
    joined = meta[meta_subset_columns].merge(
        metrics_subset,
        left_on=id_meta,
        right_on=id_metrics,
        how="inner",
        validate="one_to_one",
    )

    joined[hemilineage] = joined[hemilineage].astype("string").str.strip()
    lineage = joined[
        joined[hemilineage].notna()
        & (joined[hemilineage] != "")
        & (joined[hemilineage].str.lower() != "nan")
    ].copy()

    numeric_metrics = [name for name in METRIC_COLUMNS if name in lineage.columns]
    aggregation: dict[str, list[str]] = {
        name: ["sum", "mean", "median"] for name in numeric_metrics
    }
    grouped = lineage.groupby(hemilineage, dropna=True).agg(aggregation)
    grouped = flatten_columns(grouped)
    neuron_counts = lineage.groupby(hemilineage).size().rename("n_neurons")
    grouped.insert(0, "n_neurons", neuron_counts.reindex(grouped.index).astype(int))
    grouped = grouped.sort_values(
        by=["n_neurons", "l2_cable_length_um__sum"]
        if "l2_cable_length_um__sum" in grouped.columns
        else ["n_neurons"],
        ascending=False,
    )
    grouped.to_csv(output_root / "hemilineage_growth_proxy.csv")

    total_columns = [
        f"{name}__sum"
        for name in SUM_METRICS
        if f"{name}__sum" in grouped.columns
    ]
    profile["hemilineage"] = {
        "annotated_neurons": int(len(lineage)),
        "groups": int(len(grouped)),
        "scaling": compute_scaling(grouped, "n_neurons", total_columns),
    }

    if numeric_metrics:
        correlations = lineage[numeric_metrics].corr(
            method="spearman",
            min_periods=10,
        )
        correlations.to_csv(output_root / "neuron_metric_spearman.csv")

    if region is not None:
        region_lineage = (
            lineage.groupby([region, hemilineage], dropna=True)
            .size()
            .rename("n_neurons")
            .reset_index()
            .sort_values(["region", "n_neurons"], ascending=[True, False])
        )
        region_lineage.to_csv(
            output_root / "region_hemilineage_counts.csv",
            index=False,
        )

    top_columns = [
        "n_neurons",
        "l2_cable_length_um__sum",
        "volume_nm3__sum",
        "input_connections__sum",
        "output_connections__sum",
    ]
    available_top = [column for column in top_columns if column in grouped.columns]
    profile["top_hemilineages"] = (
        grouped[available_top]
        .head(25)
        .reset_index()
        .replace({np.nan: None})
        .to_dict(orient="records")
    )

    profile_path = output_root / "phase0_profile.json"
    profile_path.write_text(
        json.dumps(profile, indent=2, sort_keys=True, default=finite_float),
        encoding="utf-8",
    )

    print(f"profile={profile_path}")
    print(f"hemilineages={len(grouped)}")
    print(
        "synapse_rows_v3="
        f"{profile['files']['synapses_v3']['rows']}"
    )
    print(
        "edgelist_rows_v3="
        f"{profile['files']['edgelist_v3']['rows']}"
    )


if __name__ == "__main__":
    main()
