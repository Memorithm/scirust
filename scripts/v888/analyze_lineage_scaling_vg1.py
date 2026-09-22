#!/usr/bin/env python3
"""VG-1 lineage-size-adjusted adult BANC v888 analysis.

This stage asks which adult hemilineages allocate unusually high or low
morphology/connectivity after controlling for annotated neuron count. It remains
an adult-snapshot association analysis and does not infer developmental cause.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

METRICS = (
    "l2_nodes",
    "l2_cable_length_um",
    "volume_nm3",
    "input_connections",
    "output_connections",
    "branchpoints",
    "endpoints",
    "axon_length",
    "dend_length",
)

MIN_REGION_GROUPS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "datasets" / "banc_v888",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def fit_log_log(frame: pd.DataFrame, metric: str) -> dict[str, float | int | None]:
    x = pd.to_numeric(frame["n_neurons"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame[f"{metric}__sum"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if int(valid.sum()) < 3:
        return {
            "groups": int(valid.sum()),
            "slope": None,
            "intercept": None,
            "r2": None,
            "residual_rms": None,
        }

    lx = np.log(x[valid])
    ly = np.log(y[valid])
    slope, intercept = np.polyfit(lx, ly, 1)
    prediction = intercept + slope * lx
    residual = ly - prediction
    ss_res = float(np.sum(residual * residual))
    centered = ly - float(np.mean(ly))
    ss_total = float(np.sum(centered * centered))
    return {
        "groups": int(valid.sum()),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": None if ss_total <= 0.0 else float(1.0 - ss_res / ss_total),
        "residual_rms": float(np.sqrt(np.mean(residual * residual))),
    }


def add_residual(
    frame: pd.DataFrame,
    metric: str,
    fit: dict[str, float | int | None],
) -> None:
    slope = fit["slope"]
    intercept = fit["intercept"]
    column = f"{metric}__residual_log"
    frame[column] = np.nan
    if slope is None or intercept is None:
        return

    x = pd.to_numeric(frame["n_neurons"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame[f"{metric}__sum"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    frame.loc[valid, column] = (
        np.log(y[valid]) - (float(intercept) + float(slope) * np.log(x[valid]))
    )


def label_class(label: object) -> str:
    text = str(label).strip().lower()
    if "putative" in text:
        return "putative"
    if "_or_" in text or " or " in text:
        return "ambiguous_or"
    if not text or text == "nan":
        return "missing"
    return "declared"


def group_metrics(
    joined: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    aggregation = {metric: "sum" for metric in METRICS if metric in joined.columns}
    grouped = joined.groupby(group_columns, dropna=True).agg(aggregation)
    grouped.columns = [f"{column}__sum" for column in grouped.columns]
    counts = joined.groupby(group_columns, dropna=True).size().rename("n_neurons")
    grouped.insert(0, "n_neurons", counts.reindex(grouped.index).astype(int))
    return grouped.reset_index()


def fit_panel(frame: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    return {
        metric: fit_log_log(frame, metric)
        for metric in METRICS
        if f"{metric}__sum" in frame.columns
    }


def residual_extremes(
    frame: pd.DataFrame,
    metrics: list[str],
    *,
    n: int = 15,
) -> dict[str, dict[str, list[dict[str, object]]]]:
    result: dict[str, dict[str, list[dict[str, object]]]] = {}
    identity = ["hemilineage", "n_neurons", "label_class"]
    for metric in metrics:
        column = f"{metric}__residual_log"
        available = frame.dropna(subset=[column]).copy()
        if available.empty:
            continue
        fields = identity + [f"{metric}__sum", column]
        high = available.nlargest(n, column)[fields].replace({np.nan: None})
        low = available.nsmallest(n, column)[fields].replace({np.nan: None})
        result[metric] = {
            "over_allocated": high.to_dict(orient="records"),
            "under_allocated": low.to_dict(orient="records"),
        }
    return result


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else data_root / "analysis" / "brain_growth_vg1"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    meta_path = data_root / "banc_888_meta.feather"
    metrics_path = data_root / "banc_888_metrics.feather"
    if not meta_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("BANC v888 metadata/metrics are required for VG-1")

    meta = feather.read_feather(meta_path)
    metrics = feather.read_feather(metrics_path)
    required_meta = {"banc_888_id", "hemilineage", "region"}
    missing_meta = sorted(required_meta - set(meta.columns))
    if missing_meta:
        raise ValueError(f"missing required metadata columns: {missing_meta}")

    present_metrics = [metric for metric in METRICS if metric in metrics.columns]
    if len(present_metrics) < 5:
        raise ValueError("too few growth metrics available for VG-1")

    joined = meta[["banc_888_id", "hemilineage", "region"]].merge(
        metrics[["banc_888_id", *present_metrics]],
        on="banc_888_id",
        how="inner",
        validate="one_to_one",
    )
    joined["hemilineage"] = joined["hemilineage"].astype("string").str.strip()
    joined["region"] = joined["region"].astype("string").str.strip()
    joined = joined[
        joined["hemilineage"].notna()
        & (joined["hemilineage"] != "")
        & (joined["hemilineage"].str.lower() != "nan")
    ].copy()

    global_groups = group_metrics(joined, ["hemilineage"])
    global_groups["label_class"] = global_groups["hemilineage"].map(label_class)
    global_fit = fit_panel(global_groups)
    for metric, fit in global_fit.items():
        add_residual(global_groups, metric, fit)
    global_groups = global_groups.sort_values(
        ["n_neurons", "hemilineage"],
        ascending=[False, True],
    )
    global_groups.to_csv(output_root / "hemilineage_size_adjusted.csv", index=False)

    sensitivity_frames = {
        "all": global_groups,
        "declared_only": global_groups[global_groups["label_class"] == "declared"],
        "n_ge_5": global_groups[global_groups["n_neurons"] >= 5],
        "n_ge_20": global_groups[global_groups["n_neurons"] >= 20],
        "declared_n_ge_20": global_groups[
            (global_groups["label_class"] == "declared")
            & (global_groups["n_neurons"] >= 20)
        ],
    }
    sensitivity = {
        name: fit_panel(frame)
        for name, frame in sensitivity_frames.items()
        if len(frame) >= 3
    }

    region_groups = group_metrics(joined, ["region", "hemilineage"])
    region_groups["label_class"] = region_groups["hemilineage"].map(label_class)
    region_groups.to_csv(output_root / "region_hemilineage_metrics.csv", index=False)

    region_fits: dict[str, object] = {}
    for region, frame in region_groups.groupby("region", dropna=True):
        if len(frame) < MIN_REGION_GROUPS:
            continue
        fits = fit_panel(frame)
        for metric, fit in fits.items():
            add_residual(frame, metric, fit)
        frame.to_csv(
            output_root / f"region_{str(region).replace('/', '_')}_residuals.csv",
            index=False,
        )
        region_fits[str(region)] = {
            "groups": int(len(frame)),
            "fits": fits,
        }

    report = {
        "dataset": "BANC v888",
        "scope": "adult_snapshot_lineage_size_adjusted_association_not_causality",
        "joined_neurons": int(len(joined)),
        "hemilineage_groups": int(len(global_groups)),
        "metrics": present_metrics,
        "label_class_counts": {
            str(key): int(value)
            for key, value in global_groups["label_class"].value_counts().items()
        },
        "global_fits": global_fit,
        "sensitivity": sensitivity,
        "region_fits": region_fits,
        "residual_extremes": residual_extremes(global_groups, present_metrics),
    }
    path = output_root / "vg1_report.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"report={path}")
    print(f"joined_neurons={len(joined)}")
    print(f"hemilineages={len(global_groups)}")
    print(f"regions={len(region_fits)}")


if __name__ == "__main__":
    main()
