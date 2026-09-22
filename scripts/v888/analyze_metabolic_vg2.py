#!/usr/bin/env python3
"""VG-2 mitochondrial coverage and metabolic-scaling audit for BANC v888.

This stage explicitly separates null, zero and positive mitochondrial fields
before any association analysis. A zero value is treated as a recorded zero,
not automatically as evidence that the neuron biologically lacks mitochondria.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

STRUCTURAL_METRICS = (
    "l2_cable_length_um",
    "volume_nm3",
    "input_connections",
    "output_connections",
    "branchpoints",
    "endpoints",
    "axon_length",
    "dend_length",
)

MITO_METRICS = ("mitochondria", "mitochondria_volume")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "datasets" / "banc_v888",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def finite_positive(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.notna() & np.isfinite(values) & (values > 0)


def state_counts(series: pd.Series) -> dict[str, int]:
    values = pd.to_numeric(series, errors="coerce")
    null = values.isna() | ~np.isfinite(values)
    zero = (~null) & (values == 0)
    positive = (~null) & (values > 0)
    negative = (~null) & (values < 0)
    return {
        "rows": int(len(values)),
        "null_or_nonfinite": int(null.sum()),
        "zero": int(zero.sum()),
        "positive": int(positive.sum()),
        "negative": int(negative.sum()),
    }


def log_log_fit(x: pd.Series, y: pd.Series) -> dict[str, float | int | None]:
    xx = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    yy = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(xx) & np.isfinite(yy) & (xx > 0.0) & (yy > 0.0)
    if int(valid.sum()) < 3:
        return {
            "n": int(valid.sum()),
            "slope": None,
            "intercept": None,
            "r2": None,
            "residual_rms": None,
        }

    lx = np.log(xx[valid])
    ly = np.log(yy[valid])
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = intercept + slope * lx
    resid = ly - pred
    ss_res = float(np.sum(resid * resid))
    centered = ly - float(np.mean(ly))
    ss_tot = float(np.sum(centered * centered))
    return {
        "n": int(valid.sum()),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": None if ss_tot <= 0.0 else float(1.0 - ss_res / ss_tot),
        "residual_rms": float(np.sqrt(np.mean(resid * resid))),
    }


def spearman_panel(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float | None]]:
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(method="spearman", min_periods=100)
    return {
        str(row): {
            str(col): (
                None
                if pd.isna(corr.loc[row, col])
                else float(corr.loc[row, col])
            )
            for col in corr.columns
        }
        for row in corr.index
    }


def group_coverage(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for label, subset in frame.groupby(group, dropna=False):
        record: dict[str, object] = {
            group: None if pd.isna(label) else str(label),
            "n_neurons": int(len(subset)),
        }
        for metric in MITO_METRICS:
            counts = state_counts(subset[metric])
            record[f"{metric}__positive"] = counts["positive"]
            record[f"{metric}__zero"] = counts["zero"]
            record[f"{metric}__null"] = counts["null_or_nonfinite"]
            record[f"{metric}__positive_fraction"] = (
                counts["positive"] / counts["rows"] if counts["rows"] else 0.0
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def structural_comparison(
    frame: pd.DataFrame,
    mito_metric: str,
    structural_metrics: list[str],
) -> dict[str, object]:
    values = pd.to_numeric(frame[mito_metric], errors="coerce")
    state = pd.Series("null", index=frame.index, dtype="string")
    finite = values.notna() & np.isfinite(values)
    state.loc[finite & (values == 0)] = "zero"
    state.loc[finite & (values > 0)] = "positive"
    state.loc[finite & (values < 0)] = "negative"

    output: dict[str, object] = {}
    for metric in structural_metrics:
        numeric = pd.to_numeric(frame[metric], errors="coerce")
        summary: dict[str, object] = {}
        for status in ("positive", "zero", "null"):
            selected = numeric[state == status]
            selected = selected[np.isfinite(selected)]
            summary[status] = {
                "n": int(len(selected)),
                "median": None if selected.empty else float(selected.median()),
                "mean": None if selected.empty else float(selected.mean()),
            }
        output[metric] = summary
    return output


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else data_root / "analysis" / "brain_growth_vg2"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    meta_path = data_root / "banc_888_meta.feather"
    metrics_path = data_root / "banc_888_metrics.feather"
    if not meta_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("BANC v888 metadata and metrics are required")

    meta = feather.read_feather(meta_path)
    metrics = feather.read_feather(metrics_path)

    required = {"banc_888_id", *MITO_METRICS}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"missing required mitochondrial columns: {missing}")

    structural = [metric for metric in STRUCTURAL_METRICS if metric in metrics.columns]
    meta_columns = ["banc_888_id"]
    for column in ("region", "hemilineage"):
        if column in meta.columns:
            meta_columns.append(column)

    joined = meta[meta_columns].merge(
        metrics[["banc_888_id", *MITO_METRICS, *structural]],
        on="banc_888_id",
        how="inner",
        validate="one_to_one",
    )

    for column in ("region", "hemilineage"):
        if column in joined.columns:
            joined[column] = joined[column].astype("string").str.strip()

    coverage: dict[str, object] = {
        metric: state_counts(joined[metric]) for metric in MITO_METRICS
    }

    both_positive = finite_positive(joined["mitochondria"]) & finite_positive(
        joined["mitochondria_volume"]
    )
    count_positive = finite_positive(joined["mitochondria"])
    volume_positive = finite_positive(joined["mitochondria_volume"])
    coverage["pairing"] = {
        "both_positive": int(both_positive.sum()),
        "count_positive_only": int((count_positive & ~volume_positive).sum()),
        "volume_positive_only": int((volume_positive & ~count_positive).sum()),
        "neither_positive": int((~count_positive & ~volume_positive).sum()),
    }

    region_path = output_root / "mitochondrial_coverage_by_region.csv"
    if "region" in joined.columns:
        region_coverage = group_coverage(joined, "region")
        region_coverage.to_csv(region_path, index=False)
    else:
        region_coverage = pd.DataFrame()

    lineage_path = output_root / "mitochondrial_coverage_by_hemilineage.csv"
    if "hemilineage" in joined.columns:
        lineage = joined[
            joined["hemilineage"].notna()
            & (joined["hemilineage"] != "")
            & (joined["hemilineage"].str.lower() != "nan")
        ].copy()
        lineage_coverage = group_coverage(lineage, "hemilineage")
        lineage_coverage.to_csv(lineage_path, index=False)
    else:
        lineage = joined.iloc[0:0].copy()
        lineage_coverage = pd.DataFrame()

    complete = joined[both_positive].copy()
    positive_columns = [*MITO_METRICS, *structural]
    spearman = spearman_panel(complete, positive_columns)

    per_neuron_scaling: dict[str, object] = {}
    for mito in MITO_METRICS:
        per_neuron_scaling[mito] = {
            metric: log_log_fit(complete[metric], complete[mito])
            for metric in structural
        }

    region_scaling: dict[str, object] = {}
    if "region" in complete.columns:
        for region, frame in complete.groupby("region", dropna=True):
            if len(frame) < 100:
                continue
            region_scaling[str(region)] = {
                mito: {
                    metric: log_log_fit(frame[metric], frame[mito])
                    for metric in structural
                }
                for mito in MITO_METRICS
            }

    bias_audit = {
        mito: structural_comparison(joined, mito, structural)
        for mito in MITO_METRICS
    }

    report = {
        "dataset": "BANC v888",
        "scope": "adult_snapshot_mitochondrial_coverage_and_association_not_causality",
        "joined_neurons": int(len(joined)),
        "structural_metrics": structural,
        "coverage": coverage,
        "complete_positive_rows": int(len(complete)),
        "complete_positive_fraction": float(len(complete) / len(joined)),
        "spearman_complete_positive": spearman,
        "per_neuron_log_log_scaling": per_neuron_scaling,
        "region_scaling": region_scaling,
        "coverage_bias_audit": bias_audit,
        "warnings": [
            "zero mitochondrial fields are not assumed to mean biological absence",
            "associations are computed only after explicit coverage classification",
            "adult single-snapshot associations do not establish metabolic causality",
        ],
    }

    path = output_root / "vg2_report.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    complete[[*MITO_METRICS, *structural]].corr(
        method="spearman",
        min_periods=100,
    ).to_csv(output_root / "complete_positive_spearman.csv")

    print(f"report={path}")
    print(f"joined_neurons={len(joined)}")
    print(f"complete_positive_rows={len(complete)}")
    print(f"complete_positive_fraction={len(complete)/len(joined):.6f}")


if __name__ == "__main__":
    main()
