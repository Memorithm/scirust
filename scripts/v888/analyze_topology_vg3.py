#!/usr/bin/env python3
"""VG-3A directed-topology controls for BANC v888 adult growth proxies.

This stage tests whether lineage-level adult morphology is associated with
directed graph structure beyond lineage neuron count and coarse CNS region.
It is exploratory and does not infer developmental causality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

TARGET_METRICS = (
    "l2_cable_length_um",
    "volume_nm3",
    "branchpoints",
    "endpoints",
    "axon_length",
    "dend_length",
)

FOLDS = 5
PERMUTATIONS = 200
SEED = 888003


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "datasets" / "banc_v888",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def clean_label(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    invalid = values.isna() | (values == "") | (values.str.lower() == "nan")
    return values.mask(invalid)


def stable_fold(label: str, folds: int = FOLDS) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % folds


def design_matrix(
    frame: pd.DataFrame,
    *,
    augmented: bool,
    regions: list[str],
) -> np.ndarray:
    n = len(frame)
    columns: list[np.ndarray] = [np.ones(n, dtype=float)]
    columns.append(np.log(frame["n_neurons"].to_numpy(dtype=float)))

    region_values = frame["region_mode"].astype(str)
    for region in regions[1:]:
        columns.append((region_values == region).to_numpy(dtype=float))

    if augmented:
        columns.append(np.log1p(frame["partner_degree_sum"].to_numpy(dtype=float)))
        columns.append(frame["reciprocal_out_fraction"].to_numpy(dtype=float))
        columns.append(frame["within_lineage_output_fraction"].to_numpy(dtype=float))

    return np.column_stack(columns)


def fit_ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta


def cross_validated_model(
    frame: pd.DataFrame,
    target: str,
    *,
    augmented: bool,
    fold_col: str = "fold",
) -> dict[str, float | int | None]:
    target_values = pd.to_numeric(frame[f"{target}__sum"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(target_values) & (target_values > 0.0)
    valid &= frame["n_neurons"].to_numpy(dtype=float) > 0.0
    work = frame.loc[valid].copy()
    y = np.log(target_values[valid])

    if len(work) < 20:
        return {"n": int(len(work)), "cv_rmse": None, "cv_r2": None}

    regions = sorted(str(value) for value in work["region_mode"].dropna().unique())
    if not regions:
        regions = ["unknown"]
        work["region_mode"] = "unknown"

    predictions = np.full(len(work), np.nan, dtype=float)
    folds = work[fold_col].to_numpy(dtype=int)

    for fold in sorted(set(int(value) for value in folds)):
        train = folds != fold
        test = folds == fold
        if int(train.sum()) < 10 or int(test.sum()) == 0:
            continue
        x_train = design_matrix(work.loc[train], augmented=augmented, regions=regions)
        x_test = design_matrix(work.loc[test], augmented=augmented, regions=regions)
        beta = fit_ols(x_train, y[train])
        predictions[test] = x_test @ beta

    ok = np.isfinite(predictions)
    if int(ok.sum()) < max(10, len(work) // 2):
        return {"n": int(ok.sum()), "cv_rmse": None, "cv_r2": None}

    error = y[ok] - predictions[ok]
    ss_res = float(np.sum(error * error))
    centered = y[ok] - float(np.mean(y[ok]))
    ss_tot = float(np.sum(centered * centered))
    return {
        "n": int(ok.sum()),
        "cv_rmse": float(np.sqrt(np.mean(error * error))),
        "cv_r2": None if ss_tot <= 0.0 else float(1.0 - ss_res / ss_tot),
    }


def size_bins(frame: pd.DataFrame) -> pd.Series:
    ranks = frame["n_neurons"].rank(method="first")
    bins = pd.qcut(ranks, q=min(5, len(frame)), labels=False, duplicates="drop")
    return bins.astype("Int64")


def permute_topology_within_strata(
    frame: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    shuffled = frame.copy()
    topology_columns = [
        "partner_degree_sum",
        "reciprocal_out_fraction",
        "within_lineage_output_fraction",
    ]
    strata = shuffled["region_mode"].astype(str) + "|" + shuffled["size_bin"].astype(str)
    for _, indices in shuffled.groupby(strata, sort=True).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        if positions.size < 2:
            continue
        order = rng.permutation(len(positions))
        for column in topology_columns:
            values = shuffled.loc[positions, column].to_numpy(copy=True)
            shuffled.loc[positions, column] = values[order]
    return shuffled


def matched_permutation_control(
    frame: pd.DataFrame,
    target: str,
    observed: dict[str, float | int | None],
) -> dict[str, object]:
    observed_rmse = observed.get("cv_rmse")
    if observed_rmse is None:
        return {
            "permutations": 0,
            "observed_cv_rmse": None,
            "null_median_cv_rmse": None,
            "fraction_null_as_good_or_better": None,
        }

    rng = np.random.default_rng(SEED)
    rmses: list[float] = []
    for _ in range(PERMUTATIONS):
        shuffled = permute_topology_within_strata(frame, rng)
        result = cross_validated_model(shuffled, target, augmented=True)
        value = result.get("cv_rmse")
        if value is not None and math.isfinite(float(value)):
            rmses.append(float(value))

    if not rmses:
        return {
            "permutations": 0,
            "observed_cv_rmse": float(observed_rmse),
            "null_median_cv_rmse": None,
            "fraction_null_as_good_or_better": None,
        }

    array = np.asarray(rmses, dtype=float)
    return {
        "permutations": int(len(array)),
        "observed_cv_rmse": float(observed_rmse),
        "null_median_cv_rmse": float(np.median(array)),
        "null_q05_cv_rmse": float(np.quantile(array, 0.05)),
        "null_q95_cv_rmse": float(np.quantile(array, 0.95)),
        "fraction_null_as_good_or_better": float(np.mean(array <= float(observed_rmse))),
    }


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else data_root / "analysis" / "brain_growth_vg3"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    meta_path = data_root / "banc_888_meta.feather"
    metrics_path = data_root / "banc_888_metrics.feather"
    edge_path = data_root / "banc_888_edgelist_simple_v3.feather"
    for path in (meta_path, metrics_path, edge_path):
        if not path.exists():
            raise FileNotFoundError(path)

    meta = feather.read_feather(
        meta_path,
        columns=["banc_888_id", "hemilineage", "region"],
    )
    metric_columns = ["banc_888_id", *TARGET_METRICS]
    metrics = feather.read_feather(metrics_path, columns=metric_columns)
    edges = feather.read_feather(
        edge_path,
        columns=["pre", "post", "count"],
    )

    if edges.duplicated(["pre", "post"]).any():
        raise ValueError("v3 simple edgelist is expected to contain unique directed pairs")

    edges["count"] = pd.to_numeric(edges["count"], errors="raise")
    if (edges["count"] <= 0).any():
        raise ValueError("edge counts must be strictly positive")

    pair_index = pd.MultiIndex.from_arrays(
        [edges["pre"].to_numpy(), edges["post"].to_numpy()]
    )
    reverse_index = pd.MultiIndex.from_arrays(
        [edges["post"].to_numpy(), edges["pre"].to_numpy()]
    )
    reciprocal = pair_index.isin(reverse_index)
    self_loop = edges["pre"].to_numpy() == edges["post"].to_numpy()

    out_degree = edges.groupby("pre", sort=False).size().rename("out_partner_degree")
    in_degree = edges.groupby("post", sort=False).size().rename("in_partner_degree")
    out_weight = edges.groupby("pre", sort=False)["count"].sum().rename("out_synapses_v3")
    in_weight = edges.groupby("post", sort=False)["count"].sum().rename("in_synapses_v3")

    reciprocal_edges = edges.loc[reciprocal, ["pre", "count"]].copy()
    reciprocal_degree = reciprocal_edges.groupby("pre", sort=False).size().rename(
        "reciprocal_out_degree"
    )
    reciprocal_weight = reciprocal_edges.groupby("pre", sort=False)["count"].sum().rename(
        "reciprocal_out_synapses"
    )

    node_ids = pd.Index(
        pd.unique(pd.concat([edges["pre"], edges["post"]], ignore_index=True)),
        name="banc_888_id",
    )
    node = pd.DataFrame(index=node_ids)
    for series in (
        out_degree,
        in_degree,
        out_weight,
        in_weight,
        reciprocal_degree,
        reciprocal_weight,
    ):
        node = node.join(series, how="left")
    node = node.fillna(0.0).reset_index()
    node["reciprocal_out_fraction"] = np.where(
        node["out_partner_degree"] > 0,
        node["reciprocal_out_degree"] / node["out_partner_degree"],
        0.0,
    )

    meta["hemilineage"] = clean_label(meta["hemilineage"])
    meta["region"] = clean_label(meta["region"])
    annotation = meta.drop_duplicates("banc_888_id").set_index("banc_888_id")
    lineage_map = annotation["hemilineage"]
    region_map = annotation["region"]

    edges["pre_lineage"] = edges["pre"].map(lineage_map)
    edges["post_lineage"] = edges["post"].map(lineage_map)
    edges["pre_region"] = edges["pre"].map(region_map)
    edges["post_region"] = edges["post"].map(region_map)

    both_lineage = edges["pre_lineage"].notna() & edges["post_lineage"].notna()
    lineage_edges = edges.loc[both_lineage, [
        "pre_lineage",
        "post_lineage",
        "count",
    ]].copy()
    lineage_edges["within_lineage"] = (
        lineage_edges["pre_lineage"] == lineage_edges["post_lineage"]
    )

    output_by_lineage = lineage_edges.groupby("pre_lineage", sort=False)["count"].sum()
    within_output = (
        lineage_edges.loc[lineage_edges["within_lineage"]]
        .groupby("pre_lineage", sort=False)["count"]
        .sum()
    )
    unique_output_lineages = (
        lineage_edges.groupby("pre_lineage", sort=False)["post_lineage"]
        .nunique()
        .rename("unique_output_lineages")
    )

    node = node.merge(
        meta[["banc_888_id", "hemilineage", "region"]],
        on="banc_888_id",
        how="left",
        validate="one_to_one",
    )

    annotated = node[node["hemilineage"].notna()].copy()
    topology_agg = annotated.groupby("hemilineage", sort=True).agg(
        node_rows=("banc_888_id", "size"),
        out_partner_degree_sum=("out_partner_degree", "sum"),
        in_partner_degree_sum=("in_partner_degree", "sum"),
        out_synapses_v3_sum=("out_synapses_v3", "sum"),
        in_synapses_v3_sum=("in_synapses_v3", "sum"),
        reciprocal_out_degree_sum=("reciprocal_out_degree", "sum"),
        reciprocal_out_synapses_sum=("reciprocal_out_synapses", "sum"),
    )

    topology_agg["partner_degree_sum"] = (
        topology_agg["out_partner_degree_sum"] + topology_agg["in_partner_degree_sum"]
    )
    topology_agg["reciprocal_out_fraction"] = np.where(
        topology_agg["out_partner_degree_sum"] > 0,
        topology_agg["reciprocal_out_degree_sum"] / topology_agg["out_partner_degree_sum"],
        0.0,
    )
    topology_agg["unique_output_lineages"] = unique_output_lineages.reindex(
        topology_agg.index
    ).fillna(0).astype(int)
    topology_agg["within_lineage_output_fraction"] = np.where(
        output_by_lineage.reindex(topology_agg.index).fillna(0).to_numpy(dtype=float) > 0,
        within_output.reindex(topology_agg.index).fillna(0).to_numpy(dtype=float)
        / output_by_lineage.reindex(topology_agg.index).fillna(0).to_numpy(dtype=float),
        0.0,
    )

    lineage_neurons = meta[meta["hemilineage"].notna()].merge(
        metrics,
        on="banc_888_id",
        how="left",
        validate="one_to_one",
    )
    aggregation = {
        metric: "sum" for metric in TARGET_METRICS if metric in lineage_neurons.columns
    }
    morphology = lineage_neurons.groupby("hemilineage", sort=True).agg(aggregation)
    morphology.columns = [f"{column}__sum" for column in morphology.columns]
    morphology.insert(
        0,
        "n_neurons",
        lineage_neurons.groupby("hemilineage", sort=True).size().reindex(morphology.index),
    )

    region_counts = (
        lineage_neurons.groupby(["hemilineage", "region"], dropna=True)
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["hemilineage", "n", "region"], ascending=[True, False, True])
    )
    region_mode = (
        region_counts.drop_duplicates("hemilineage")
        .set_index("hemilineage")["region"]
        .rename("region_mode")
    )

    lineage = morphology.join(topology_agg, how="left").join(region_mode, how="left")
    lineage = lineage.fillna(
        {
            "node_rows": 0,
            "out_partner_degree_sum": 0,
            "in_partner_degree_sum": 0,
            "out_synapses_v3_sum": 0,
            "in_synapses_v3_sum": 0,
            "reciprocal_out_degree_sum": 0,
            "reciprocal_out_synapses_sum": 0,
            "partner_degree_sum": 0,
            "unique_output_lineages": 0,
            "within_lineage_output_fraction": 0.0,
            "reciprocal_out_fraction": 0.0,
            "region_mode": "unknown",
        }
    ).reset_index()

    lineage["fold"] = lineage["hemilineage"].map(stable_fold).astype(int)
    lineage["size_bin"] = size_bins(lineage)

    model_results: dict[str, object] = {}
    for target in TARGET_METRICS:
        if f"{target}__sum" not in lineage.columns:
            continue
        baseline = cross_validated_model(lineage, target, augmented=False)
        augmented = cross_validated_model(lineage, target, augmented=True)
        delta_rmse = None
        if baseline.get("cv_rmse") is not None and augmented.get("cv_rmse") is not None:
            delta_rmse = float(baseline["cv_rmse"]) - float(augmented["cv_rmse"])
        model_results[target] = {
            "baseline_size_region": baseline,
            "augmented_topology": augmented,
            "cv_rmse_improvement": delta_rmse,
            "matched_permutation": matched_permutation_control(
                lineage,
                target,
                augmented,
            ),
        }

    graph_summary = {
        "directed_edge_pairs": int(len(edges)),
        "weighted_synapses_v3": int(edges["count"].sum()),
        "self_loop_edge_pairs": int(self_loop.sum()),
        "reciprocal_edge_pairs": int(reciprocal.sum()),
        "reciprocal_edge_pair_fraction": float(reciprocal.mean()),
        "reciprocal_weight_fraction": float(
            edges.loc[reciprocal, "count"].sum() / edges["count"].sum()
        ),
        "lineage_annotated_edge_pairs": int(len(lineage_edges)),
        "lineage_annotated_weight": int(lineage_edges["count"].sum()),
        "within_lineage_edge_pairs": int(lineage_edges["within_lineage"].sum()),
        "within_lineage_weight": int(
            lineage_edges.loc[lineage_edges["within_lineage"], "count"].sum()
        ),
        "within_lineage_edge_pair_fraction": float(lineage_edges["within_lineage"].mean()),
        "within_lineage_weight_fraction": float(
            lineage_edges.loc[lineage_edges["within_lineage"], "count"].sum()
            / lineage_edges["count"].sum()
        ),
    }

    lineage.to_csv(output_root / "lineage_topology_metrics.csv", index=False)

    report = {
        "dataset": "BANC v888",
        "scope": "adult_directed_topology_association_not_developmental_causality",
        "folds": FOLDS,
        "matched_permutations": PERMUTATIONS,
        "graph": graph_summary,
        "lineage_groups": int(len(lineage)),
        "model_results": model_results,
        "warnings": [
            "v3 topology is exploratory future-work connectivity, not the v2 paper graph",
            "cross-validation is exploratory model comparison, not a confirmatory holdout",
            "topology association does not establish a developmental growth mechanism",
            "matched permutation preserves coarse region and lineage-size strata only",
        ],
    }
    path = output_root / "vg3_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"report={path}")
    print(f"directed_edge_pairs={len(edges)}")
    print(f"reciprocal_edge_pair_fraction={reciprocal.mean():.6f}")
    print(f"within_lineage_weight_fraction={graph_summary['within_lineage_weight_fraction']:.6f}")


if __name__ == "__main__":
    main()
