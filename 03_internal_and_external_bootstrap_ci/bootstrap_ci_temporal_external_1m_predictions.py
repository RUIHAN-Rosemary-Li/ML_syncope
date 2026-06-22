#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bootstrap 95% CI for temporal 1-month external validation results.

This script does NOT reload or retrain models. It uses the external validation
prediction files that were already generated yesterday:

  external_validation_1m_predictions.csv
  external_validation_1m_metrics.csv

For each group:
1) Read predicted probability and true 1m outcome.
2) Read the primary internal-threshold row from external_validation_1m_metrics.csv.
3) Fix that threshold during bootstrap.
4) Bootstrap validation samples 1000 times.
5) Report 95% CI for AUC, AUPRC, Brier score, accuracy, sensitivity,
   specificity, PPV, NPV, F1, and Youden index.

Default groups:
- overall external validation
- age subgroup external validation: young / old
- sub4 subgroup external validation: sub4 / sub4_minus
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    f1_score,
    confusion_matrix,
)


METRICS = [
    "auc", "auprc", "brier_score",
    "accuracy", "sensitivity", "specificity",
    "ppv", "npv", "f1", "youden",
]


def find_col(df, candidates, required=True, desc="column"):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Cannot find {desc}. Tried: {candidates}. Current columns: {list(df.columns)}")
    return None


def calc_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)

    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    else:
        auc = np.nan
        auprc = np.nan

    return {
        "auc": float(auc) if pd.notna(auc) else np.nan,
        "auprc": float(auprc) if pd.notna(auprc) else np.nan,
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "youden": float(sensitivity + specificity - 1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def bootstrap_ci(y_true, y_prob, threshold, n_bootstrap=1000, seed=2026):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)

    point = calc_metrics(y_true, y_prob, threshold)

    boot_rows = []
    skipped = 0

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        p_b = y_prob[idx]

        if len(np.unique(y_b)) < 2:
            skipped += 1
            continue

        row = calc_metrics(y_b, p_b, threshold)
        row["bootstrap_id"] = i + 1
        boot_rows.append(row)

    boot_df = pd.DataFrame(boot_rows)

    summary_rows = []
    for metric in METRICS:
        vals = pd.to_numeric(boot_df[metric], errors="coerce").dropna()
        pe = point[metric]

        if len(vals) == 0:
            lo, hi, bmean, bsd = np.nan, np.nan, np.nan, np.nan
            text = ""
        else:
            lo = float(np.percentile(vals, 2.5))
            hi = float(np.percentile(vals, 97.5))
            bmean = float(vals.mean())
            bsd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            text = f"{pe:.3f} ({lo:.3f}-{hi:.3f})"

        summary_rows.append({
            "metric": metric,
            "point_estimate": pe,
            "ci_lower_2.5": lo,
            "ci_upper_97.5": hi,
            "point_95ci": text,
            "bootstrap_mean": bmean,
            "bootstrap_sd": bsd,
            "n_bootstrap_valid": int(len(vals)),
            "n_bootstrap_skipped": int(skipped),
        })

    return pd.DataFrame([point]), boot_df, pd.DataFrame(summary_rows)


def read_threshold(metrics_path, manual_threshold=None):
    if manual_threshold is not None:
        return float(manual_threshold), "manual_argument", "", ""

    df = pd.read_csv(metrics_path)

    row = df.copy()
    if "validation_type" in df.columns:
        primary = df[df["validation_type"].astype(str).str.contains("primary_internal_threshold", na=False)]
        if len(primary) > 0:
            row = primary

    if len(row) == 0:
        raise ValueError(f"No rows found in metrics file: {metrics_path}")

    row0 = row.iloc[0]

    for c in ["threshold", "optimal_threshold", "reported_threshold"]:
        if c in row.columns:
            return (
                float(row0[c]),
                f"from_{Path(metrics_path).name}:{c}",
                str(row0.get("model_name", "")),
                str(row0.get("validation_type", "")),
            )

    raise ValueError(f"No threshold column found in {metrics_path}. Columns: {list(df.columns)}")


def paper_row_from_summary(summary_df, meta):
    row = dict(meta)
    label_map = {
        "auc": "AUC",
        "auprc": "AUPRC",
        "brier_score": "Brier score",
        "accuracy": "Accuracy",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "ppv": "PPV",
        "npv": "NPV",
        "f1": "F1 score",
        "youden": "Youden index",
    }

    for metric, label in label_map.items():
        sub = summary_df[summary_df["metric"] == metric]
        row[label] = sub["point_95ci"].iloc[0] if len(sub) else ""

    return row


def process_one_group(group_name, result_dir, out_root, n_bootstrap=1000, seed=2026, manual_threshold=None):
    result_dir = Path(result_dir)

    pred_path = result_dir / "external_validation_1m_predictions.csv"
    metrics_path = result_dir / "external_validation_1m_metrics.csv"

    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    pred_df = pd.read_csv(pred_path)

    y_col = find_col(
        pred_df,
        ["1mo", "Outcome1m", "Outcome 1m", "y_true", "label", "target"],
        required=True,
        desc="true 1m outcome column",
    )

    prob_col = find_col(
        pred_df,
        ["predicted_probability_1mo", "predicted_probability", "y_prob", "probability", "pred_prob"],
        required=True,
        desc="predicted probability column",
    )

    y_true = pd.to_numeric(pred_df[y_col], errors="coerce")
    y_prob = pd.to_numeric(pred_df[prob_col], errors="coerce")

    valid = y_true.notna() & y_prob.notna()
    y_true = y_true[valid].astype(int).values
    y_prob = y_prob[valid].astype(float).values

    threshold, threshold_source, model_name, validation_type = read_threshold(metrics_path, manual_threshold=manual_threshold)

    point_df, boot_df, summary_df = bootstrap_ci(
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    meta = {
        "Group": group_name,
        "result_dir": str(result_dir),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "model_name": model_name,
        "validation_type": validation_type,
        "n": int(len(y_true)),
        "event_n": int(np.sum(y_true == 1)),
        "event_rate": float(np.mean(y_true)),
        "n_bootstrap_requested": int(n_bootstrap),
        "bootstrap_seed": int(seed),
    }

    for k, v in meta.items():
        summary_df[k] = v
        point_df[k] = v
        boot_df[k] = v

    group_out = out_root / group_name
    group_out.mkdir(parents=True, exist_ok=True)

    pred_out = pred_df.loc[valid].copy()
    pred_out["bootstrap_y_true"] = y_true
    pred_out["bootstrap_y_prob"] = y_prob
    pred_out["threshold_used"] = threshold
    pred_out["bootstrap_predicted_label"] = (y_prob >= threshold).astype(int)

    pred_out.to_csv(group_out / "external_1m_predictions_for_bootstrap.csv", index=False)
    point_df.to_csv(group_out / "external_1m_point_metrics.csv", index=False)
    boot_df.to_csv(group_out / "external_1m_bootstrap_distribution.csv", index=False)
    summary_df.to_csv(group_out / "external_1m_bootstrap_95ci_summary.csv", index=False)

    return summary_df, boot_df, point_df, paper_row_from_summary(summary_df, meta)


def default_jobs(project_dir):
    p = Path(project_dir)

    jobs = [
        {
            "Group": "overall",
            "result_dir": p / "autogluon_data10_total_top10_shap_youden" / "external_validation_1m",
        },
        {
            "Group": "young",
            "result_dir": p / "autogluon_data10_old_young_top10_shap_youden" / "external_validation_age_subgroups_1m" / "young",
        },
        {
            "Group": "old",
            "result_dir": p / "autogluon_data10_old_young_top10_shap_youden" / "external_validation_age_subgroups_1m" / "old",
        },
        {
            "Group": "sub4",
            "result_dir": p / "autogluon_data10_sub4_sub4minus_top10_shap_youden" / "external_validation_sub4_subgroups_1m" / "sub4",
        },
        {
            "Group": "sub4_minus",
            "result_dir": p / "autogluon_data10_sub4_sub4minus_top10_shap_youden" / "external_validation_sub4_subgroups_1m" / "sub4_minus",
        },
    ]

    return jobs


def main():
    parser = argparse.ArgumentParser(description="Bootstrap CI for temporal 1m external validation predictions.")
    parser.add_argument("--project-dir", type=str, default="/Users/yan_1nvincible/Documents/muhong 机器学习")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--only", nargs="*", default=None, help="Optional groups: overall young old sub4 sub4_minus")
    parser.add_argument("--single-result-dir", type=str, default=None, help="Optional: process only one external validation result directory.")
    parser.add_argument("--single-group-name", type=str, default="single")
    parser.add_argument("--manual-threshold", type=float, default=None, help="Optional: override threshold for all groups.")
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    if args.outdir is None:
        out_root = project_dir / "data10_temporal_external_1m_bootstrap_95ci"
    else:
        out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.single_result_dir:
        jobs = [{"Group": args.single_group_name, "result_dir": Path(args.single_result_dir)}]
    else:
        jobs = default_jobs(project_dir)

    if args.only:
        keep = set(args.only)
        jobs = [j for j in jobs if j["Group"] in keep]

    all_summary = []
    all_boot = []
    all_point = []
    paper_rows = []
    errors = []

    for j in jobs:
        group = j["Group"]
        result_dir = Path(j["result_dir"])

        if not result_dir.exists():
            msg = f"[{group}] missing result directory, skipped: {result_dir}"
            print(msg)
            errors.append(msg)
            continue

        try:
            print("=" * 100)
            print("Processing:", group)
            print("Result dir:", result_dir)

            summary_df, boot_df, point_df, paper_row = process_one_group(
                group_name=group,
                result_dir=result_dir,
                out_root=out_root,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed + len(all_summary) * 1000,
                manual_threshold=args.manual_threshold,
            )

            all_summary.append(summary_df)
            all_boot.append(boot_df)
            all_point.append(point_df)
            paper_rows.append(paper_row)

        except Exception as e:
            import traceback
            err = f"[{group}] FAILED: {e}\n{traceback.format_exc()}"
            print(err)
            errors.append(err)

    long_df = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    boot_all_df = pd.concat(all_boot, ignore_index=True) if all_boot else pd.DataFrame()
    point_all_df = pd.concat(all_point, ignore_index=True) if all_point else pd.DataFrame()
    paper_df = pd.DataFrame(paper_rows)

    long_path = out_root / "temporal_external_1m_bootstrap_95ci_long.csv"
    boot_path = out_root / "temporal_external_1m_bootstrap_distribution.csv"
    point_path = out_root / "temporal_external_1m_point_metrics.csv"
    paper_path = out_root / "temporal_external_1m_bootstrap_95ci_paper_table.csv"
    xlsx_path = out_root / "temporal_external_1m_bootstrap_95ci_summary.xlsx"

    long_df.to_csv(long_path, index=False)
    boot_all_df.to_csv(boot_path, index=False)
    point_all_df.to_csv(point_path, index=False)
    paper_df.to_csv(paper_path, index=False)

    if errors:
        (out_root / "temporal_external_1m_bootstrap_errors.log").write_text("\n\n".join(errors), encoding="utf-8")

    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            paper_df.to_excel(writer, sheet_name="Paper_table", index=False)
            long_df.to_excel(writer, sheet_name="Long_summary", index=False)
            point_all_df.to_excel(writer, sheet_name="Point_metrics", index=False)
            boot_all_df.to_excel(writer, sheet_name="Bootstrap_distribution", index=False)
            if errors:
                pd.DataFrame({"error": errors}).to_excel(writer, sheet_name="Errors", index=False)
    except Exception as e:
        print("Excel export failed:", e)

    print("=" * 100)
    print("Temporal external 1m bootstrap 95% CI completed.")
    print("Output directory:", out_root)
    print("Key outputs:")
    print(" -", paper_path)
    print(" -", xlsx_path)


if __name__ == "__main__":
    main()
