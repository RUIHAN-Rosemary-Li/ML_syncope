#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Directly create zoomed DCA plots with legacy syncope scores from prediction files.

This script does NOT require the previous *_dca_net_benefit.csv files.
It directly reads:
  temporal_external_1m_three_youden_thresholds_no_spo2/<group>/external_1m_predictions_three_thresholds.csv

Default groups:
  total, age_plus, age_minus, cv_plus, cv_minus

Default zoom:
  threshold probability 0.01-0.30

Outputs:
  temporal_external_1m_DCA_with_legacy_scores_no_spo2_DIRECT_ZOOMED/<group>/
    <group>_dca_with_legacy_scores_zoomed.png
    <group>_dca_with_legacy_scores_zoomed.pdf
    <group>_dca_net_benefit_zoomed.csv
    <group>_legacy_score_predictions.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


DEFAULT_GROUPS = ["total", "age_plus", "age_minus", "cv_plus", "cv_minus"]


def find_col(df, candidates):
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in lower_map:
            return lower_map[key]
    return None


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def binary_positive(df, candidates):
    col = find_col(df, candidates)
    if col is None:
        return pd.Series(False, index=df.index), None

    x = df[col]
    if pd.api.types.is_numeric_dtype(x):
        return to_num(x).fillna(0) > 0, col

    xs = x.astype(str).str.strip().str.lower()
    pos_values = {
        "1", "1.0", "true", "t", "yes", "y", "positive", "pos",
        "有", "是", "阳性", "异常", "存在",
    }
    return xs.isin(pos_values), col


def numeric_condition(df, candidates, condition_func):
    col = find_col(df, candidates)
    if col is None:
        return pd.Series(False, index=df.index), None

    x = to_num(df[col])
    return condition_func(x).fillna(False), col


def qrs_abnormal(df):
    qrs_col = find_col(df, ["QRSd", "QRS", "QRS_duration", "QRS duration", "QRSWide", "wide_QRS", "宽QRS波"])
    lbbb_pos, lbbb_col = binary_positive(df, ["LBBB", "left bundle branch block", "左束支"])

    qrs_pos = pd.Series(False, index=df.index)
    used = []

    if qrs_col is not None:
        q = to_num(df[qrs_col])
        q_nonmiss = q.dropna()

        if len(q_nonmiss) > 0:
            if q_nonmiss.max() > 20:
                qrs_pos = (q > 130).fillna(False)
            else:
                qrs_pos = (q.fillna(0) > 0)

        used.append(qrs_col)

    if lbbb_col is not None:
        used.append(lbbb_col)

    return (qrs_pos | lbbb_pos).fillna(False), used


def hct_low(df):
    col = find_col(df, ["HCT", "HCT%", "Hct", "hematocrit", "红细胞压积"])
    if col is None:
        return pd.Series(False, index=df.index), None

    x = to_num(df[col])
    x_nonmiss = x.dropna()

    if len(x_nonmiss) == 0:
        return pd.Series(False, index=df.index), col

    if x_nonmiss.median() <= 1.5:
        return (x < 0.30).fillna(False), col

    return (x < 30).fillna(False), col


def hgb_low_rose(df):
    col = find_col(df, ["HGB", "Hgb", "Hb", "hemoglobin", "血红蛋白"])
    if col is None:
        return pd.Series(False, index=df.index), None

    x = to_num(df[col])
    x_nonmiss = x.dropna()

    if len(x_nonmiss) == 0:
        return pd.Series(False, index=df.index), col

    if x_nonmiss.median() <= 25:
        return (x <= 9.0).fillna(False), col

    return (x <= 90).fillna(False), col


def compute_legacy_scores(df):
    df = df.copy()

    ecg_abn, ecg_cols = qrs_abnormal(df)
    hf_pos, hf_col = binary_positive(df, ["HF", "Heart failure", "congestive HF", "CHF", "心衰"])
    arry_pos, arry_col = binary_positive(df, ["Arrythmia", "Arrhythmia", "心律失常"])
    chd_pos, chd_col = binary_positive(df, ["CHD", "heart disease", "Cardiac disease", "冠心病"])

    sbp_low, sbp_col = numeric_condition(df, ["SBP", "Systolic BP", "收缩压"], lambda x: x < 90)
    sbp_extreme, sbp_col2 = numeric_condition(df, ["SBP", "Systolic BP", "收缩压"], lambda x: (x < 90) | (x > 180))

    hct_pos, hct_col = hct_low(df)
    hgb_pos, hgb_col = hgb_low_rose(df)
    bnp_pos, bnp_col = numeric_condition(df, ["BNP", "NTproBNP", "NT-proBNP", "NT_proBNP"], lambda x: x >= 300)
    pulse_pos, pulse_col = numeric_condition(df, ["Pulse", "pulse", "脉搏", "HR", "heart rate"], lambda x: x <= 50)
    tni_pos, tni_col = numeric_condition(df, ["Tni", "TnI", "TNI", "troponin", "肌钙蛋白"], lambda x: x > 0.03)
    qrs_gt130, qrs_dur_col = numeric_condition(df, ["QRSd", "QRS", "QRS_duration", "QRS duration"], lambda x: x > 130)

    df["legacy_SFSR_positive"] = (ecg_abn | hf_pos | hct_pos | sbp_low).astype(int)
    df["legacy_ROSE_positive"] = (bnp_pos | pulse_pos | hgb_pos).astype(int)
    df["legacy_CSRS_score"] = (
        chd_pos.astype(int) * 1
        + sbp_extreme.astype(int) * 2
        + tni_pos.astype(int) * 2
        + qrs_gt130.astype(int) * 1
    )
    df["legacy_CSRS_positive"] = (df["legacy_CSRS_score"] >= 4).astype(int)
    df["legacy_FAINT_positive"] = (bnp_pos | hf_pos | arry_pos | ecg_abn).astype(int)

    missing_rows = [
        {"score": "SFSR", "component": "ECG_abnormal", "used_column": ";".join(map(str, ecg_cols))},
        {"score": "SFSR", "component": "HF", "used_column": "" if hf_col is None else hf_col},
        {"score": "SFSR", "component": "HCT", "used_column": "" if hct_col is None else hct_col},
        {"score": "SFSR", "component": "SBP", "used_column": "" if sbp_col is None else sbp_col},

        {"score": "ROSE", "component": "BNP", "used_column": "" if bnp_col is None else bnp_col},
        {"score": "ROSE", "component": "Pulse", "used_column": "" if pulse_col is None else pulse_col},
        {"score": "ROSE", "component": "HGB", "used_column": "" if hgb_col is None else hgb_col},

        {"score": "CSRS", "component": "CHD", "used_column": "" if chd_col is None else chd_col},
        {"score": "CSRS", "component": "SBP", "used_column": "" if sbp_col2 is None else sbp_col2},
        {"score": "CSRS", "component": "Tni", "used_column": "" if tni_col is None else tni_col},
        {"score": "CSRS", "component": "QRSd_gt130", "used_column": "" if qrs_dur_col is None else qrs_dur_col},

        {"score": "FAINT", "component": "BNP", "used_column": "" if bnp_col is None else bnp_col},
        {"score": "FAINT", "component": "HF", "used_column": "" if hf_col is None else hf_col},
        {"score": "FAINT", "component": "Arrhythmia", "used_column": "" if arry_col is None else arry_col},
        {"score": "FAINT", "component": "ECG_abnormal", "used_column": ";".join(map(str, ecg_cols))},
    ]

    missing_df = pd.DataFrame(missing_rows)
    missing_df["missing_or_unavailable"] = missing_df["used_column"].astype(str).str.len() == 0

    return df, missing_df


def net_benefit_from_prob(y_true, y_prob, thresholds):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)

    out = []
    for pt in thresholds:
        pred = (y_prob >= pt).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        nb = tp / n - fp / n * (pt / (1 - pt))
        out.append(nb)
    return np.asarray(out)


def net_benefit_from_binary_rule(y_true, pred_positive, thresholds):
    y_true = np.asarray(y_true).astype(int)
    pred_positive = np.asarray(pred_positive).astype(int)
    n = len(y_true)

    out = []
    for pt in thresholds:
        tn, fp, fn, tp = confusion_matrix(y_true, pred_positive, labels=[0, 1]).ravel()
        nb = tp / n - fp / n * (pt / (1 - pt))
        out.append(nb)
    return np.asarray(out)


def summarize_binary_score(y_true, pred, score_name, group):
    y_true = np.asarray(y_true).astype(int)
    pred = np.asarray(pred).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)

    return {
        "group": group,
        "score": score_name,
        "positive_n": int(pred.sum()),
        "positive_rate": float(pred.mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "youden_index": float(sens + spec - 1),
    }


def auto_ylim(dca_df):
    curve_cols = [c for c in ["Model", "SFSR", "ROSE", "CSRS", "FAINT", "Treat all", "Treat none"] if c in dca_df.columns]
    vals = []
    for c in curve_cols:
        vals.extend(pd.to_numeric(dca_df[c], errors="coerce").values)
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return -0.05, 0.10

    lo = max(float(np.min(vals)) - 0.02, -0.10)
    hi = max(float(np.max(vals)) + 0.02, 0.05)

    if hi - lo < 0.08:
        mid = (hi + lo) / 2
        lo = mid - 0.04
        hi = mid + 0.04

    return lo, hi


def plot_group(group, pred_file, out_root, xmin, xmax, n_thresholds, ymin=None, ymax=None):
    pred_file = Path(pred_file)
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")

    out_dir = Path(out_root) / group
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_file)

    if "y_true" not in df.columns or "y_prob" not in df.columns:
        raise ValueError(f"{pred_file} must contain y_true and y_prob columns.")

    df, missing_df = compute_legacy_scores(df)

    y_true = pd.to_numeric(df["y_true"], errors="coerce")
    y_prob = pd.to_numeric(df["y_prob"], errors="coerce")
    mask = y_true.notna() & y_prob.notna()

    df = df.loc[mask].copy()
    y_true = df["y_true"].astype(int).values
    y_prob = pd.to_numeric(df["y_prob"], errors="coerce").values

    thresholds = np.linspace(xmin, xmax, n_thresholds)
    prevalence = float(np.mean(y_true))

    dca_df = pd.DataFrame({
        "threshold": thresholds,
        "Model": net_benefit_from_prob(y_true, y_prob, thresholds),
        "Treat all": prevalence - (1 - prevalence) * thresholds / (1 - thresholds),
        "Treat none": np.zeros_like(thresholds),
        "SFSR": net_benefit_from_binary_rule(y_true, df["legacy_SFSR_positive"].values, thresholds),
        "ROSE": net_benefit_from_binary_rule(y_true, df["legacy_ROSE_positive"].values, thresholds),
        "CSRS": net_benefit_from_binary_rule(y_true, df["legacy_CSRS_positive"].values, thresholds),
        "FAINT": net_benefit_from_binary_rule(y_true, df["legacy_FAINT_positive"].values, thresholds),
    })

    summary_rows = [
        summarize_binary_score(y_true, df["legacy_SFSR_positive"].values, "SFSR", group),
        summarize_binary_score(y_true, df["legacy_ROSE_positive"].values, "ROSE", group),
        summarize_binary_score(y_true, df["legacy_CSRS_positive"].values, "CSRS", group),
        summarize_binary_score(y_true, df["legacy_FAINT_positive"].values, "FAINT", group),
    ]

    dca_df.to_csv(out_dir / f"{group}_dca_net_benefit_zoomed.csv", index=False)
    df.to_csv(out_dir / f"{group}_legacy_score_predictions.csv", index=False)
    missing_df.insert(0, "group", group)
    missing_df.to_csv(out_dir / f"{group}_legacy_score_missing_components.csv", index=False)

    ylo, yhi = auto_ylim(dca_df)
    if ymin is not None:
        ylo = ymin
    if ymax is not None:
        yhi = ymax

    plt.figure(figsize=(7.2, 5.4), dpi=300)

    plt.plot(dca_df["threshold"], dca_df["Model"], linewidth=2.4, label="Machine learning model")
    plt.plot(dca_df["threshold"], dca_df["SFSR"], linewidth=1.7, linestyle="--", label="SFSR")
    plt.plot(dca_df["threshold"], dca_df["ROSE"], linewidth=1.7, linestyle="--", label="ROSE")
    plt.plot(dca_df["threshold"], dca_df["CSRS"], linewidth=1.7, linestyle="--", label="CSRS")
    plt.plot(dca_df["threshold"], dca_df["FAINT"], linewidth=1.7, linestyle="--", label="FAINT")
    plt.plot(dca_df["threshold"], dca_df["Treat all"], linewidth=1.4, linestyle=":", label="Treat all")
    plt.plot(dca_df["threshold"], dca_df["Treat none"], linewidth=1.4, linestyle="-.", label="Treat none")

    # mark existing threshold columns, if present
    for c in df.columns:
        if c.startswith("threshold_") and c.endswith("_youden"):
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(vals) > 0:
                x = float(vals.iloc[0])
                if xmin <= x <= xmax:
                    plt.axvline(x, linewidth=0.8, linestyle=":", alpha=0.6)

    plt.axhline(0, linewidth=0.8, linestyle="-", alpha=0.5)
    plt.xlim(xmin, xmax)
    plt.ylim(ylo, yhi)
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title(f"{group} DCA with legacy scores")
    plt.legend(frameon=False, loc="best", fontsize=8)
    plt.tight_layout()

    png = out_dir / f"{group}_dca_with_legacy_scores_zoomed.png"
    pdf = out_dir / f"{group}_dca_with_legacy_scores_zoomed.pdf"
    plt.savefig(png, dpi=300)
    plt.savefig(pdf)
    plt.close()

    return pd.DataFrame(summary_rows), missing_df, png


def main():
    parser = argparse.ArgumentParser(description="Direct zoomed DCA with legacy scores from prediction files.")
    parser.add_argument("--project-dir", default="/Users/yan_1nvincible/Documents/muhong 机器学习")
    parser.add_argument("--pred-root", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--groups", nargs="*", default=DEFAULT_GROUPS)
    parser.add_argument("--xmin", type=float, default=0.01)
    parser.add_argument("--xmax", type=float, default=0.30)
    parser.add_argument("--n-thresholds", type=int, default=100)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)
    args = parser.parse_args()

    project_dir = Path(args.project_dir)

    pred_root = Path(args.pred_root) if args.pred_root else project_dir / "temporal_external_1m_three_youden_thresholds_no_spo2"
    out_root = Path(args.outdir) if args.outdir else project_dir / "temporal_external_1m_DCA_with_legacy_scores_no_spo2_DIRECT_ZOOMED"

    out_root.mkdir(parents=True, exist_ok=True)

    all_summary = []
    all_missing = []
    outputs = []
    errors = []

    for group in args.groups:
        pred_file = pred_root / group / "external_1m_predictions_three_thresholds.csv"
        try:
            print("=" * 80)
            print(f"Processing {group}")
            print(f"Prediction file: {pred_file}")
            summary, missing, png = plot_group(
                group=group,
                pred_file=pred_file,
                out_root=out_root,
                xmin=args.xmin,
                xmax=args.xmax,
                n_thresholds=args.n_thresholds,
                ymin=args.ymin,
                ymax=args.ymax,
            )
            all_summary.append(summary)
            all_missing.append(missing)
            outputs.append({"group": group, "output_png": str(png)})
            print("Saved:", png)
        except Exception as e:
            msg = f"[{group}] FAILED: {e}"
            print(msg)
            errors.append(msg)

    summary_df = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    missing_df = pd.concat(all_missing, ignore_index=True) if all_missing else pd.DataFrame()
    outputs_df = pd.DataFrame(outputs)

    summary_df.to_csv(out_root / "all_groups_legacy_score_summary.csv", index=False)
    missing_df.to_csv(out_root / "all_groups_legacy_score_missing_components.csv", index=False)
    outputs_df.to_csv(out_root / "zoomed_dca_outputs.csv", index=False)

    with pd.ExcelWriter(out_root / "all_groups_DCA_with_legacy_scores_zoomed_summary.xlsx", engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Legacy_score_summary", index=False)
        missing_df.to_excel(writer, sheet_name="Missing_components", index=False)
        outputs_df.to_excel(writer, sheet_name="Outputs", index=False)

    if errors:
        (out_root / "errors.log").write_text("\n".join(errors), encoding="utf-8")

    print("=" * 80)
    print("Done.")
    print("Output directory:", out_root)
    if errors:
        print("Some groups failed. See errors.log")


if __name__ == "__main__":
    main()
