#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Internal validation DCA with conventional syncope risk scores.

Purpose:
- Generate new internal-validation DCA plots adding legacy scores:
  SFSR, ROSE, CSRS, FAINT.
- Works for no-SpO2 trained AutoGluon models.
- Supports 5 groups:
  total, age_plus, age_minus, cv_plus, cv_minus
- Supports 3 outcomes by default:
  1w, 1m, 1y

Key logic:
- Recreate the internal 70/30 holdout split using random_state=42 and stratification.
- Load each trained AutoGluon model.
- Select the best non-WeightedEnsemble model using existing metrics files if available.
- Predict probabilities on the internal validation/test set.
- Reconstruct SFSR, ROSE, CSRS, and FAINT from available variables.
- Plot zoomed DCA by default over threshold probability 0.01-0.30.
- Does not retrain models.

Outputs:
internal_validation_DCA_with_legacy_scores_no_spo2/
  <group>/<outcome>/
    <outcome>_internal_dca_with_legacy_scores_zoomed.png
    <outcome>_internal_dca_with_legacy_scores_zoomed.pdf
    <outcome>_internal_dca_net_benefit_zoomed.csv
    <outcome>_internal_validation_predictions_with_legacy_scores.csv
    <outcome>_legacy_score_summary.csv
    <outcome>_legacy_score_missing_components.csv
  all_groups_all_outcomes_legacy_score_summary.xlsx
"""

import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    f1_score,
)

from autogluon.tabular import TabularPredictor


RANDOM_STATE = 42
TEST_SIZE = 0.30

OUTCOME_MAP = {
    "1w": "1wk",
    "1m": "1mo",
    "1y": "1yr",
}

COAG_COLUMNS = ["PT", "INR", "APTT", "Fbg", "TT"]
DEFAULT_GROUPS = ["total", "age_plus", "age_minus", "cv_plus", "cv_minus"]
DEFAULT_OUTCOMES = ["1w", "1m", "1y"]


def is_weightedensemble(name):
    s = str(name).replace("_", "").replace(" ", "").replace("-", "").lower()
    return "weightedensemble" in s


def first_existing(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    return Path(paths[0])


def normalize_gender_value(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ["1", "1.0", "m", "male", "man", "男", "男性"]:
        return "Male"
    if s in ["0", "0.0", "2", "2.0", "f", "female", "woman", "女", "女性"]:
        return "Female"
    return str(x).strip()


def safe_to_numeric(series):
    if pd.api.types.is_numeric_dtype(series):
        return series
    converted = pd.to_numeric(series, errors="coerce")
    non_missing = series.notna().sum()
    if non_missing == 0:
        return series
    if converted.notna().sum() >= max(1, int(non_missing * 0.8)):
        return converted
    return series


def normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    rename_map = {
        "Sex": "gender",
        "sex": "gender",
        "Gender": "gender",
        "性别": "gender",
        "Age": "age",
        "年龄": "age",

        "血压": "BP",
        "Systolic BP": "SBP",
        "Diastolic BP": "DBP",
        "脉搏": "Pulse",

        "血氧(%)": "Spo",
        "血氧（%）": "Spo",
        "血氧": "Spo",
        "SpO2": "Spo",
        "SPO2": "Spo",
        "SPO₂": "Spo",

        "HTN": "HT",
        "Hypertension": "HT",
        "Old MI": "OMI",
        "Old myocardial infarction": "OMI",
        "Old CVA": "OCI",
        "Old cerebral infarction": "OCI",
        "Arrhythmia": "Arrythmia",
        "心律失常": "Arrythmia",
        "肿瘤史": "Tumor",

        "Hgb": "HGB",
        "Hb": "HGB",
        "HCT%": "HCT",
        "Hct": "HCT",
        "Cr": "Crea",
        "Creatinine": "Crea",
        "CK-MB": "CKMB",
        "CK-MB ng/ml": "CKMB",
        "TnI": "Tni",
        "TNI": "Tni",

        "QRSWide": "QRSd",
        "QRS wide": "QRSd",
        "宽QRS波": "QRSd",
        "电轴异常": "AxisAb",
        "QT延长": "QTProl",
        "左束支": "LBBB",
        "左心肥厚": "LVH",

        "Outcome1w": "1wk",
        "Outcome 1w": "1wk",
        "1周内就诊原因": "1wk",
        "1w": "1wk",

        "Outcome1m": "1mo",
        "Outcome 1m": "1mo",
        "1月内就诊原因": "1mo",
        "1m": "1mo",

        "Outcome1y": "1yr",
        "Outcome 1y": "1yr",
        "1年内就诊原因": "1yr",
        "1y": "1yr",
    }

    return df.rename(columns={c: rename_map.get(c, c) for c in df.columns})


def load_data(path):
    df = pd.read_excel(path)
    df = normalize_columns(df)

    if "gender" in df.columns:
        df["gender"] = df["gender"].apply(normalize_gender_value)

    for c in df.columns:
        if c != "gender":
            df[c] = safe_to_numeric(df[c])

    return df


def preprocess_for_outcome(df_raw, outcome_col, remove_spo2=True):
    df = df_raw.copy()

    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column {outcome_col} not found. Current columns: {list(df.columns)}")

    df = df.dropna(subset=[outcome_col]).copy()

    # Drop other outcome columns to avoid leakage.
    for c in ["1wk", "1mo", "1yr"]:
        if c != outcome_col and c in df.columns:
            df = df.drop(columns=c)

    # Keep the target outcome.
    df[outcome_col] = df[outcome_col].astype(int)

    # Same no-SpO2 setting.
    if remove_spo2 and "Spo" in df.columns:
        df = df.drop(columns=["Spo"])

    # Keep consistent with prior scripts: remove coagulation variables if present.
    drop_cols = [c for c in COAG_COLUMNS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Remove rows with too much missingness.
    df = df.loc[df.isna().mean(axis=1) <= 0.5].copy()
    return df.reset_index(drop=True)


def get_model_features(predictor):
    for attr in ["feature_metadata_in", "feature_metadata"]:
        try:
            feats = list(getattr(predictor, attr).get_features())
            if feats:
                return feats
        except Exception:
            pass

    try:
        feats = list(predictor.features())
        if feats:
            return feats
    except Exception:
        pass

    raise RuntimeError("Cannot determine model features from AutoGluon predictor.")


def complete_missing_features(df, features):
    df = df.copy()
    added = []

    # Some scripts may have created BP from SBP/DBP.
    if "BP" in features and "BP" not in df.columns:
        if "SBP" in df.columns and "DBP" in df.columns:
            df["BP"] = (
                (pd.to_numeric(df["SBP"], errors="coerce") >= 140) |
                (pd.to_numeric(df["DBP"], errors="coerce") >= 90)
            ).astype(float)
            added.append("BP_created_from_SBP_DBP_ge_140_90")
        else:
            df["BP"] = np.nan
            added.append("BP_created_as_missing")

    # no-SpO2 model should not require Spo; this is only safety.
    if "Spo" in features and "Spo" not in df.columns:
        df["Spo"] = 98.0
        added.append("WARNING_Spo_created_as_constant_98_model_still_requires_Spo")

    return df, added


def positive_proba(predictor, X, model):
    p = predictor.predict_proba(X, model=model)

    if isinstance(p, pd.DataFrame):
        if 1 in p.columns:
            return p[1].astype(float).values
        if "1" in p.columns:
            return p["1"].astype(float).values
        return p.iloc[:, -1].astype(float).values

    if isinstance(p, pd.Series):
        return p.astype(float).values

    arr = np.asarray(p)
    if arr.ndim == 2:
        return arr[:, -1].astype(float)
    return arr.astype(float)


def select_non_weighted_model(predictor, X_probe, outcome_dir, outcome_key):
    outcome_dir = Path(outcome_dir)

    candidate_files = [
        outcome_dir / f"{outcome_key}_model_metrics_optimal_threshold.csv",
        outcome_dir / f"{outcome_key}_model_metrics_default_threshold.csv",
        outcome_dir / f"{outcome_key}_model_metrics.csv",
        outcome_dir / f"{outcome_key}_leaderboard_full.csv",
    ]

    for fp in candidate_files:
        if not fp.exists():
            continue

        try:
            df = pd.read_csv(fp)
            if "model" not in df.columns:
                continue

            df = df.loc[~df["model"].astype(str).map(is_weightedensemble)].copy()
            if len(df) == 0:
                continue

            sort_cols = [c for c in ["auc", "auprc", "f1", "score_test", "score_val"] if c in df.columns]
            if sort_cols:
                df = df.sort_values(sort_cols, ascending=False)

            for m in df["model"].astype(str).tolist():
                try:
                    _ = positive_proba(predictor, X_probe.head(min(30, len(X_probe))), m)
                    return m, f"selected_from_{fp.name}"
                except Exception:
                    pass
        except Exception:
            pass

    try:
        lb = predictor.leaderboard(silent=True)
    except TypeError:
        lb = predictor.leaderboard()

    if "model" not in lb.columns:
        raise ValueError("AutoGluon leaderboard has no model column.")

    for m in lb["model"].astype(str).tolist():
        if is_weightedensemble(m):
            continue
        try:
            _ = positive_proba(predictor, X_probe.head(min(30, len(X_probe))), m)
            return m, "selected_from_predictor_leaderboard"
        except Exception:
            pass

    raise RuntimeError("No usable non-WeightedEnsemble model found.")


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

    # SFSR: ECG abnormal, HF, HCT<30%, SBP<90; any positive.
    df["legacy_SFSR_positive"] = (ecg_abn | hf_pos | hct_pos | sbp_low).astype(int)

    # ROSE: BNP>=300, Pulse<=50, HGB<=90 g/L or <=9 g/dL; any positive.
    df["legacy_ROSE_positive"] = (bnp_pos | pulse_pos | hgb_pos).astype(int)

    # CSRS simplified available variables.
    df["legacy_CSRS_score"] = (
        chd_pos.astype(int) * 1
        + sbp_extreme.astype(int) * 2
        + tni_pos.astype(int) * 2
        + qrs_gt130.astype(int) * 1
    )
    df["legacy_CSRS_positive"] = (df["legacy_CSRS_score"] >= 4).astype(int)

    # FAINT: BNP>=300, HF, arrhythmia, abnormal ECG; any positive.
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


def summarize_binary_score(y_true, pred, score_name, group, outcome_key):
    y_true = np.asarray(y_true).astype(int)
    pred = np.asarray(pred).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)

    return {
        "group": group,
        "outcome": outcome_key,
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


def calc_model_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)

    out = {
        "auc": np.nan,
        "auprc": np.nan,
        "brier_score": brier_score_loss(y_true, y_prob),
        "accuracy": accuracy_score(y_true, pred),
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "f1": f1_score(y_true, pred, zero_division=0),
        "youden_index": sens + spec - 1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if len(np.unique(y_true)) == 2:
        out["auc"] = roc_auc_score(y_true, y_prob)
        out["auprc"] = average_precision_score(y_true, y_prob)

    return out


def find_youden_threshold(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    thresholds = np.unique(np.quantile(y_prob, np.linspace(0.01, 0.99, 99)))
    if len(thresholds) == 0:
        thresholds = np.array([0.5])

    best = {
        "threshold": 0.5,
        "youden_index": -np.inf,
        "sensitivity": np.nan,
        "specificity": np.nan,
    }

    for th in thresholds:
        metrics = calc_model_metrics(y_true, y_prob, th)
        if metrics["youden_index"] > best["youden_index"]:
            best = {
                "threshold": float(th),
                "youden_index": float(metrics["youden_index"]),
                "sensitivity": float(metrics["sensitivity"]),
                "specificity": float(metrics["specificity"]),
            }

    return best


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


def plot_dca(group, outcome_key, dca_df, model_threshold, out_dir, xmin, xmax, ymin=None, ymax=None):
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

    if xmin <= model_threshold <= xmax:
        plt.axvline(model_threshold, linewidth=0.8, linestyle=":", alpha=0.6)

    plt.axhline(0, linewidth=0.8, linestyle="-", alpha=0.5)
    plt.xlim(xmin, xmax)
    plt.ylim(ylo, yhi)

    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title(f"{group} {outcome_key} internal validation DCA")
    plt.legend(frameon=False, loc="best", fontsize=8)
    plt.tight_layout()

    out_png = out_dir / f"{outcome_key}_internal_dca_with_legacy_scores_zoomed.png"
    out_pdf = out_dir / f"{outcome_key}_internal_dca_with_legacy_scores_zoomed.pdf"

    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()

    return out_png


def build_jobs(project_dir):
    p = Path(project_dir)

    return {
        "total": {
            "data_file": first_existing([p / "data10_no_spo2.xlsx", p / "data10.xlsx"]),
            "model_root": p / "autogluon_data10_total_top10_shap_youden_no_spo2",
        },
        "age_plus": {
            "data_file": first_existing([p / "data10-old_no_spo2.xlsx", p / "data10-old.xlsx"]),
            "model_root": p / "autogluon_data10_old_young_top10_shap_youden_no_spo2" / "old",
        },
        "age_minus": {
            "data_file": first_existing([p / "data10-young_no_spo2.xlsx", p / "data10-young.xlsx"]),
            "model_root": p / "autogluon_data10_old_young_top10_shap_youden_no_spo2" / "young",
        },
        "cv_plus": {
            "data_file": first_existing([p / "data10-sub4_no_spo2.xlsx", p / "data10-sub4.xlsx"]),
            "model_root": p / "autogluon_data10_sub4_sub4minus_top10_shap_youden_no_spo2" / "sub4",
        },
        "cv_minus": {
            "data_file": first_existing([p / "data10-sub4-_no_spo2.xlsx", p / "data10-sub4-.xlsx"]),
            "model_root": p / "autogluon_data10_sub4_sub4minus_top10_shap_youden_no_spo2" / "sub4_minus",
        },
    }


def process_one(group, outcome_key, job, output_root, xmin, xmax, n_thresholds, ymin=None, ymax=None):
    outcome_col = OUTCOME_MAP[outcome_key]
    data_file = Path(job["data_file"])
    model_root = Path(job["model_root"])
    outcome_dir = model_root / outcome_key
    model_dir = outcome_dir / "ag_model"

    print("=" * 100)
    print(f"Group: {group}; Outcome: {outcome_key}")
    print(f"Data file: {data_file}")
    print(f"Model dir: {model_dir}")

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    raw_df = load_data(data_file)
    df = preprocess_for_outcome(raw_df, outcome_col, remove_spo2=True)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[outcome_col],
    )

    predictor = TabularPredictor.load(str(model_dir))
    features = get_model_features(predictor)

    test_df, added_features = complete_missing_features(test_df, features)
    missing_features = [f for f in features if f not in test_df.columns]
    if missing_features:
        raise ValueError(f"Missing model features in test set: {missing_features}")

    X_test = test_df[features].copy()
    y_true = test_df[outcome_col].astype(int).values

    model_name, model_source = select_non_weighted_model(
        predictor=predictor,
        X_probe=X_test,
        outcome_dir=outcome_dir,
        outcome_key=outcome_key,
    )

    print(f"Selected model: {model_name} ({model_source})")

    y_prob = positive_proba(predictor, X_test, model=model_name)

    youden = find_youden_threshold(y_true, y_prob)
    model_threshold = youden["threshold"]

    test_with_scores, missing_df = compute_legacy_scores(test_df.copy())
    test_with_scores["y_true"] = y_true
    test_with_scores["y_prob"] = y_prob
    test_with_scores["selected_model"] = model_name
    test_with_scores["model_source"] = model_source
    test_with_scores["internal_youden_threshold"] = model_threshold

    thresholds = np.linspace(xmin, xmax, n_thresholds)
    prevalence = float(np.mean(y_true))

    dca_df = pd.DataFrame({
        "threshold": thresholds,
        "Model": net_benefit_from_prob(y_true, y_prob, thresholds),
        "Treat all": prevalence - (1 - prevalence) * thresholds / (1 - thresholds),
        "Treat none": np.zeros_like(thresholds),
        "SFSR": net_benefit_from_binary_rule(y_true, test_with_scores["legacy_SFSR_positive"].values, thresholds),
        "ROSE": net_benefit_from_binary_rule(y_true, test_with_scores["legacy_ROSE_positive"].values, thresholds),
        "CSRS": net_benefit_from_binary_rule(y_true, test_with_scores["legacy_CSRS_positive"].values, thresholds),
        "FAINT": net_benefit_from_binary_rule(y_true, test_with_scores["legacy_FAINT_positive"].values, thresholds),
    })

    model_metrics = calc_model_metrics(y_true, y_prob, model_threshold)
    model_metrics.update({
        "group": group,
        "outcome": outcome_key,
        "selected_model": model_name,
        "model_source": model_source,
        "threshold": model_threshold,
        "threshold_source": "internal_test_youden",
        "n": int(len(y_true)),
        "events": int(np.sum(y_true == 1)),
        "event_rate": prevalence,
        "data_file": str(data_file),
        "model_dir": str(model_dir),
        "added_features": "; ".join(added_features),
    })

    score_summary_rows = [
        summarize_binary_score(y_true, test_with_scores["legacy_SFSR_positive"].values, "SFSR", group, outcome_key),
        summarize_binary_score(y_true, test_with_scores["legacy_ROSE_positive"].values, "ROSE", group, outcome_key),
        summarize_binary_score(y_true, test_with_scores["legacy_CSRS_positive"].values, "CSRS", group, outcome_key),
        summarize_binary_score(y_true, test_with_scores["legacy_FAINT_positive"].values, "FAINT", group, outcome_key),
    ]

    for row in score_summary_rows:
        row["n"] = int(len(y_true))
        row["events"] = int(np.sum(y_true == 1))
        row["event_rate"] = prevalence
        row["data_file"] = str(data_file)

    out_dir = Path(output_root) / group / outcome_key
    out_dir.mkdir(parents=True, exist_ok=True)

    test_with_scores.to_csv(out_dir / f"{outcome_key}_internal_validation_predictions_with_legacy_scores.csv", index=False)
    dca_df.to_csv(out_dir / f"{outcome_key}_internal_dca_net_benefit_zoomed.csv", index=False)
    pd.DataFrame([model_metrics]).to_csv(out_dir / f"{outcome_key}_internal_model_metrics.csv", index=False)
    pd.DataFrame(score_summary_rows).to_csv(out_dir / f"{outcome_key}_legacy_score_summary.csv", index=False)

    missing_df.insert(0, "outcome", outcome_key)
    missing_df.insert(0, "group", group)
    missing_df.to_csv(out_dir / f"{outcome_key}_legacy_score_missing_components.csv", index=False)

    out_png = plot_dca(
        group=group,
        outcome_key=outcome_key,
        dca_df=dca_df,
        model_threshold=model_threshold,
        out_dir=out_dir,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
    )

    print(f"Saved DCA: {out_png}")

    return pd.DataFrame([model_metrics]), pd.DataFrame(score_summary_rows), missing_df, pd.DataFrame({
        "group": [group],
        "outcome": [outcome_key],
        "dca_png": [str(out_png)],
    })


def main():
    parser = argparse.ArgumentParser(description="Internal validation DCA with legacy syncope risk scores.")
    parser.add_argument("--project-dir", default="/Users/yan_1nvincible/Documents/muhong 机器学习")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--groups", nargs="*", default=DEFAULT_GROUPS, help="total age_plus age_minus cv_plus cv_minus")
    parser.add_argument("--outcomes", nargs="*", default=DEFAULT_OUTCOMES, help="1w 1m 1y")
    parser.add_argument("--xmin", type=float, default=0.01)
    parser.add_argument("--xmax", type=float, default=0.30)
    parser.add_argument("--n-thresholds", type=int, default=100)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)
    args = parser.parse_args()

    project_dir = Path(args.project_dir)

    if args.outdir is None:
        output_root = project_dir / "internal_validation_DCA_with_legacy_scores_no_spo2"
    else:
        output_root = Path(args.outdir)

    output_root.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(project_dir)

    all_model_metrics = []
    all_score_summary = []
    all_missing = []
    all_outputs = []
    errors = []

    for group in args.groups:
        if group not in jobs:
            errors.append(f"[{group}] Unknown group.")
            continue

        for outcome_key in args.outcomes:
            if outcome_key not in OUTCOME_MAP:
                errors.append(f"[{group}/{outcome_key}] Unknown outcome.")
                continue

            try:
                model_metrics, score_summary, missing_df, output_df = process_one(
                    group=group,
                    outcome_key=outcome_key,
                    job=jobs[group],
                    output_root=output_root,
                    xmin=args.xmin,
                    xmax=args.xmax,
                    n_thresholds=args.n_thresholds,
                    ymin=args.ymin,
                    ymax=args.ymax,
                )

                all_model_metrics.append(model_metrics)
                all_score_summary.append(score_summary)
                all_missing.append(missing_df)
                all_outputs.append(output_df)

            except Exception as e:
                msg = f"[{group}/{outcome_key}] FAILED: {e}\n{traceback.format_exc()}"
                print(msg)
                errors.append(msg)

    model_metrics_df = pd.concat(all_model_metrics, ignore_index=True) if all_model_metrics else pd.DataFrame()
    score_summary_df = pd.concat(all_score_summary, ignore_index=True) if all_score_summary else pd.DataFrame()
    missing_df = pd.concat(all_missing, ignore_index=True) if all_missing else pd.DataFrame()
    outputs_df = pd.concat(all_outputs, ignore_index=True) if all_outputs else pd.DataFrame()

    model_metrics_df.to_csv(output_root / "all_groups_all_outcomes_internal_model_metrics.csv", index=False)
    score_summary_df.to_csv(output_root / "all_groups_all_outcomes_legacy_score_summary.csv", index=False)
    missing_df.to_csv(output_root / "all_groups_all_outcomes_legacy_score_missing_components.csv", index=False)
    outputs_df.to_csv(output_root / "all_groups_all_outcomes_DCA_outputs.csv", index=False)

    with pd.ExcelWriter(output_root / "all_groups_all_outcomes_internal_DCA_with_legacy_scores_summary.xlsx", engine="openpyxl") as writer:
        model_metrics_df.to_excel(writer, sheet_name="ML_model_metrics", index=False)
        score_summary_df.to_excel(writer, sheet_name="Legacy_score_summary", index=False)
        missing_df.to_excel(writer, sheet_name="Missing_components", index=False)
        outputs_df.to_excel(writer, sheet_name="DCA_outputs", index=False)

    if errors:
        (output_root / "errors.log").write_text("\n\n".join(errors), encoding="utf-8")

    print("=" * 100)
    print("Internal validation DCA with legacy scores finished.")
    print("Output directory:", output_root)
    print("Summary Excel:", output_root / "all_groups_all_outcomes_internal_DCA_with_legacy_scores_summary.xlsx")
    if errors:
        print("Some groups/outcomes failed. See errors.log")


if __name__ == "__main__":
    main()
