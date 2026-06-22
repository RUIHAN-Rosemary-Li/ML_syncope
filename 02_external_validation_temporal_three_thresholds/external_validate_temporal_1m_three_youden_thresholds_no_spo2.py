#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# External validation for no-SpO2 1-month AutoGluon models.
# External files:
#   data_temporal - total.xlsx
#   data_temporal - Age+.xlsx
#   data_temporal - Age-.xlsx
#   data_temporal - CV+.xlsx
#   data_temporal - CV-.xlsx
# For each group, output metrics at 1.0x, 1.25x, and 1.5x of the internal Youden threshold.
# WeightedEnsemble models are excluded.

import argparse
import os
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve

from autogluon.tabular import TabularPredictor


RANDOM_STATE = 42
TEST_SIZE = 0.30
OUTCOME_KEY = "1m"
OUTCOME_COL = "1mo"
THRESHOLD_MULTIPLIERS = [1.0, 1.25, 1.5]
COAG_COLUMNS = ["PT", "INR", "APTT", "Fbg", "TT"]


def is_weightedensemble(name):
    s = str(name).replace("_", "").replace(" ", "").replace("-", "").lower()
    return "weightedensemble" in s


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


def preprocess_for_1m(df_raw, remove_spo2=True):
    df = df_raw.copy()

    if OUTCOME_COL not in df.columns:
        raise ValueError(f"Outcome column {OUTCOME_COL} not found. Current columns: {list(df.columns)}")

    df = df.dropna(subset=[OUTCOME_COL]).copy()

    for c in ["1wk", "1yr"]:
        if c in df.columns:
            df = df.drop(columns=c)

    drop_cols = [c for c in COAG_COLUMNS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if remove_spo2 and "Spo" in df.columns:
        df = df.drop(columns=["Spo"])

    df = df.loc[df.isna().mean(axis=1) <= 0.5].copy()
    df[OUTCOME_COL] = df[OUTCOME_COL].astype(int)
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


def select_non_weighted_model(predictor, X_probe, outcome_dir):
    outcome_dir = Path(outcome_dir)
    candidate_files = [
        outcome_dir / f"{OUTCOME_KEY}_model_metrics_optimal_threshold.csv",
        outcome_dir / f"{OUTCOME_KEY}_model_metrics_default_threshold.csv",
        outcome_dir / f"{OUTCOME_KEY}_model_metrics.csv",
        outcome_dir / f"{OUTCOME_KEY}_leaderboard_full.csv",
    ]

    for fp in candidate_files:
        if fp.exists():
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


def find_youden_threshold(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    thresholds = np.unique(np.quantile(y_prob, np.linspace(0.01, 0.99, 99)))
    if len(thresholds) == 0:
        thresholds = np.array([0.5])

    best = {
        "threshold": 0.5,
        "youden": -np.inf,
        "sensitivity": np.nan,
        "specificity": np.nan,
    }

    for th in thresholds:
        pred = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        youden = sens + spec - 1

        if youden > best["youden"]:
            best = {
                "threshold": float(th),
                "youden": float(youden),
                "sensitivity": float(sens),
                "specificity": float(spec),
            }

    return best


def calc_metrics(y_true, y_prob, threshold):
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


def bootstrap_ci(y_true, y_prob, threshold, n_bootstrap=1000, seed=2026):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)

    point = calc_metrics(y_true, y_prob, threshold)

    rows = []
    skipped = 0
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yy = y_true[idx]
        pp = y_prob[idx]

        if len(np.unique(yy)) < 2:
            skipped += 1
            continue

        row = calc_metrics(yy, pp, threshold)
        row["bootstrap_id"] = i + 1
        rows.append(row)

    boot_df = pd.DataFrame(rows)

    metric_names = [
        "auc", "auprc", "brier_score",
        "accuracy", "sensitivity", "specificity",
        "ppv", "npv", "f1", "youden_index",
    ]

    summary_rows = []
    for m in metric_names:
        vals = pd.to_numeric(boot_df[m], errors="coerce").dropna() if m in boot_df.columns else pd.Series(dtype=float)
        pe = point[m]

        if len(vals) == 0 or pd.isna(pe):
            lo, hi, text = np.nan, np.nan, ""
        else:
            lo = float(np.percentile(vals, 2.5))
            hi = float(np.percentile(vals, 97.5))
            text = f"{pe:.3f} ({lo:.3f}-{hi:.3f})"

        summary_rows.append({
            "metric": m,
            "point_estimate": pe,
            "ci_lower_2.5": lo,
            "ci_upper_97.5": hi,
            "point_95ci": text,
            "bootstrap_mean": float(vals.mean()) if len(vals) else np.nan,
            "bootstrap_sd": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
            "n_bootstrap_valid": int(len(vals)),
            "n_bootstrap_skipped": int(skipped),
        })

    return pd.DataFrame([point]), boot_df, pd.DataFrame(summary_rows)


def net_benefit(y_true, y_prob, thresholds):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)

    out = []
    for pt in thresholds:
        pred = (y_prob >= pt).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        out.append(nb)
    return np.asarray(out)


def save_base_plots(y_true, y_prob, group_out, title_prefix):
    group_out = Path(group_out)
    group_out.mkdir(parents=True, exist_ok=True)

    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5), dpi=300)
    plt.plot(fpr, tpr, linewidth=2, label=f"Model (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Reference")
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(f"{title_prefix} ROC")
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(group_out / "external_1m_roc.png", dpi=300)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    event_rate = float(np.mean(y_true))

    plt.figure(figsize=(6, 5), dpi=300)
    plt.plot(recall, precision, linewidth=2, label=f"Model (AUPRC={auprc:.3f})")
    plt.axhline(event_rate, linestyle="--", linewidth=1, label=f"Event rate={event_rate:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_prefix} PR curve")
    plt.legend(frameon=False, loc="lower left")
    plt.tight_layout()
    plt.savefig(group_out / "external_1m_pr.png", dpi=300)
    plt.close()

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=min(10, max(2, len(y_true)//5)), strategy="quantile")

    plt.figure(figsize=(6, 5), dpi=300)
    plt.plot(mean_pred, frac_pos, marker="o", linewidth=2, label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Ideal")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed probability")
    plt.title(f"{title_prefix} calibration")
    plt.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.savefig(group_out / "external_1m_calibration.png", dpi=300)
    plt.close()

    thresholds = np.linspace(0.01, 0.99, 99)
    nb_model = net_benefit(y_true, y_prob, thresholds)
    treat_all = event_rate - (1 - event_rate) * thresholds / (1 - thresholds)
    treat_none = np.zeros_like(thresholds)

    plt.figure(figsize=(6, 5), dpi=300)
    plt.plot(thresholds, nb_model, linewidth=2, label="Model")
    plt.plot(thresholds, treat_all, linestyle="--", linewidth=1.3, label="Treat all")
    plt.plot(thresholds, treat_none, linestyle="-", linewidth=1.3, label="Treat none")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title(f"{title_prefix} DCA")
    plt.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(group_out / "external_1m_dca.png", dpi=300)
    plt.close()


def save_confusion_plot(y_true, y_prob, threshold, group_out, name):
    pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    cm = np.array([[tn, fp], [fn, tp]])

    plt.figure(figsize=(4.8, 4.2), dpi=300)
    plt.imshow(cm, interpolation="nearest")
    plt.title(name)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(Path(group_out) / f"{name.replace(' ', '_').replace('/', '_')}_confusion_matrix.png", dpi=300)
    plt.close()


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
        "youden_index": "Youden index",
    }

    for metric, label in label_map.items():
        sub = summary_df[summary_df["metric"] == metric]
        row[label] = sub["point_95ci"].iloc[0] if len(sub) else ""

    return row


def first_existing(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    return Path(paths[0])


def build_jobs(project_dir):
    p = Path(project_dir)

    return [
        {
            "group": "total",
            "internal_file": first_existing([p / "data10_no_spo2.xlsx", p / "data10.xlsx"]),
            "external_file": p / "data_temporal - total.xlsx",
            "model_root": p / "autogluon_data10_total_top10_shap_youden_no_spo2",
        },
        {
            "group": "age_plus",
            "internal_file": first_existing([p / "data10-old_no_spo2.xlsx", p / "data10-old.xlsx"]),
            "external_file": p / "data_temporal - Age+.xlsx",
            "model_root": p / "autogluon_data10_old_young_top10_shap_youden_no_spo2" / "old",
        },
        {
            "group": "age_minus",
            "internal_file": first_existing([p / "data10-young_no_spo2.xlsx", p / "data10-young.xlsx"]),
            "external_file": p / "data_temporal - Age-.xlsx",
            "model_root": p / "autogluon_data10_old_young_top10_shap_youden_no_spo2" / "young",
        },
        {
            "group": "cv_plus",
            "internal_file": first_existing([p / "data10-sub4_no_spo2.xlsx", p / "data10-sub4.xlsx"]),
            "external_file": p / "data_temporal - CV+.xlsx",
            "model_root": p / "autogluon_data10_sub4_sub4minus_top10_shap_youden_no_spo2" / "sub4",
        },
        {
            "group": "cv_minus",
            "internal_file": first_existing([p / "data10-sub4-_no_spo2.xlsx", p / "data10-sub4-.xlsx"]),
            "external_file": p / "data_temporal - CV-.xlsx",
            "model_root": p / "autogluon_data10_sub4_sub4minus_top10_shap_youden_no_spo2" / "sub4_minus",
        },
    ]


def process_group(job, output_root, n_bootstrap, seed):
    group = job["group"]
    internal_file = Path(job["internal_file"])
    external_file = Path(job["external_file"])
    model_root = Path(job["model_root"])
    outcome_dir = model_root / OUTCOME_KEY
    model_dir = outcome_dir / "ag_model"

    print("=" * 100)
    print(f"Group: {group}")
    print(f"Internal file: {internal_file}")
    print(f"External file: {external_file}")
    print(f"Model dir: {model_dir}")

    if not internal_file.exists():
        raise FileNotFoundError(f"Internal data not found: {internal_file}")
    if not external_file.exists():
        raise FileNotFoundError(f"External data not found: {external_file}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    predictor = TabularPredictor.load(str(model_dir))

    internal_df = preprocess_for_1m(load_data(internal_file), remove_spo2=True)
    external_df = preprocess_for_1m(load_data(external_file), remove_spo2=True)

    train_df, test_df = train_test_split(
        internal_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=internal_df[OUTCOME_COL],
    )

    features = get_model_features(predictor)

    test_df, added_internal = complete_missing_features(test_df, features)
    external_df, added_external = complete_missing_features(external_df, features)

    missing_internal = [f for f in features if f not in test_df.columns]
    missing_external = [f for f in features if f not in external_df.columns]

    if missing_internal:
        raise ValueError(f"Missing internal model features: {missing_internal}")
    if missing_external:
        raise ValueError(f"Missing external model features: {missing_external}")

    X_internal = test_df[features].copy()
    y_internal = test_df[OUTCOME_COL].astype(int).values

    X_external = external_df[features].copy()
    y_external = external_df[OUTCOME_COL].astype(int).values

    model_name, model_source = select_non_weighted_model(predictor, X_external, outcome_dir)
    print(f"Selected model: {model_name} ({model_source})")

    internal_prob = positive_proba(predictor, X_internal, model=model_name)
    external_prob = positive_proba(predictor, X_external, model=model_name)

    youden_info = find_youden_threshold(y_internal, internal_prob)
    base_threshold = float(youden_info["threshold"])

    group_out = Path(output_root) / group
    group_out.mkdir(parents=True, exist_ok=True)

    pred_df = external_df.copy()
    pred_df["y_true"] = y_external
    pred_df["y_prob"] = external_prob
    pred_df["model_name"] = model_name
    pred_df["internal_youden_threshold"] = base_threshold

    save_base_plots(y_external, external_prob, group_out, f"{group} external 1m")

    all_point = []
    all_summary = []
    all_boot = []
    paper_rows = []

    for mult in THRESHOLD_MULTIPLIERS:
        th = base_threshold * float(mult)
        th = max(0.0, min(0.999999, th))

        label = f"{mult:g}x_youden"
        print(f"  Threshold {label}: {th:.6f}")

        point_df, boot_df, summary_df = bootstrap_ci(
            y_true=y_external,
            y_prob=external_prob,
            threshold=th,
            n_bootstrap=n_bootstrap,
            seed=seed + int(mult * 10000),
        )

        meta = {
            "Group": group,
            "External file": external_file.name,
            "Model": model_name,
            "Model source": model_source,
            "Threshold multiplier": float(mult),
            "Internal Youden threshold": base_threshold,
            "Applied threshold": th,
            "Threshold label": label,
            "N": int(len(y_external)),
            "Events": int(np.sum(y_external == 1)),
            "Event rate": float(np.mean(y_external)),
            "Internal N for threshold": int(len(y_internal)),
            "Internal events for threshold": int(np.sum(y_internal == 1)),
            "Internal Youden sensitivity": youden_info["sensitivity"],
            "Internal Youden specificity": youden_info["specificity"],
            "Internal Youden index": youden_info["youden"],
            "internal_file": str(internal_file),
            "external_file": str(external_file),
            "model_root": str(model_root),
            "model_dir": str(model_dir),
            "added_internal_features": "; ".join(added_internal),
            "added_external_features": "; ".join(added_external),
            "n_bootstrap": n_bootstrap,
        }

        for k, v in meta.items():
            point_df[k] = v
            boot_df[k] = v
            summary_df[k] = v

        all_point.append(point_df)
        all_summary.append(summary_df)
        all_boot.append(boot_df)
        paper_rows.append(paper_row_from_summary(summary_df, meta))

        pred_df[f"threshold_{label}"] = th
        pred_df[f"y_pred_{label}"] = (external_prob >= th).astype(int)

        save_confusion_plot(
            y_true=y_external,
            y_prob=external_prob,
            threshold=th,
            group_out=group_out,
            name=f"{group} {label}",
        )

    pred_df.to_csv(group_out / "external_1m_predictions_three_thresholds.csv", index=False)

    group_point = pd.concat(all_point, ignore_index=True)
    group_summary = pd.concat(all_summary, ignore_index=True)
    group_boot = pd.concat(all_boot, ignore_index=True)
    group_paper = pd.DataFrame(paper_rows)

    group_point.to_csv(group_out / "external_1m_point_metrics_three_thresholds.csv", index=False)
    group_summary.to_csv(group_out / "external_1m_bootstrap_95ci_three_thresholds_long.csv", index=False)
    group_boot.to_csv(group_out / "external_1m_bootstrap_distribution_three_thresholds.csv", index=False)
    group_paper.to_csv(group_out / "external_1m_paper_table_three_thresholds.csv", index=False)

    return group_summary, group_boot, group_point, group_paper


def main():
    parser = argparse.ArgumentParser(description="External 1m validation using temporal files and 1/1.25/1.5x Youden thresholds.")
    parser.add_argument("--project-dir", default="/Users/yan_1nvincible/Documents/muhong 机器学习")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--only", nargs="*", default=None, help="Optional: total age_plus age_minus cv_plus cv_minus")
    args = parser.parse_args()

    project_dir = Path(args.project_dir)

    if args.outdir is None:
        output_root = project_dir / "temporal_external_1m_three_youden_thresholds_no_spo2"
    else:
        output_root = Path(args.outdir)

    output_root.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(project_dir)

    if args.only:
        keep = set(args.only)
        jobs = [j for j in jobs if j["group"] in keep]

    all_summary = []
    all_boot = []
    all_point = []
    all_paper = []
    errors = []

    for idx, job in enumerate(jobs):
        try:
            summary_df, boot_df, point_df, paper_df = process_group(
                job=job,
                output_root=output_root,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed + idx * 100000,
            )
            all_summary.append(summary_df)
            all_boot.append(boot_df)
            all_point.append(point_df)
            all_paper.append(paper_df)
        except Exception as e:
            err = f"[{job['group']}] FAILED: {e}\n{traceback.format_exc()}"
            print(err)
            errors.append(err)

    long_df = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    boot_df = pd.concat(all_boot, ignore_index=True) if all_boot else pd.DataFrame()
    point_df = pd.concat(all_point, ignore_index=True) if all_point else pd.DataFrame()
    paper_df = pd.concat(all_paper, ignore_index=True) if all_paper else pd.DataFrame()

    long_path = output_root / "temporal_external_1m_three_youden_thresholds_long.csv"
    point_path = output_root / "temporal_external_1m_three_youden_thresholds_point_metrics.csv"
    paper_path = output_root / "temporal_external_1m_three_youden_thresholds_paper_table.csv"
    boot_path = output_root / "temporal_external_1m_three_youden_thresholds_bootstrap_distribution.csv"
    xlsx_path = output_root / "temporal_external_1m_three_youden_thresholds_summary.xlsx"

    long_df.to_csv(long_path, index=False)
    point_df.to_csv(point_path, index=False)
    paper_df.to_csv(paper_path, index=False)
    boot_df.to_csv(boot_path, index=False)

    if errors:
        (output_root / "temporal_external_1m_three_youden_thresholds_errors.log").write_text("\n\n".join(errors), encoding="utf-8")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        paper_df.to_excel(writer, sheet_name="Paper_table", index=False)
        long_df.to_excel(writer, sheet_name="Long_summary", index=False)
        point_df.to_excel(writer, sheet_name="Point_metrics", index=False)
        boot_df.to_excel(writer, sheet_name="Bootstrap_distribution", index=False)
        if errors:
            pd.DataFrame({"error": errors}).to_excel(writer, sheet_name="Errors", index=False)

    print("=" * 100)
    print("External temporal validation finished.")
    print("Output directory:", output_root)
    print("Key Excel:", xlsx_path)
    if errors:
        print("Some groups failed. See errors log in output folder.")


if __name__ == "__main__":
    main()
