#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bootstrap 95% CI for AutoGluon internal or external validation.

Core logic:
- Load a trained AutoGluon model.
- Reconstruct internal 7:3 test set OR read external validation data.
- Predict probability.
- Fix the internal Youden threshold from *_model_metrics_optimal_threshold.csv.
- Bootstrap validation samples 1000 times.
- Report 95% CI using 2.5% and 97.5% percentiles.

Outputs:
<outdir>/
  <outcome>_<mode>_predictions.csv
  <outcome>_<mode>_point_metrics.csv
  <outcome>_<mode>_bootstrap_distribution.csv
  <outcome>_<mode>_bootstrap_95ci_summary.csv
  <outcome>_<mode>_bootstrap_95ci_summary.xlsx
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, f1_score, confusion_matrix
)
from autogluon.tabular import TabularPredictor


OUTCOME_MAP = {"1w": "1wk", "1m": "1mo", "1y": "1yr"}
RANDOM_STATE = 42
TEST_SIZE = 0.30
COAG_COLUMNS = ["PT", "INR", "APTT", "Fbg", "TT"]


def normalize_gender_value(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ["1", "1.0", "m", "male", "man", "男", "男性"]:
        return "Male"
    if s in ["0", "0.0", "2", "2.0", "f", "female", "woman", "女", "女性"]:
        return "Female"
    return str(x).strip()


def safe_to_numeric(s):
    if pd.api.types.is_numeric_dtype(s):
        return s
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() >= max(1, int(s.notna().sum() * 0.8)):
        return x
    return s


def normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    rename_map = {
        "Sex": "gender", "sex": "gender", "Gender": "gender", "性别": "gender",
        "Age": "age", "年龄": "age",
        "血压": "BP",
        "Systolic BP": "SBP", "Diastolic BP": "DBP",
        "脉搏": "Pulse",
        "血氧(%)": "Spo", "血氧（%）": "Spo", "血氧": "Spo",
        "SpO2": "Spo", "SPO2": "Spo",
        "HTN": "HT", "Hypertension": "HT",
        "Old MI": "OMI", "Old myocardial infarction": "OMI",
        "Old CVA": "OCI", "Old cerebral infarction": "OCI", "OCI 梗": "OCI",
        "Arrhythmia": "Arrythmia", "心律失常": "Arrythmia",
        "肿瘤史": "Tumor",
        "Hgb": "HGB", "Hb": "HGB", "HCT%": "HCT", "Hct": "HCT",
        "Cr": "Crea", "Creatinine": "Crea",
        "CK-MB": "CKMB", "CK-MB ng/ml": "CKMB",
        "TnI": "Tni", "TNI": "Tni",
        "QRSWide": "QRSd", "QRS wide": "QRSd", "宽QRS波": "QRSd",
        "电轴异常": "AxisAb", "QT延长": "QTProl",
        "左束支": "LBBB", "左心肥厚": "LVH",
        "Outcome1w": "1wk", "Outcome 1w": "1wk", "1周内就诊原因": "1wk", "1w": "1wk",
        "Outcome1m": "1mo", "Outcome 1m": "1mo", "1月内就诊原因": "1mo", "1m": "1mo",
        "Outcome1y": "1yr", "Outcome 1y": "1yr", "1年内就诊原因": "1yr", "1y": "1yr",
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


def preprocess_internal(df_raw, outcome_col):
    df = df_raw.copy()
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column not found: {outcome_col}; columns={list(df.columns)}")
    df = df.dropna(subset=[outcome_col]).copy()
    for c in OUTCOME_MAP.values():
        if c != outcome_col and c in df.columns:
            df = df.drop(columns=c)
    df = df.loc[df.isna().mean(axis=1) <= 0.5].copy()
    drop_cols = [c for c in COAG_COLUMNS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    if "Spo" in df.columns:
        n_missing = df["Spo"].isna().sum()
        if n_missing:
            rng = np.random.default_rng(RANDOM_STATE)
            df.loc[df["Spo"].isna(), "Spo"] = np.round(rng.uniform(95, 100, n_missing), 1)
    df[outcome_col] = df[outcome_col].astype(int)
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
    raise RuntimeError("Cannot determine model features from predictor.")


def complete_missing_features(df, features):
    df = df.copy()
    added = []
    if "BP" in features and "BP" not in df.columns:
        if "SBP" in df.columns and "DBP" in df.columns:
            df["BP"] = ((pd.to_numeric(df["SBP"], errors="coerce") >= 140) |
                        (pd.to_numeric(df["DBP"], errors="coerce") >= 90)).astype(float)
            added.append("BP_created_from_SBP_DBP_ge_140_90")
        else:
            df["BP"] = np.nan
            added.append("BP_created_as_missing")
    if "Spo" in features and "Spo" not in df.columns:
        df["Spo"] = 98.0
        added.append("Spo_created_as_constant_98")
    return df, added


def outcome_dir_from_root(model_root, outcome_key):
    root = Path(model_root)
    if (root / outcome_key / "ag_model").exists():
        return root / outcome_key
    if (root / "ag_model").exists():
        return root
    raise FileNotFoundError(f"Cannot find ag_model for {outcome_key} under {model_root}")


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
    return arr[:, -1].astype(float) if arr.ndim == 2 else arr.astype(float)


def candidate_models(outcome_dir, outcome_key):
    files = [
        outcome_dir / f"{outcome_key}_model_metrics_optimal_threshold.csv",
        outcome_dir / f"{outcome_key}_model_metrics_optimal_threshold_recalculated.csv",
        outcome_dir / f"{outcome_key}_model_metrics_default_threshold.csv",
    ]
    for f in files:
        if f.exists():
            df = pd.read_csv(f)
            if "model" not in df.columns:
                continue
            sort_cols = [c for c in ["auc", "auprc", "f1"] if c in df.columns]
            if sort_cols:
                df = df.sort_values(sort_cols, ascending=False)
            return df["model"].astype(str).tolist(), df, str(f)
    return [], None, ""


def threshold_for_model(outcome_dir, outcome_key, model, explicit_threshold=None):
    if explicit_threshold is not None:
        return float(explicit_threshold), "manual_argument"
    models, df, f = candidate_models(outcome_dir, outcome_key)
    if df is not None:
        row = df[df["model"].astype(str) == str(model)]
        if len(row):
            for c in ["optimal_threshold", "reported_threshold", "threshold"]:
                if c in row.columns:
                    return float(row.iloc[0][c]), f"internal_youden_from_{Path(f).name}:{c}"
    return 0.5, "fallback_0.5"


def select_model(predictor, X, outcome_dir, outcome_key, explicit_model=None, explicit_threshold=None):
    if explicit_model:
        models = [explicit_model]
    else:
        models, _, _ = candidate_models(outcome_dir, outcome_key)
        if not models:
            models = predictor.leaderboard(silent=True)["model"].astype(str).tolist()
    failed = []
    for m in models:
        try:
            _ = positive_proba(predictor, X.head(min(30, len(X))), m)
            thr, thr_src = threshold_for_model(outcome_dir, outcome_key, m, explicit_threshold)
            return m, thr, thr_src, failed
        except Exception as e:
            failed.append({"model": m, "error": str(e)[:1000]})
    raise RuntimeError(f"No working model. Failed candidates: {failed}")


def metrics(y, p, threshold):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    return {
        "auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        "auprc": average_precision_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        "brier_score": brier_score_loss(y, p),
        "accuracy": accuracy_score(y, pred),
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "f1": f1_score(y, pred, zero_division=0),
        "youden": sens + spec - 1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def bootstrap(y, p, threshold, n_boot=1000, seed=2026):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    n = len(y)
    point = metrics(y, p, threshold)
    rows = []
    skipped = 0
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        pp = p[idx]
        if len(np.unique(yy)) < 2:
            skipped += 1
            continue
        row = metrics(yy, pp, threshold)
        row["bootstrap_id"] = i + 1
        rows.append(row)
    boot_df = pd.DataFrame(rows)
    summary = []
    for m in ["auc", "auprc", "brier_score", "accuracy", "sensitivity", "specificity", "ppv", "npv", "f1", "youden"]:
        vals = pd.to_numeric(boot_df[m], errors="coerce").dropna()
        lo = np.percentile(vals, 2.5) if len(vals) else np.nan
        hi = np.percentile(vals, 97.5) if len(vals) else np.nan
        pe = point[m]
        summary.append({
            "metric": m,
            "point_estimate": pe,
            "ci_lower_2.5": lo,
            "ci_upper_97.5": hi,
            "point_95ci": f"{pe:.3f} ({lo:.3f}-{hi:.3f})" if pd.notna(lo) else "",
            "bootstrap_mean": vals.mean() if len(vals) else np.nan,
            "bootstrap_sd": vals.std(ddof=1) if len(vals) > 1 else np.nan,
            "n_bootstrap_valid": len(vals),
            "n_bootstrap_skipped": skipped,
        })
    return pd.DataFrame([point]), boot_df, pd.DataFrame(summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["internal", "external"])
    ap.add_argument("--data-file", required=True)
    ap.add_argument("--model-root", required=True)
    ap.add_argument("--outcome", default="1m", choices=["1w", "1m", "1y"])
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--test-size", type=float, default=TEST_SIZE)
    ap.add_argument("--split-seed", type=int, default=RANDOM_STATE)
    args = ap.parse_args()

    outcome_col = OUTCOME_MAP[args.outcome]
    outcome_dir = outcome_dir_from_root(args.model_root, args.outcome)
    predictor = TabularPredictor.load(str(outcome_dir / "ag_model"))

    df = load_data(args.data_file)
    if args.mode == "internal":
        dfp = preprocess_internal(df, outcome_col)
        _, eval_df = train_test_split(
            dfp,
            test_size=args.test_size,
            random_state=args.split_seed,
            stratify=dfp[outcome_col],
        )
    else:
        if outcome_col not in df.columns:
            raise ValueError(f"Outcome column {outcome_col} not found. Columns={list(df.columns)}")
        eval_df = df.dropna(subset=[outcome_col]).copy()
        eval_df[outcome_col] = eval_df[outcome_col].astype(int)

    features = get_model_features(predictor)
    eval_df, added = complete_missing_features(eval_df, features)

    missing = [f for f in features if f not in eval_df.columns]
    outdir = Path(args.outdir) if args.outdir else outcome_dir / f"bootstrap_95ci_{args.mode}"
    outdir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{
        "feature": f,
        "available": f in eval_df.columns,
        "missing_rate": eval_df[f].isna().mean() if f in eval_df.columns else np.nan,
        "dtype": str(eval_df[f].dtype) if f in eval_df.columns else "",
    } for f in features]).to_csv(outdir / "feature_compatibility_report.csv", index=False)

    if added:
        (outdir / "derived_features_added.txt").write_text("\n".join(added), encoding="utf-8")
    if missing:
        (outdir / "missing_model_features.txt").write_text("\n".join(missing), encoding="utf-8")
        raise ValueError(f"Missing model-required features: {missing}")

    X = eval_df[features].copy()
    y = eval_df[outcome_col].astype(int).values

    model_name, threshold, threshold_source, failed = select_model(
        predictor, X, outcome_dir, args.outcome,
        explicit_model=args.model,
        explicit_threshold=args.threshold,
    )
    if failed:
        pd.DataFrame(failed).to_csv(outdir / "failed_model_candidates.csv", index=False)

    p = positive_proba(predictor, X, model_name)

    pred_df = eval_df.copy()
    pred_df["predicted_probability"] = p
    pred_df["predicted_label"] = (p >= threshold).astype(int)
    pred_df.to_csv(outdir / f"{args.outcome}_{args.mode}_predictions.csv", index=False)

    point_df, boot_df, summary_df = bootstrap(y, p, threshold, args.n_bootstrap, args.seed)

    meta = {
        "mode": args.mode,
        "outcome": args.outcome,
        "outcome_col": outcome_col,
        "data_file": args.data_file,
        "model_root": args.model_root,
        "outcome_dir": str(outcome_dir),
        "model_name": model_name,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "n": len(y),
        "event_n": int(np.sum(y == 1)),
        "event_rate": float(np.mean(y)),
        "n_bootstrap_requested": args.n_bootstrap,
        "bootstrap_seed": args.seed,
    }

    point_df = pd.concat([pd.DataFrame([meta]), point_df], axis=1)
    for k, v in meta.items():
        summary_df[k] = v

    first = ["mode", "outcome", "model_name", "threshold", "threshold_source", "n", "event_n", "event_rate",
             "metric", "point_estimate", "ci_lower_2.5", "ci_upper_97.5", "point_95ci",
             "bootstrap_mean", "bootstrap_sd", "n_bootstrap_valid", "n_bootstrap_skipped"]
    summary_df = summary_df[first + [c for c in summary_df.columns if c not in first]]

    point_df.to_csv(outdir / f"{args.outcome}_{args.mode}_point_metrics.csv", index=False)
    boot_df.to_csv(outdir / f"{args.outcome}_{args.mode}_bootstrap_distribution.csv", index=False)
    summary_df.to_csv(outdir / f"{args.outcome}_{args.mode}_bootstrap_95ci_summary.csv", index=False)

    try:
        with pd.ExcelWriter(outdir / f"{args.outcome}_{args.mode}_bootstrap_95ci_summary.xlsx", engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Bootstrap_95CI", index=False)
            point_df.to_excel(writer, sheet_name="Point_metrics", index=False)
            boot_df.to_excel(writer, sheet_name="Bootstrap_distribution", index=False)
    except Exception as e:
        print("Excel export failed:", e)

    print("=" * 100)
    print("Bootstrap 95% CI completed.")
    print("Mode:", args.mode)
    print("Outcome:", args.outcome)
    print("Model:", model_name)
    print("Threshold:", threshold, "|", threshold_source)
    print("N:", len(y), "| Events:", int(np.sum(y == 1)), "| Event rate:", float(np.mean(y)))
    print("Saved to:", outdir)
    print(summary_df[["metric", "point_95ci"]])


if __name__ == "__main__":
    main()
