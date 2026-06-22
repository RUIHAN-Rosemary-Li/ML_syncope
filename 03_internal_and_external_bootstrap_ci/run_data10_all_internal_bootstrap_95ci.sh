#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="/Users/yan_1nvincible/Documents/muhong 机器学习"
PYTHON_BIN="/Users/yan_1nvincible/ag_env/bin/python"
BOOTSTRAP_SCRIPT="${PROJECT_DIR}/bootstrap_ci_autogluon_validation.py"

N_BOOTSTRAP=1000
LOG_FILE="${PROJECT_DIR}/data10_all_internal_bootstrap_95ci.log"

cd "${PROJECT_DIR}"

echo "============================================================" | tee "${LOG_FILE}"
echo "Run internal bootstrap 95% CI for data10 overall, age subgroups, and sub4 subgroups" | tee -a "${LOG_FILE}"
echo "Project dir: ${PROJECT_DIR}" | tee -a "${LOG_FILE}"
echo "Python: ${PYTHON_BIN}" | tee -a "${LOG_FILE}"
echo "Bootstrap script: ${BOOTSTRAP_SCRIPT}" | tee -a "${LOG_FILE}"
echo "Bootstrap repeats: ${N_BOOTSTRAP}" | tee -a "${LOG_FILE}"
echo "Started at: $(date)" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "ERROR: Python not found or not executable: ${PYTHON_BIN}" | tee -a "${LOG_FILE}"
  exit 1
fi

if [ ! -f "${BOOTSTRAP_SCRIPT}" ]; then
  echo "ERROR: bootstrap script not found: ${BOOTSTRAP_SCRIPT}" | tee -a "${LOG_FILE}"
  echo "Please put bootstrap_ci_autogluon_validation.py into ${PROJECT_DIR}" | tee -a "${LOG_FILE}"
  exit 1
fi

run_one () {
  local group_name="$1"
  local data_file="$2"
  local model_root="$3"
  local outcome="$4"
  local outdir="${model_root}/${outcome}/bootstrap_95ci_internal"

  echo "" | tee -a "${LOG_FILE}"
  echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"
  echo "Group: ${group_name}" | tee -a "${LOG_FILE}"
  echo "Outcome: ${outcome}" | tee -a "${LOG_FILE}"
  echo "Data file: ${data_file}" | tee -a "${LOG_FILE}"
  echo "Model root: ${model_root}" | tee -a "${LOG_FILE}"
  echo "Outdir: ${outdir}" | tee -a "${LOG_FILE}"
  echo "Started: $(date)" | tee -a "${LOG_FILE}"

  if [ ! -f "${data_file}" ]; then
    echo "ERROR: data file not found: ${data_file}" | tee -a "${LOG_FILE}"
    return 1
  fi

  if [ ! -d "${model_root}/${outcome}/ag_model" ]; then
    echo "ERROR: model dir not found: ${model_root}/${outcome}/ag_model" | tee -a "${LOG_FILE}"
    return 1
  fi

  "${PYTHON_BIN}" -u "${BOOTSTRAP_SCRIPT}" \
    --mode internal \
    --data-file "${data_file}" \
    --model-root "${model_root}" \
    --outcome "${outcome}" \
    --n-bootstrap "${N_BOOTSTRAP}" \
    --outdir "${outdir}" \
    2>&1 | tee -a "${LOG_FILE}"

  local status=${PIPESTATUS[0]}
  if [ ${status} -ne 0 ]; then
    echo "FAILED: ${group_name} ${outcome}" | tee -a "${LOG_FILE}"
    return ${status}
  fi

  echo "Finished: ${group_name} ${outcome} at $(date)" | tee -a "${LOG_FILE}"
  return 0
}

# ------------------------------------------------------------
# Internal bootstrap tasks
# ------------------------------------------------------------

ERROR_COUNT=0

# 1. Overall data10
for outcome in 1w 1m 1y; do
  run_one \
    "overall" \
    "${PROJECT_DIR}/data10.xlsx" \
    "${PROJECT_DIR}/autogluon_data10_total_top10_shap_youden" \
    "${outcome}" || ERROR_COUNT=$((ERROR_COUNT + 1))
done

# 2. Age subgroups: young / old
for outcome in 1w 1m 1y; do
  run_one \
    "young" \
    "${PROJECT_DIR}/data10-young.xlsx" \
    "${PROJECT_DIR}/autogluon_data10_old_young_top10_shap_youden/young" \
    "${outcome}" || ERROR_COUNT=$((ERROR_COUNT + 1))

  run_one \
    "old" \
    "${PROJECT_DIR}/data10-old.xlsx" \
    "${PROJECT_DIR}/autogluon_data10_old_young_top10_shap_youden/old" \
    "${outcome}" || ERROR_COUNT=$((ERROR_COUNT + 1))
done

# 3. sub4 subgroups: sub4 / sub4_minus
for outcome in 1w 1m 1y; do
  run_one \
    "sub4" \
    "${PROJECT_DIR}/data10-sub4.xlsx" \
    "${PROJECT_DIR}/autogluon_data10_sub4_sub4minus_top10_shap_youden/sub4" \
    "${outcome}" || ERROR_COUNT=$((ERROR_COUNT + 1))

  run_one \
    "sub4_minus" \
    "${PROJECT_DIR}/data10-sub4-.xlsx" \
    "${PROJECT_DIR}/autogluon_data10_sub4_sub4minus_top10_shap_youden/sub4_minus" \
    "${outcome}" || ERROR_COUNT=$((ERROR_COUNT + 1))
done

# ------------------------------------------------------------
# Combine all bootstrap summaries into one table
# ------------------------------------------------------------

echo "" | tee -a "${LOG_FILE}"
echo "Combining bootstrap 95% CI summaries..." | tee -a "${LOG_FILE}"

"${PYTHON_BIN}" - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path("/Users/yan_1nvincible/Documents/muhong 机器学习")

tasks = [
    {
        "group": "overall",
        "data_file": PROJECT_DIR / "data10.xlsx",
        "model_root": PROJECT_DIR / "autogluon_data10_total_top10_shap_youden",
    },
    {
        "group": "young",
        "data_file": PROJECT_DIR / "data10-young.xlsx",
        "model_root": PROJECT_DIR / "autogluon_data10_old_young_top10_shap_youden" / "young",
    },
    {
        "group": "old",
        "data_file": PROJECT_DIR / "data10-old.xlsx",
        "model_root": PROJECT_DIR / "autogluon_data10_old_young_top10_shap_youden" / "old",
    },
    {
        "group": "sub4",
        "data_file": PROJECT_DIR / "data10-sub4.xlsx",
        "model_root": PROJECT_DIR / "autogluon_data10_sub4_sub4minus_top10_shap_youden" / "sub4",
    },
    {
        "group": "sub4_minus",
        "data_file": PROJECT_DIR / "data10-sub4-.xlsx",
        "model_root": PROJECT_DIR / "autogluon_data10_sub4_sub4minus_top10_shap_youden" / "sub4_minus",
    },
]

outcomes = ["1w", "1m", "1y"]

long_parts = []
paper_rows = []

for task in tasks:
    group = task["group"]
    model_root = task["model_root"]

    for outcome in outcomes:
        f = model_root / outcome / "bootstrap_95ci_internal" / f"{outcome}_internal_bootstrap_95ci_summary.csv"

        if not f.exists():
            print("Missing summary:", f)
            continue

        df = pd.read_csv(f)
        df.insert(0, "Group", group)
        df.insert(1, "Outcome", outcome)
        long_parts.append(df)

        row = {
            "Group": group,
            "Outcome": outcome,
            "Model": df["model_name"].iloc[0] if "model_name" in df.columns else "",
            "Threshold": df["threshold"].iloc[0] if "threshold" in df.columns else "",
            "Threshold source": df["threshold_source"].iloc[0] if "threshold_source" in df.columns else "",
            "N": df["n"].iloc[0] if "n" in df.columns else "",
            "Events": df["event_n"].iloc[0] if "event_n" in df.columns else "",
            "Event rate": df["event_rate"].iloc[0] if "event_rate" in df.columns else "",
        }

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
            sub = df[df["metric"] == metric]
            row[label] = sub["point_95ci"].iloc[0] if len(sub) and "point_95ci" in sub.columns else ""

        paper_rows.append(row)

export_dir = PROJECT_DIR / "data10_internal_bootstrap_95ci_all_groups"
export_dir.mkdir(exist_ok=True)

if long_parts:
    long_df = pd.concat(long_parts, ignore_index=True)
else:
    long_df = pd.DataFrame()

paper_df = pd.DataFrame(paper_rows)

long_path = export_dir / "data10_all_groups_internal_bootstrap_95ci_long.csv"
paper_path = export_dir / "data10_all_groups_internal_bootstrap_95ci_paper_table.csv"
xlsx_path = export_dir / "data10_all_groups_internal_bootstrap_95ci_summary.xlsx"

long_df.to_csv(long_path, index=False)
paper_df.to_csv(paper_path, index=False)

with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    paper_df.to_excel(writer, sheet_name="Paper_table", index=False)
    long_df.to_excel(writer, sheet_name="Long_summary", index=False)

print("Saved:", long_path)
print("Saved:", paper_path)
print("Saved:", xlsx_path)
PY

echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "All bootstrap tasks finished with ERROR_COUNT=${ERROR_COUNT}" | tee -a "${LOG_FILE}"
echo "Combined output dir: ${PROJECT_DIR}/data10_internal_bootstrap_95ci_all_groups" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "Finished at: $(date)" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

if [ "${ERROR_COUNT}" -ne 0 ]; then
  exit 1
fi
