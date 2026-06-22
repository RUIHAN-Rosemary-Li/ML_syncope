#!/usr/bin/env bash
set -euo pipefail

# 同时重新绘制 CV+ / CV− 的内部验证和外部验证 DCA，加旧有评分。
# 目的：把 DCA 横轴阈值范围从 0.01–0.20 缩小到 0.01–0.15，
# 让曲线展示更集中，避免净获益快速下降后影响视觉呈现。
#
# 外部验证：只针对 1m temporal external validation
# 内部验证：默认跑 1w、1m、1y；如果只想跑 1m，把 INTERNAL_OUTCOMES 改成 "1m"

PROJECT_DIR="/Users/yan_1nvincible/Documents/muhong 机器学习"
PYTHON_BIN="/Users/yan_1nvincible/ag_env/bin/python"

XMIN="0.01"
XMAX="0.15"
INTERNAL_OUTCOMES="1w 1m 1y"

EXTERNAL_SCRIPT="make_dca_with_legacy_scores_direct_zoomed.py"
INTERNAL_SCRIPT="make_internal_dca_with_legacy_scores_no_spo2.py"

EXTERNAL_OUTDIR="temporal_external_1m_DCA_with_legacy_scores_ONLY_CV_0_01_0_15"
INTERNAL_OUTDIR="internal_validation_DCA_with_legacy_scores_ONLY_CV_0_01_0_15"

LOG_FILE="rerun_internal_external_cv_dca_legacy_0_01_0_15.log"

cd "${PROJECT_DIR}"

echo "============================================================" | tee "${LOG_FILE}"
echo "Replot CV+ and CV- DCA with legacy scores: internal + external" | tee -a "${LOG_FILE}"
echo "Started at: $(date)" | tee -a "${LOG_FILE}"
echo "Project dir: ${PROJECT_DIR}" | tee -a "${LOG_FILE}"
echo "Threshold probability range: ${XMIN} - ${XMAX}" | tee -a "${LOG_FILE}"
echo "Internal outcomes: ${INTERNAL_OUTCOMES}" | tee -a "${LOG_FILE}"
echo "External outdir: ${EXTERNAL_OUTDIR}" | tee -a "${LOG_FILE}"
echo "Internal outdir: ${INTERNAL_OUTDIR}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "ERROR: Python not found: ${PYTHON_BIN}" | tee -a "${LOG_FILE}"
  exit 1
fi

if [ ! -f "${EXTERNAL_SCRIPT}" ]; then
  echo "ERROR: ${EXTERNAL_SCRIPT} not found." | tee -a "${LOG_FILE}"
  echo "请先确认 make_dca_with_legacy_scores_direct_zoomed.py 已经在项目目录。" | tee -a "${LOG_FILE}"
  exit 1
fi

if [ ! -f "${INTERNAL_SCRIPT}" ]; then
  echo "ERROR: ${INTERNAL_SCRIPT} not found." | tee -a "${LOG_FILE}"
  echo "请先确认 make_internal_dca_with_legacy_scores_no_spo2.py 已经在项目目录。" | tee -a "${LOG_FILE}"
  exit 1
fi

echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "[1/2] Replot EXTERNAL CV+ / CV- DCA, x=${XMIN}-${XMAX}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

rm -rf "${EXTERNAL_OUTDIR}"

"${PYTHON_BIN}" -u "${EXTERNAL_SCRIPT}" \
  --project-dir "${PROJECT_DIR}" \
  --groups cv_plus cv_minus \
  --xmin "${XMIN}" \
  --xmax "${XMAX}" \
  --outdir "${EXTERNAL_OUTDIR}" \
  2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "[2/2] Replot INTERNAL CV+ / CV- DCA, x=${XMIN}-${XMAX}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

rm -rf "${INTERNAL_OUTDIR}"

"${PYTHON_BIN}" -u "${INTERNAL_SCRIPT}" \
  --project-dir "${PROJECT_DIR}" \
  --groups cv_plus cv_minus \
  --outcomes ${INTERNAL_OUTCOMES} \
  --xmin "${XMIN}" \
  --xmax "${XMAX}" \
  --outdir "${INTERNAL_OUTDIR}" \
  2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "DONE." | tee -a "${LOG_FILE}"
echo "Finished at: $(date)" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "External CV+ figure:" | tee -a "${LOG_FILE}"
echo "${PROJECT_DIR}/${EXTERNAL_OUTDIR}/cv_plus/cv_plus_dca_with_legacy_scores_zoomed.png" | tee -a "${LOG_FILE}"
echo "External CV- figure:" | tee -a "${LOG_FILE}"
echo "${PROJECT_DIR}/${EXTERNAL_OUTDIR}/cv_minus/cv_minus_dca_with_legacy_scores_zoomed.png" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Internal CV+ / CV- figures are under:" | tee -a "${LOG_FILE}"
echo "${PROJECT_DIR}/${INTERNAL_OUTDIR}/cv_plus/" | tee -a "${LOG_FILE}"
echo "${PROJECT_DIR}/${INTERNAL_OUTDIR}/cv_minus/" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
