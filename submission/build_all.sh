#!/usr/bin/env bash
set -euo pipefail

SUBMISSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SUBMISSION_DIR/.." && pwd)"
LOG_DIR="$SUBMISSION_DIR/build_logs"
CACHE_DIR="$PROJECT_DIR/Source/results/reporting/.cache"
LATEXMK="/Library/TeX/texbin/latexmk"

mkdir -p "$LOG_DIR" "$CACHE_DIR/matplotlib" "$CACHE_DIR/fontconfig"
export MPLCONFIGDIR="$CACHE_DIR/matplotlib"
export XDG_CACHE_HOME="$CACHE_DIR"

cd "$PROJECT_DIR"
conda run -n ML python Source/reporting/audit_results.py --allow-declared-exceptions | tee "$LOG_DIR/audit.log"
conda run -n ML pytest -q Source/reporting/test_reporting.py | tee "$LOG_DIR/tests.log"
conda run -n ML python Source/reporting/build_analysis_tables.py | tee "$LOG_DIR/analysis.log"
conda run -n ML python Source/reporting/build_figures.py | tee "$LOG_DIR/figures.log"
conda run -n ML python Source/reporting/verify_report_values.py | tee "$LOG_DIR/verification_precompile.log"

cd "$SUBMISSION_DIR/report"
"$LATEXMK" -g -xelatex -jobname=report -interaction=nonstopmode -halt-on-error main.tex 2>&1 | tee "$LOG_DIR/report_latex.log"

cd "$SUBMISSION_DIR/presentation"
"$LATEXMK" -g -xelatex -jobname=slides -interaction=nonstopmode -halt-on-error slides.tex 2>&1 | tee "$LOG_DIR/slides_latex.log"

cd "$PROJECT_DIR"
conda run -n ML python Source/reporting/verify_report_values.py --require-pdfs | tee "$LOG_DIR/verification_postcompile.log"

mkdir -p "$LOG_DIR/rendered_report" "$LOG_DIR/rendered_slides"
find "$LOG_DIR/rendered_report" "$LOG_DIR/rendered_slides" -type f -name 'page-*.png' -delete
pdftoppm -png -r 110 "$SUBMISSION_DIR/report/report.pdf" "$LOG_DIR/rendered_report/page" >/dev/null
pdftoppm -png -r 100 "$SUBMISSION_DIR/presentation/slides.pdf" "$LOG_DIR/rendered_slides/page" >/dev/null

echo "Built submission/report/report.pdf and submission/presentation/slides.pdf"
