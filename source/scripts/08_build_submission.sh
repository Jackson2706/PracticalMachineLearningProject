#!/usr/bin/env bash
# Compile report and slides with XeLaTeX (needed for Korean text via kotex):
# latexmk -xelatex if xelatex + kotex are installed, otherwise tectonic (conda-forge, in env ML).
set -euo pipefail
source "$(dirname "$0")/_common.sh"

build() {  # $1 = directory, $2 = main .tex
  pushd "$1" >/dev/null
  if command -v latexmk >/dev/null 2>&1 && command -v xelatex >/dev/null 2>&1 && kpsewhich kotex.sty >/dev/null 2>&1; then
    latexmk -xelatex -interaction=nonstopmode -halt-on-error "$2"
  elif command -v tectonic >/dev/null 2>&1; then
    tectonic --keep-logs --keep-intermediates "$2"
  else
    echo "Neither latexmk nor tectonic found (install TeX Live / MacTeX, or tectonic)"; exit 1
  fi
  popd >/dev/null
}
build ../submission/report main.tex
build ../submission/slides slides.tex
ls -l ../submission/report/main.pdf ../submission/slides/slides.pdf
python - <<'PY'
from ksent.analysis.make_numbers import build
tbd = [n for n, v, _ in build().items if v is None]
print(f"TBD macros remaining: {len(tbd)}" + (f" (e.g. {tbd[:5]})" if tbd else ""))
import glob
ph = [f for f in glob.glob("../submission/report/tables/*.tex") if "--" in open(f).read()]
print(f"tables with '--' placeholders: {len(ph)}")
PY
