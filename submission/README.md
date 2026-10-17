# Submission: report and slides

* `report/main.tex`: the written report (sections in `report/sections/`: intro, related, data, preprocessing, methods, setup, results, ablation, discussion, limitations, conclusion, appendix).
* `slides/slides.tex`: the Beamer presentation. It reuses the report's tables, figures and numbers through the symlinks `slides/tables`, `slides/figures` and `slides/generated`.

## Auto-generated files (do not edit by hand)

| Path | Produced by |
|---|---|
| `report/tables/*.tex` (Tables 1–6, A1–A6) | `python -m ksent.analysis.make_tables` |
| `report/figures/fig2–fig5, figA1 *.pdf/.png` | `python -m ksent.analysis.make_figures` (`--placeholder` for dummy layouts) |
| `report/generated/numbers.tex` | `python -m ksent.analysis.make_numbers` |

* `report/figures/fig1_pipeline.tex` (TikZ) is hand-written.
* The prose cites results **only** through the macros in `numbers.tex`.
* Until real results exist:
  * a macro without data expands to a red `[TBD: name]`;
  * a table cell without data prints `--`;
  * a figure drawn without results carries a red PLACEHOLDER watermark.

## Build (XeLaTeX: Korean text uses `kotex` with the UnBatang/UnDotum fonts from TeX Live)

```bash
cd source && bash scripts/07_analyze.sh && bash scripts/08_build_submission.sh
# or manually:
cd submission/report && latexmk -xelatex main.tex      # or: tectonic main.tex
cd submission/slides && latexmk -xelatex slides.tex    # or: tectonic slides.tex
```

If your TeX distribution lacks the Un fonts, install the TeX Live package `unfonts-core`.
Alternatively, change `\setmainhangulfont{...}` to any installed Korean font.

Before submitting, check that nothing is left unfilled. `08_build_submission.sh` prints the number of remaining TBD macros and tables containing `--`.
