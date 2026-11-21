# Reproducible reporting pipeline

The reporting pipeline reads completed atomic run records and authoritative result CSVs. It does not run model training.

## Rebuild

From the repository root:

```bash
cd submission
./build_all.sh
```

The script uses the existing `ML` Conda environment for Python, audits the frozen evidence, runs reporting unit tests, rebuilds derived CSVs, tables, and figures, verifies number traceability and citations, compiles both XeLaTeX documents, and checks extracted PDF text. Logs are retained in `submission/build_logs/`.

Python reporting dependencies are declared in the `report` optional dependency group in `pyproject.toml`. The installed `ML` environment must include Matplotlib. XeLaTeX, `latexmk`, and Poppler (`pdftotext` and `pdftoppm`) are required. Final page renders are saved under `submission/build_logs/rendered_report/` and `rendered_slides/` for visual review.

## Authoritative inputs and outputs

- Evidence snapshot: `Source/results/reporting/frozen_result_manifest.json`
- Audit: `Source/results/reporting/reporting_audit.md` and `.csv`
- Derived analyses: `Source/results/reporting/derived_metrics/`
- Figure plotting data: `Source/results/reporting/figure_data/`
- Displayed-number map: `Source/results/reporting/displayed_value_traceability.json`
- Generated tables and figures: `submission/assets/`
- Final report: `submission/report/report.pdf`
- Final presentation: `submission/presentation/slides.pdf`

The report build does not copy terminal output or hand-enter measured values. Narrative result numbers use generated LaTeX macros; table cells and figures are built from derived data and covered by the traceability check.

## Audit exception

The atomic evidence is complete. The original bonus aggregate chose a TF-IDF winner independently within each seed in seven rows. This reporting pipeline corrects only the derived reference: it chooses one TF-IDF model per target and size by the complete three-seed Part 1 mean, then reuses its aligned seed rows. No atomic measurement was changed.
