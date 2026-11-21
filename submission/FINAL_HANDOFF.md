# Final handoff

## Status

The reporting audit is `complete_with_declared_exceptions`. All 1,251 planned jobs completed: 270 Part 1 tuning, 837 Part 1 final, 36 bonus tuning, and 108 bonus final. Every final comparison cell contains seeds 13, 42, and 2026. The audit found no failed, blocked, stale, skipped, or missing planned jobs.

The declared exception is confined to an older derived bonus summary. Seven TF-IDF rows chose a different winner inside each seed. The final reporting code instead chooses one TF-IDF model per target and size from the complete three-seed Part 1 mean, then reuses its aligned seed rows. No atomic result was altered.

## Five findings to lead with

1. NSMC is the only domain with a sustained neural crossover at the prespecified 0.01 macro-F1 margin, beginning at 64,000 labels. Shopping and Steam do not cross through 159,918 and 79,828 labels.
2. Neural models lead at full data in all three domains, but the gains over the best TF-IDF model are 1.77 points on NSMC, 0.81 on Shopping, and 0.85 on Steam.
3. Those gains cost 46.5, 48.7, and 27.7 times the best TF-IDF fit time, respectively. These are practical mixed-device comparisons because TF-IDF ran on CPU and neural models on CUDA.
4. At 200 target labels, pooled CE improves on the target-only Transformer by 12.05 points on NSMC, 5.19 on Shopping, and 13.54 on Steam. Against TF-IDF, the corresponding differences are +2.20, -2.81, and +2.21 points.
5. SupCon and DB-CDSupCon do not consistently beat pooled CE. The target-only Transformer never crosses the TF-IDF reference within 200--2,000 labels, so new target-label savings are not estimable on this grid.

The mixed bonus outcome is part of the result: DB-CDSupCon helps selected cells, changes sign elsewhere, and improves the mean separation gap only on Shopping. Its fit-time ratio versus pooled CE ranges from 1.08 to 1.41, while the training-only projection head adds no deployed inference path.

## Deliverables

- Report: `submission/report/report.pdf` (10 main pages, followed by references and appendix; 16 pages total)
- Slides: `submission/presentation/slides.pdf` (17 main slides and 6 backup slides; about 12 minutes 10 seconds)
- Speaker notes: `submission/presentation/speaker_notes.md`
- Rehearsal script: `submission/presentation/presentation_script.md`
- Professor Q&A: `submission/presentation/professor_q_and_a.md`
- Evidence memo: `Source/results/reporting/evidence_memo.md`
- Audit: `Source/results/reporting/reporting_audit.md` and `reporting_audit.csv`
- Derived CSVs: `Source/results/reporting/derived_metrics/`
- Figure plotting data: `Source/results/reporting/figure_data/`
- Generated tables and figures: `submission/assets/tables/` and `submission/assets/figures/`
- Numeric trace map: `Source/results/reporting/displayed_value_traceability.json`

Rebuild from the repository root with:

```bash
cd submission
./build_all.sh
```

The last build passed the evidence audit, five reporting unit tests, displayed-number traceability, citation resolution, XeLaTeX compilation, and PDF text extraction with `pdftotext`. All report pages and slides were rendered to PNG and visually inspected. The remaining LaTeX messages are underfull-line notices caused by narrow two-column text, URLs, and monospace configuration rows; no content is clipped and there are no overfull boxes or unresolved references.

No citation is unresolved and no submission decision remains. The main limits to state during presentation are the three final seeds, three binary review domains, learned-from-scratch models, fixed tuning budget, and mixed CPU/GPU timing.

Ponytail full mode kept the reporting implementation to five focused scripts and reused repository schemas and helpers. Humanizer was applied only to report prose, slide wording, speaker materials, and this handoff; measured values, equations, citations, code, and generated tables were left unchanged.
