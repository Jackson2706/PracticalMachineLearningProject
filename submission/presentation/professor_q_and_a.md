# Professor Q&A

## Why these three datasets?

They provide three Korean review domains with different vocabulary and label mechanisms: movie sentiment, product ratings, and game recommendations. That variation tests whether a crossover is shared or domain-specific. The answer is domain-specific: only NSMC reaches the sustained one-point margin. See slides 4 and 8, Table 1, and Figure 3.

## Why begin at 200 and continue to full data?

Two hundred labels creates a genuinely low-data regime where learned-from-scratch representations are stressed. Geometric growth makes changes interpretable across orders of magnitude without spending runs on many nearly identical sizes. Appending the exact cleaned pool answers the practical full-data question. See slides 2, 4, and 7.

## Why macro-F1?

Macro-F1 gives equal weight to negative and positive performance and remains appropriate if a split or prediction becomes less balanced. The cleaned splits are close to balanced, so accuracy is similar, but macro-F1 preserves the class-level objective. See slide 6 and Table 1.

## Why no pretrained Korean language model?

The approved question compares classical pipelines with neural representations learned from the available labels. A pretrained model would add outside language knowledge and change the research question. It is the strongest next family to test, not a missing baseline for the stated from-scratch benchmark. See slides 3 and 16.

## How was model fairness controlled?

All models use the same fixed splits and identical nested subset IDs for each dataset, size, and seed. Data-dependent features are fitted only on the active subset. Every dataset-model pair gets ten tuning candidates, the same validation metric, and a frozen winner. Cost is reported separately because representations and devices differ. See slides 4--6 and the audit appendix.

## Why tune at 8,000 and freeze configurations?

Tuning at one shared, sufficiently informative budget prevents repeated test-like selection at every size and keeps the grid computationally feasible. Freezing makes the learning curve reflect added data rather than a changing hyperparameter search. The limitation is that an 8,000-label winner may not be optimal at 200 labels or full data. See slide 5 and the tuning backup slide.

## Why only three seeds, and what does that limit?

The complete protocol already contains 1,251 planned jobs. Three seeds provide visible run-to-run variation across the full grid, but they have little power for formal inference. I report sample SD and aligned seed differences and avoid significance claims. See slides 6 and 16 and the completeness backup slide.

## How is crossover defined?

At a size, the best complete neural three-seed mean must exceed the best complete TF-IDF three-seed mean by at least 0.01 macro-F1. A sustained crossover must satisfy that rule at every larger tested size. NSMC reaches it at 64,000; Shopping and Steam do not. See slides 6 and 8 and Table 4.

## Why can label savings be claimed only conditionally?

Savings are the target-only crossover size minus a candidate method's crossover size. Both thresholds must be observed on the tested grid. The target-only Transformer never crosses the TF-IDF reference from 200 to 2,000 labels, so the subtraction is undefined. The correct label-savings result is "not estimable," not zero. See slide 14 and Table 7.

## Why compare pooled CE before contrastive objectives?

SupCon and DB-CDSupCon receive the same 4,000 source labels as pooled CE. Without pooled CE, any improvement could come entirely from those extra examples. The matched pooled baseline isolates the incremental effect of the contrastive loss. See slide 11 and Figure 7.

## What does domain-balanced sampling control?

Each batch has eight examples from every domain-label cell, for six cells and an effective batch of 48. This prevents large domains or one sentiment from dominating which positives and negatives appear. The three new objectives use the same sampler, so their difference is the loss. See slide 12 and the Methods section.

## What is novel about DB-CDSupCon?

It is a limited objective modification for this setting. It upweights same-sentiment positives that come from another domain, with positive weights normalized per anchor, and combines that loss with cross-entropy. The project does not claim to invent supervised contrastive learning. See slide 12 and Equations 2--3.

## How do you know improvement is not caused only by additional source data?

The large improvement over target-only CE is mostly the pooling effect. The comparison between standard SupCon or DB-CDSupCon and pooled CE uses identical source data, target data, seeds, backbone, and sampler. Those incremental differences are small and mixed. See slides 11, 13, and 15 and the paired-difference CSV.

## Did source-domain transfer help every target domain?

Pooling improved target-only Transformer mean macro-F1 at every tested target and size. It did not beat the stronger TF-IDF reference everywhere. At 200 labels, pooled CE was 2.20 points above TF-IDF on NSMC, 2.81 below on Shopping, and 2.21 above on Steam. See slide 13 and Table 6.

## What is the inference overhead of the projection head?

There is none in the deployed graph. The 24,768-parameter projection head is used only to compute the training contrastive loss. Deployment retains the same 2,841,858-parameter Transformer and classifier for pooled CE, SupCon, and DB-CDSupCon. Measured latency differences among those objectives are run-level noise, not a different inference path. See slide 15 and the final backup slide.

## What would be the strongest next experiment?

Run the same low-label target rotations with a pretrained Korean encoder and an equal-total-label control. The pretrained encoder tests whether cross-domain contrastive learning helps when representations are already strong. The equal-total-label control separates the value of domain diversity from the value of simply adding more examples. Increase the seed count for the smaller bonus grid so paired uncertainty is more informative.

## Why did Shopping behave differently?

Shopping's short product reviews appear especially compatible with sparse lexical cues, and its TF-IDF baseline is strong from the smallest sizes. That is a plausible interpretation, not a tested causal mechanism. A follow-up could stratify by review length, vocabulary overlap, and sentiment cue type. See slides 7, 8, and 13.

## Why is Mamba not the winner?

The maintained Mamba implementation is included as a genuine selective state-space baseline, but this task uses short sequences capped at 128 tokens. Mamba's linear sequence-length advantage is less important there, and the frozen configurations did not produce the top macro-F1. This result does not generalize to long-document tasks. See Table 3 and the full-data backup slide.

## Was the report built from clean results?

Yes, with one declared reporting correction. All 1,251 planned jobs completed, all final cells contain three seeds, tuning records contain no test metrics, and one hardware ID covers timing. The original bonus summary mixed per-seed TF-IDF winners in seven rows. The final reporting pipeline selects one model by three-seed mean and reuses its aligned seeds; atomic values remain unchanged. See the audit backup slide and Source/results/reporting/reporting_audit.md.
