# Evidence memo

This memo is generated from the audited atomic records and corrected derived tables. Differences are descriptive because each final condition has three seeds.

## Part 1: macro-F1 and crossover

### NSMC

- Direct answer: the first sustained 0.01 crossover was 64,000 labels. At full data, Transformer reached 0.8185 mean macro-F1; the best TF-IDF method reached 0.8007.
- Support: part1_best_by_size.csv, part1_full_data_ranking.csv, and the Part 1 learning-curve and crossover figures.
- Interpretation: learned representations benefited more from the largest data budgets, but the size at which that mattered depended on domain.
- Alternative explanation: the fixed tuning budget and from-scratch vocabulary may favor different families at different scales.
- Strength: complete three-seed grid with shared subsets; descriptive, not a significance claim.
- Cannot conclude: performance of pretrained Korean language models or other domains.

### Shopping

- Direct answer: the first sustained 0.01 crossover was not observed through 159,918 labels. At full data, Transformer reached 0.8914 mean macro-F1; the best TF-IDF method reached 0.8834.
- Support: part1_best_by_size.csv, part1_full_data_ranking.csv, and the Part 1 learning-curve and crossover figures.
- Interpretation: learned representations benefited more from the largest data budgets, but the size at which that mattered depended on domain.
- Alternative explanation: the fixed tuning budget and from-scratch vocabulary may favor different families at different scales.
- Strength: complete three-seed grid with shared subsets; descriptive, not a significance claim.
- Cannot conclude: performance of pretrained Korean language models or other domains.

### Steam

- Direct answer: the first sustained 0.01 crossover was not observed through 79,828 labels. At full data, LSTM reached 0.7668 mean macro-F1; the best TF-IDF method reached 0.7582.
- Support: part1_best_by_size.csv, part1_full_data_ranking.csv, and the Part 1 learning-curve and crossover figures.
- Interpretation: learned representations benefited more from the largest data budgets, but the size at which that mattered depended on domain.
- Alternative explanation: the fixed tuning budget and from-scratch vocabulary may favor different families at different scales.
- Strength: complete three-seed grid with shared subsets; descriptive, not a significance claim.
- Cannot conclude: performance of pretrained Korean language models or other domains.

## Part 1: computational cost

- NSMC: Transformer gained +0.0177 macro-F1 over TF-IDF + LR at full data, with 46.5x mean fit time and 3.7x p50 end-to-end latency. CPU/GPU device differences confound direct latency attribution.
- Shopping: Transformer gained +0.0081 macro-F1 over TF-IDF + LR at full data, with 48.7x mean fit time and 3.9x p50 end-to-end latency. CPU/GPU device differences confound direct latency attribution.
- Steam: LSTM gained +0.0085 macro-F1 over TF-IDF + LR at full data, with 27.7x mean fit time and 2.0x p50 end-to-end latency. CPU/GPU device differences confound direct latency attribution.

## Bonus: target transfer

### NSMC

- Pooled CE minus Target-only Transformer averaged +0.1204 across the four target-label sizes; bonus_paired_differences.csv gives seed-level values.
- SupCon minus Pooled CE averaged -0.0009 across the four target-label sizes; bonus_paired_differences.csv gives seed-level values.
- DB-CDSupCon minus SupCon averaged +0.0045 across the four target-label sizes; bonus_paired_differences.csv gives seed-level values.
- DB-CDSupCon minus Pooled CE averaged +0.0036 across the four target-label sizes; bonus_paired_differences.csv gives seed-level values.
- Interpretation: pooling is compared before contrastive objectives, so source-data benefit is separated from loss-function benefit.
- Alternative explanation: pooled methods see more total labeled examples and balanced batches; this is the intended existing-source-data setting, not an equal-total-label comparison.
- Strength: matched seeds, examples, backbone, and sampler across the three newly trained objectives.
- Cannot conclude: formal significance or transfer to an unseen fourth domain.

### Shopping

- Pooled CE minus Target-only Transformer averaged +0.0539 across the four target-label sizes; bonus_paired_differences.csv gives seed-level values.
- SupCon minus Pooled CE averaged +0.0070 across the four target-label sizes; bonus_paired_differences.csv gives seed-level values.
- DB-CDSupCon minus SupCon averaged -0.0000 across the four target-label sizes; bonus_paired_differences.csv gives seed-level values.
- DB-CDSupCon minus Pooled CE averaged +0.0070 across the four target-label sizes; bonus_paired_differences.csv gives seed-level values.
- Interpretation: pooling is compared before contrastive objectives, so source-data benefit is separated from loss-function benefit.
- Alternative explanation: pooled methods see more total labeled examples and balanced batches; this is the intended existing-source-data setting, not an equal-total-label comparison.
- Strength: matched seeds, examples, backbone, and sampler across the three newly trained objectives.
- Cannot conclude: formal significance or transfer to an unseen fourth domain.

### Steam

- Pooled CE minus Target-only Transformer averaged +0.0957 across the four target-label sizes; bonus_paired_differences.csv gives seed-level values.
- SupCon minus Pooled CE averaged -0.0005 across the four target-label sizes; bonus_paired_differences.csv gives seed-level values.
- DB-CDSupCon minus SupCon averaged -0.0002 across the four target-label sizes; bonus_paired_differences.csv gives seed-level values.
- DB-CDSupCon minus Pooled CE averaged -0.0007 across the four target-label sizes; bonus_paired_differences.csv gives seed-level values.
- Interpretation: pooling is compared before contrastive objectives, so source-data benefit is separated from loss-function benefit.
- Alternative explanation: pooled methods see more total labeled examples and balanced batches; this is the intended existing-source-data setting, not an equal-total-label comparison.
- Strength: matched seeds, examples, backbone, and sampler across the three newly trained objectives.
- Cannot conclude: formal significance or transfer to an unseen fourth domain.

## Bonus: labels, source retention, representation, and overhead

- NSMC: new target-label savings were not estimable because the target-only crossover was not observed. DB-CDSupCon changed the mean separation gap by -0.0071 versus pooled CE and required 1.27x mean fit time. Its projection head is training-only, so deployed parameter count is unchanged.
- Shopping: new target-label savings were not estimable because the target-only crossover was not observed. DB-CDSupCon changed the mean separation gap by +0.0081 versus pooled CE and required 1.41x mean fit time. Its projection head is training-only, so deployed parameter count is unchanged.
- Steam: new target-label savings were not estimable because the target-only crossover was not observed. DB-CDSupCon changed the mean separation gap by -0.0074 versus pooled CE and required 1.08x mean fit time. Its projection head is training-only, so deployed parameter count is unchanged.

The evidence supports a mixed bonus result. Pooled source data usually helps the weak target-only Transformer, but it does not consistently beat the stronger TF-IDF reference. SupCon and DB-CDSupCon effects change sign across domains and sizes. No total annotation-savings claim is supported.
