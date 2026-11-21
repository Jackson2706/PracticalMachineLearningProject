# Speaker notes

Target: 12 minutes, followed by 3 minutes of questions. The 17 main slides total about 12:10 at the target pace. Backup slides are not timed.

## Slide 1: Title

- Purpose: state the project in one sentence.
- Say: "I tested when learned-from-scratch neural sentiment models become worth their extra cost as Korean review labels grow. The answer depends strongly on the domain."
- Transition: "A single full-data score cannot answer that question."
- Target time: 0:30

## Slide 2: Why measure the whole learning curve?

- Purpose: motivate label scale as the independent variable.
- Say: At 200 labels, TF-IDF can estimate lexical associations while a neural model is still learning its representation. At full data, the neural model has a chance to close the gap, but cost becomes part of the decision.
- Transition: "That leads to the approved benchmark and a separate transfer extension."
- Target time: 0:40

## Slide 3: Approved benchmark first; bonus second

- Purpose: keep the course contribution and bonus extension distinct.
- Say: Part 1 is the approved benchmark. Part 2 asks a follow-on question using the completed Part 1 results. Continual learning and pretrained language models are outside the project.
- Transition: "The benchmark starts with three deliberately different review domains."
- Target time: 0:35

## Slide 4: Three domains and nested subsets

- Purpose: establish data scale, balance, and leakage control.
- Say: NSMC is movie text, Shopping is product text, and Steam is game text. Splits are nearly balanced. For each seed, every smaller training set is a strict prefix of the next, and every model gets identical IDs.
- Transition: "Those subsets feed nine complete pipelines."
- Target time: 0:40

## Slide 5: Nine pipelines

- Purpose: show breadth without reciting hyperparameters.
- Say: Four pipelines share word unigram and bigram TF-IDF. Five neural backbones share the tokenizer, vocabulary rule, training loop, and early stopping. Every dataset-model pair gets ten tuning candidates at 8,000 labels, then one frozen configuration.
- Transition: "I used one operational definition to decide whether the added complexity paid off."
- Target time: 0:45

## Slide 6: Metric and crossover

- Purpose: define macro-F1 and the sustained threshold before results.
- Say: Macro-F1 weights the two sentiment classes equally. The best neural mean must exceed the best TF-IDF mean by at least 0.01 at a size and at every larger tested size. Three seeds describe variability, not formal significance.
- Transition: "Here is what the complete learning curves show."
- Target time: 0:40

## Slide 7: Part 1 learning curves

- Purpose: show the scale-dependent ranking.
- Say: TF-IDF SVM or logistic regression leads early in all three domains. Neural curves rise more sharply near the largest budgets. The effect is clearest on NSMC, while Shopping keeps a particularly strong sparse baseline.
- Transition: "The difference plot makes the threshold result easier to see."
- Target time: 0:55

## Slide 8: Crossover

- Purpose: answer the primary Part 1 question directly.
- Say: NSMC first reaches and sustains the one-point margin at 64,000 labels. Shopping does not reach it through 159,918 labels. Steam does not reach it through 79,828. Neural methods still lead at full data, but the prespecified practical margin appears in only one domain.
- Transition: "Quality is only half of the model-selection question."
- Target time: 0:45

## Slide 9: Cost trade-off

- Purpose: connect full-data quality to training cost.
- Say: The strongest neural model takes 46.5 times the best TF-IDF fit time on NSMC, 48.7 times on Shopping, and 27.7 times on Steam. Full-data gains are 1.77, 0.81, and 0.85 macro-F1 points. CPU and GPU timing makes this a practical system comparison, not a same-device kernel comparison.
- Transition: "The low-label weakness of the target-only Transformer motivates the bonus."
- Target time: 0:50

## Slide 10: Why cross-domain learning?

- Purpose: define the transfer resource assumption.
- Say: The target contributes 200 to 2,000 new labels. Each source contributes 2,000 labels that already exist. Any savings would mean fewer new target labels, never fewer total annotations.
- Transition: "Five methods separate the effect of extra source data from the effect of the loss."
- Target time: 0:30

## Slide 11: Five bonus methods

- Purpose: clarify reused references and newly trained runs.
- Say: TF-IDF and target-only Transformer come directly from Part 1. Pooled CE, SupCon, and DB-CDSupCon are new, matched runs. Pooled CE is essential because a contrastive method also sees the source examples.
- Transition: "DB-CDSupCon changes how positives are weighted inside those matched runs."
- Target time: 0:35

## Slide 12: DB-CDSupCon

- Purpose: explain the proposed modification at presentation depth.
- Say: Balanced batches contain all six domain-label cells. Same-label positives from another domain receive weight alpha; within-domain positives keep weight one. The projection head exists only during training, so inference uses the same classifier as pooled CE.
- Transition: "The large result comes from pooling; the objective-specific result is smaller."
- Target time: 0:45

## Slide 13: Bonus target results

- Purpose: compare all five methods over target-label scale.
- Say: At 200 labels, pooled CE improves on target-only Transformer by 12.05 points on NSMC, 5.19 on Shopping, and 13.54 on Steam. Against TF-IDF, pooled CE is 2.20 points higher on NSMC, 2.81 lower on Shopping, and 2.21 higher on Steam. SupCon and DB-CDSupCon move above and below pooled CE depending on the cell.
- Transition: "That mixed pattern matters for the label-savings claim."
- Target time: 0:55

## Slide 14: Label efficiency

- Purpose: state why no numeric savings result is defensible.
- Say: Some pooled methods cross TF-IDF at 200 labels on NSMC and Steam. The target-only Transformer never crosses in the tested range. Since savings are the target-only threshold minus the candidate threshold, the reference quantity is missing. The correct result is not estimable, not zero.
- Transition: "The secondary diagnostics are also mixed."
- Target time: 0:45

## Slide 15: Transfer, representation, and overhead

- Purpose: show that source retention and representation separation do not guarantee target gain.
- Say: Contrastive methods sometimes improve target and source performance together, sometimes trade one for the other. DB-CDSupCon improves the average separation gap only on Shopping. Its mean fit-time overhead ranges from 1.08 to 1.41 times pooled CE, while deployed parameters and latency remain effectively unchanged.
- Transition: "The pattern is useful even though it is not a clean win."
- Target time: 0:55

## Slide 16: What worked and what did not

- Purpose: turn mixed findings into precise scope.
- Say: The controlled benchmark, NSMC crossover, and source pooling are stable findings. A universal crossover and contrastive dominance are not. The main limits are three seeds, three binary domains, no pretrained model, and hardware-dependent timing.
- Transition: "I would leave the model-selection rule as four short takeaways."
- Target time: 0:50

## Slide 17: Takeaways

- Purpose: finish with actionable decisions.
- Say: Start with linear TF-IDF. Test neural models near large label budgets because crossover is domain-specific. Price the last quality increment. In transfer work, establish pooled CE before adding contrastive losses.
- Transition: "I am happy to go into the audit, configurations, confusion matrices, or loss details."
- Target time: 0:35
