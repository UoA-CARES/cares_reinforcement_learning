# Reporting Guidelines

The CARES RL Statistical Tool is designed to support rigorous, transparent, and reproducible reporting of reinforcement learning experiments. This guide explains how to interpret the generated statistical outputs and how they should be presented in publications.

The recommended reporting philosophy is:

1. Estimate the effect.
2. Quantify the uncertainty.
3. Compare competing methods.
4. Use hypothesis tests only as supplementary evidence.

This follows modern recommendations for empirical machine learning by emphasizing effect estimation and confidence intervals over binary significance testing.

The recommendations in this guide apply equally to the generated PDF report, publication-ready tables, figures, and raw CSV outputs.

## Choosing the Appropriate Statistics

The recommended statistics depend on the level of analysis.

| Analysis Level | Recommended Statistics |
|----------------|------------------------|
| Learning behaviour | Full, Early or Final AUC |
| Individual task | IQM, BCa 95% Confidence Interval, Probability of Improvement |
| Cross-task benchmark | Mean Superiority, Cross-task Probability of Improvement |
| Supplementary analysis | Average Rank, Friedman Test, Nemenyi Test |

Different statistics answer different research questions.

For example:

- **Early Window AUC** evaluates sample efficiency.
- **Full AUC** evaluates overall learning performance.
- **Final Window AUC** evaluates final policy quality.
- **IQM** provides a robust summary of task performance.
- **Mean Superiority** summarises performance across an entire benchmark.

## Reporting Workflow

When reviewing the generated outputs, the recommended workflow is:

1. Ensure all validation checks pass.
2. Examine the selected primary performance metric.
3. Assess the corresponding confidence interval.
4. Compare methods using Probability of Improvement or Mean Superiority.
5. Review supplementary statistical tests.
6. Draw conclusions supported by the reported evidence.

This progression naturally moves from estimating performance to quantifying uncertainty before considering statistical significance.

###  1. Confirm validity

Check that the run completed without validation warnings or incompatible configurations.

###  2. Inspect per-task performance

Look for:

- low or negative-return failures;
- unusually wide IQM intervals;
- tasks where the reference comparison clearly underperforms;
- sensitivity to early versus final performance.

###  3. Inspect direct reference comparisons

For each baseline, record:

- probability the reference comparison is better;
- confidence interval;
- W-T-L task record;
- tasks responsible for losses.

### 4. Inspect the benchmark summary

Use mean superiority to describe roster-wide performance and rank statistics to describe consistency.

### 5. Review supplementary statistical tests

Use Friedman and Nemenyi tests to determine whether observed ranking differences provide additional evidence beyond the reported benchmark statistics.

### 6. Write a bounded conclusion

A strong conclusion contains:

- benchmark scope;
- effect estimate;
- uncertainty;
- consistency;
- known weaknesses.

!!! example "Balanced conclusion"
    Across 12 tasks and 10 seeds per algorithm, YourAlgorithm achieved the highest mean superiority and an estimated 0.82 probability of improvement over SAC. The 95% interval [0.74, 0.89] favoured YourAlgorithm, although it ranked below SAC on one task and had wider uncertainty on two additional tasks.

## Reporting Task-Level Results

Task-level results should focus on the estimated performance, the associated uncertainty, and the practical interpretation of the observed effect. Statistical significance should support these conclusions rather than replace them.

When reporting task-level results, include:

- Primary performance metric (e.g., IQM).
- Corresponding 95% confidence interval.
- Probability of Improvement where appropriate.
- Discussion of the practical significance of the observed effect.

Avoid unsupported causal interpretations. The reported statistics describe **what happened**, not **why it happened**.

A supported statement is:

> SAC achieved a higher IQM than PPO on Walker Walk (IQM = 0.81, BCa 95% CI [0.76, 0.86]). The Probability of Improvement was 0.91, indicating consistently stronger performance across training seeds.

An unsupported causal interpretation is:

> SAC performs better because it explores more effectively.

Prefer objective, evidence-based language.

**Strong**

> SAC achieved a higher IQM than PPO on Walker Walk (IQM = 0.81, BCa 95% CI [0.76, 0.86]). The Probability of Improvement was 0.91.

**Acceptable**

> SAC achieved the highest IQM on Walker Walk.

**Weak**

> SAC won.

**Unsupported**

> SAC is the best reinforcement learning algorithm.

## Reporting Benchmark Results

Benchmark results should summarise the overall evidence across the evaluated benchmark while acknowledging meaningful task-level differences.

When reporting benchmark results, include:

- Number of benchmark tasks.
- Number of training seeds.
- Mean Superiority.
- Cross-task Probability of Improvement.
- Corresponding confidence intervals.
- Discussion of notable task-specific successes and limitations.

Benchmark statistics summarise evidence across the evaluated benchmark rather than proving universal superiority.

Prefer objective, evidence-based language.

**Strong**

> Algorithm X achieved the highest benchmark Mean Superiority across the evaluated benchmark (0.82, 95% CI [0.75, 0.88]) and a Cross-task Probability of Improvement of 0.82.

**Acceptable**

> Algorithm X achieved the highest benchmark Mean Superiority.

**Weak**

> Algorithm X won the benchmark.

**Unsupported**

> Algorithm X is superior to all reinforcement learning algorithms.

## Reference Comparison

Pairwise comparisons, Probability of Improvement, confidence intervals, and statistical significance tests are reported relative to the selected reference comparison.

The reference comparison should normally correspond to the primary baseline used throughout the study. Selecting a consistent reference ensures that pairwise analyses remain directly comparable across all tasks and benchmark summaries.

When choosing a reference comparison:

- Select the primary baseline before beginning the analysis.
- Use the same reference throughout the study.
- Choose the strongest or most widely accepted baseline where appropriate.
- Avoid changing the reference after reviewing the results.

Changing the reference comparison does not affect the underlying benchmark statistics, but it changes the interpretation of all pairwise analyses and can make comparisons between tables and figures more difficult.

For parameter sweeps and ablation studies, the reference comparison should normally correspond to the default or baseline configuration being evaluated.

### Recommended Wording

Prefer objective, evidence-based language when discussing experimental results.

| Instead of | Prefer |
|------------|--------|
| Algorithm A won. | Algorithm A achieved the highest benchmark Mean Superiority. |
| Significant improvement. | Higher IQM with a BCa 95% confidence interval excluding the baseline. |
| Better algorithm. | Higher estimated performance on the evaluated benchmark. |
| Always better. | Consistently better across the evaluated benchmark. |
| Clearly superior. | Demonstrated stronger benchmark performance within the evaluated tasks. |

The goal is to describe the statistical evidence rather than make unsupported claims.

When reporting a single task, discuss the effect estimate first, followed by the confidence interval, the Probability of Improvement, and finally any supplementary statistical tests.

### Reviewer Expectations

Most reinforcement learning conferences expect reported conclusions to be supported by more than a single point estimate.

A publication-quality benchmark should therefore report:

- Benchmark composition.
- Evaluation protocol.
- Number of training seeds.
- Primary performance metric.
- Confidence intervals.
- Probability of Improvement.
- Benchmark summary statistic.
- Discussion of limitations.

!!! note "Reviewer expectations"
    The CARES RL Statistical Tool is designed to support these expectations by providing robust effect estimates, confidence intervals, and benchmark summaries that allow readers to assess both the magnitude and reliability of reported improvements.

## Common Reporting Mistakes

### Reporting only p-values

Avoid reporting statistical significance without the corresponding effect estimate.

Instead report:

- IQM or Mean Superiority.
- Confidence intervals.
- Probability of Improvement.

before discussing p-values.

---

### Comparing raw rewards across tasks

Raw rewards often have different scales across environments.

Instead compare task-level summaries and benchmark statistics.

---

### Cherry-picking performance metrics

Choose the primary performance metric before analysing the benchmark.

Do not switch between Full, Early or Final AUC simply because one produces a more favourable result.

Different AUC summaries answer different scientific questions.

---

### Ignoring validation warnings

Validation failures indicate that compared experiments are not directly comparable.

Validation issues should be resolved before interpreting or publishing results.

---

### Changing the reference comparison

Changing the reference comparison after reviewing the results can make pairwise analyses difficult to interpret.

Choose the reference comparison before beginning the analysis and use it consistently throughout the study.

---

### Reporting averages without uncertainty

Point estimates alone provide no indication of uncertainty.

Always report confidence intervals alongside the corresponding statistic.

---

### Overstating benchmark conclusions

Benchmark statistics summarise performance across the evaluated benchmark.

They do not imply universal superiority across every reinforcement learning task.

Always discuss important task-specific behaviour where scientifically relevant.

---

### Comparing incompatible experiments

Experiments with different evaluation schedules, reward definitions or incompatible configurations should not be compared directly.

The validation stage is designed to detect these situations before analysis.

## Minimum Reporting Standard

A publication-quality reinforcement learning benchmark should report:

- benchmark composition;
- evaluation protocol;
- number of tasks;
- number of training seeds;
- primary performance metric;
- confidence intervals;
- probability of improvement;
- benchmark summary statistic;
- discussion of notable task-specific behaviour;
- experimental limitations.

Following this reporting standard allows readers to assess both the magnitude and reliability of the reported improvements while supporting transparent and reproducible reinforcement learning research.

## Final Remarks

Statistical analysis should support scientific conclusions rather than replace scientific reasoning.

The reported statistics quantify the strength and uncertainty of the observed evidence, while the broader scientific interpretation should also consider the experimental design, benchmark selection, and practical significance of the results.

A well-reported benchmark communicates not only which method performed best, but also how reliable that conclusion is and the limitations under which it should be interpreted.