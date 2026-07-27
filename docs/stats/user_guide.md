--8<-- "include/glossary.md"

# User Guide

This page explains how to structure experiment logs, run analyses, select comparison conditions, and interpret the generated report and output files.

## Data Format

The tool discovers either a single task or a directory containing multiple tasks - all tasks must be structured consistently. Each task contains one or more algorithm runs, each of which contains one or more numeric seed directories. Every seed directory must contain a `data/eval.csv` file with evaluation metrics logged at one or more training steps.

!!! note "CARES RL Logging"
    The CARES RL framework automatically generates the required directory structure and evaluation logs. Users of other frameworks must ensure their logs are compatible with the required format.

### Multi-Task benchmark

```text
benchmark_root/
├── walker_walk/
│   ├── SAC/
│   │   ├── alg_config.json
│   │   ├── env_config.json
│   │   ├── train_config.json
│   │   ├── 0/data/eval.csv
│   │   ├── 1/data/eval.csv
│   │   └── ...
│   ├── TD3/
│   └── YourAlgorithm/
├── walker/
│   ├── SACYourAlgorithm/
│   ├── TD3/
│   └── YourAlgorithm/
└── ...
```

### Single Task

```text
walker_walk/
├── SAC/
├── TD3/
└── YourAlgorithm/
```

The algorithm name is read from:

```json
{
  "algorithm": "SAC"
}
```

inside `alg_config.json`. The algorithm directory name itself is not used as the canonical algorithm identity.

!!! warning "Comparison labels must be unique"
    In a standard benchmark, algorithm names must be unique within each task. When multiple runs use the same algorithm, supply one or more `--comparison-parameter` paths so each experimental condition receives a distinct comparison label.

### Required files

Every algorithm run requires:

- `alg_config.json`
- `env_config.json`
- `train_config.json`
- one or more numeric seed directories
- `data/eval.csv` inside every numeric seed directory

Non-numeric directories are ignored when seeds are discovered.

### Required `eval.csv` columns

At minimum:

| total_steps | episode_reward |
|------------:|---------------:|
| 0           | 12.4           |
| 10000       | 42.8           |
| 20000       | 69.8           |

`total_steps` identifies each evaluation step point throughout training. The selected evaluation metric, `episode_reward` by default, contains one value per evaluation episode.

The tool first averages all evaluation episodes recorded at the same step, creating one evaluation curve per seed.

!!! important "Complete and matched evaluation grids"
    All algorithms and seeds within a task must use the same complete evaluation-step grid and the same number of evaluation episodes at every step. The tool will raise an error if the evaluation grids are not compatible between algorithms in any given task.

### Configuration compatibility checks

The tool verifies selected environment, training and algorithm settings before comparison. These include the environment identity, observation settings, evaluation cadence, number of evaluation episodes and maximum training steps.

This prevents statistics from being generated for runs that are not experimentally comparable - in order to generate statistically valid results, the compared runs must be compatible in their configuration.

### Seeds

By default, seed IDs and counts must match across algorithms for a given task. Use `--allow-unmatched-seeds` only when the experiments were intentionally designed as independent samples or this isn't a concern for the analysis.

```text
SAC:  [0, 1, 2, 3, 4]
TD3:  [0, 1, 2, 3, 4]
Novel:[0, 1, 2, 3, 4]
```

!!! note "Why matched seeds are still preferred"
    Probability of improvement compares all runs from one algorithm with all runs from another and does not use one-to-one seed identities. Matched seeds are nevertheless good experimental practice and support the supplementary paired tests retained in raw task outputs.

## Running an Analysis

### Standard benchmark command

```bash
cares-rl-stats benchmark_root \
    --output results \
    --reference-comparison YourAlgorithm
```

This analyses every discovered task, writes task-level outputs, performs cross-task analysis when more than one task is present and generates `statistical_report.pdf`.

The value supplied to `--reference-comparison` must exactly match a discovered comparison label. In a standard benchmark this is normally the algorithm name.

### Ablations and Parameter Sweeps

When several run directories use the same algorithm, select the configuration values that distinguish them:

```bash
cares-rl-stats benchmark_root \
    --output results \
    --comparison-parameter alg_config.plasticity.replacement_rate \
    --reference-comparison "PPO [replacement_rate=1e-05]"
```

Multiple distinguishing parameters may be supplied:

```bash
cares-rl-stats benchmark_root \
    --output results \
    --comparison-parameter alg_config.plasticity.replacement_rate \
    --comparison-parameter alg_config.plasticity.maturity_threshold
```

The dotted path begins with one of the loaded configuration names:

- `alg_config`
- `env_config`
- `train_config`

Do not add a comparison parameter when it has the same value for every run and does not define a meaningful experimental condition.

### Single-task command

```bash
cares-rl-stats walker_walk --output results
```

A single-task run produces task-level files and a task-focused PDF. It does not produce a cross-task benchmark directory.

### Multiple evaluation metrics

```bash
cares-rl-stats benchmark_root \
    --output results \
    --metric episode_reward:higher \
    --metric episode_length:lower
```

Each metric is analysed independently. Metrics with different units or meanings are never compared or pooled.

!!! warning "Set direction correctly"
    Use `higher` when larger values are better and `lower` when smaller values are better. Direction changes the orientation of probability of improvement and Cliff's delta.

### Choosing the primary curve summary

The tool calculates three AUC-based summaries for every seed:

- `auc`: the full evaluation curve
- `early_window_auc`: the initial fraction of training
- `final_window_auc`: the final fraction of training

Choose which summary drives the publication-focused tables and report:

```bash
cares-rl-stats benchmark_root \
    --output results \
    --primary-performance-metric final_window_auc
```

!!! note
    The other curve summaries are still retained in the raw CSV outputs.

### Reproducibility controls

```bash
cares-rl-stats benchmark_root \
    --output results \
    --bootstrap-samples 10000 \
    --bootstrap-confidence 0.95 \
    --random-seed 0 \
    --figure-dpi 300
```

### Optional output controls

```bash
#### Skip statistical figures
cares-rl-stats benchmark_root --output results --no-statistical-figures

#### Skip only the PDF report
cares-rl-stats benchmark_root --output results --no-pdf-report
```

!!! tip "Recommended paper workflow"
    Specify `--reference-comparison` for the final paper analysis. The PDF and benchmark folder will then include a dedicated reference-comparison analysis against every baseline.



## Generated Outputs

A multi-task analysis produces:

```text
results/
├── statistical_report.pdf
├── tasks/
│   ├── walker_walk/
│   │   ├── seed_metrics.csv
│   │   ├── algorithm_summary.csv
│   │   ├── pairwise_comparisons.csv
│   │   ├── task_summary_all_metrics.csv
│   │   ├── task_performance.csv
│   │   ├── task_performance.tex
│   │   ├── pairwise_statistics.csv
│   │   ├── pairwise_statistics.tex
│   │   ├── task_summary.csv
│   │   ├── task_summary.tex
│   │   └── methodology.md
│   └── ...
└── benchmark/
    ├── task_algorithm_summaries.csv
    ├── task_pairwise_comparisons.csv
    ├── task_seed_metrics.csv
    ├── task_superiority.csv
    ├── benchmark_summary.csv
    ├── cross_task_pairwise.csv
    ├── friedman_tests.csv
    ├── nemenyi_posthoc.csv
    ├── benchmark_performance.csv
    ├── benchmark_performance.tex
    ├── cross_task_pairwise_publication.csv
    ├── cross_task_pairwise_publication.tex
    ├── friedman.csv
    ├── friedman.tex
    ├── nemenyi.csv
    ├── nemenyi.tex
    ├── reference_comparison.csv
    ├── reference_comparison.tex
    ├── methodology.md
    └── figures/
```

`reference_comparison.csv` and `reference_comparison.tex` are created only when `--reference-comparison` is supplied and a cross-task benchmark exists.

### Raw versus publication-focused outputs

| Output type | Purpose |
|---|---|
| Raw CSVs | Complete intermediate and supplementary results for reproducibility and further analysis. |
| Publication CSVs | Reduced tables containing the most useful paper-facing measures. |
| LaTeX | Directly reusable table source. |
| Figures | Cross-task probability and rank visualisations. |
| PDF | Guided summary of the entire analysis. |

!!! important "Keep the raw outputs"
    Do not retain only the PDF. The raw CSV files record the exact seed-level summaries, task-level estimates and supplementary tests needed to audit or regenerate a result.

### Single-task outputs

For one task, the `tasks/<task>/` directory and PDF are produced. The `benchmark/` directory is omitted because there is no cross-task aggregation.

## Reading the PDF Report

The PDF is a guided summary, not a replacement for the raw data.

The reference-comparison section appears only when `--reference-comparison` is supplied.

### Recommended reading order

1. Read the **Executive Summary** for the factual headline.
2. Inspect the **Benchmark Overview** for roster-wide performance and consistency.
3. Use the **Reference Comparison** section for direct claims about the selected reference comparison.
4. Inspect **Per-Task Results** to identify strengths, weaknesses and unstable environments.
5. Use **Friedman/Nemenyi** only as supplementary consistency evidence.
6. Confirm configuration and bootstrap details in **Methodology**.

!!! warning "The top-ranked method is not automatically decisively better"
    Check confidence intervals and direct pairwise probabilities. A small difference in mean superiority or average rank may be practically inconclusive.

### Factual reporting style

A supported statement is:

> YourAlgorithm achieved a mean superiority of 0.79 with a 95% confidence interval of [0.73, 0.84] and ranked first on 8 of 12 tasks.

An unsupported causal interpretation is:

> YourAlgorithm performs better because it explores more effectively.

The statistics describe **what happened**, not **why it happened**.



## Interpretation Workflow

Use this sequence for every final analysis.

### 1. Confirm validity

Check that the run completed without validation warnings or incompatible configurations.

### 2. Inspect per-task performance

Look for:

- low or negative-return failures;
- unusually wide IQM intervals;
- tasks where the reference comparison clearly underperforms;
- sensitivity to early versus final performance.

### 3. Inspect direct reference comparisons

For each baseline, record:

- probability the reference comparison is better;
- confidence interval;
- W-T-L task record;
- tasks responsible for losses.

### 4. Inspect the benchmark summary

Use mean superiority to describe roster-wide performance and rank statistics to describe consistency.

### 5. Inspect supplementary rank tests

Use Friedman/Nemenyi to check whether the rank pattern is strong enough to support a benchmark-wide rank distinction.

### 6. Write a bounded conclusion

A strong conclusion contains:

- benchmark scope;
- effect estimate;
- uncertainty;
- consistency;
- known weaknesses.

!!! example "Balanced conclusion"
    Across 12 tasks and 10 seeds per algorithm, YourAlgorithm achieved the highest mean superiority and an estimated 0.82 probability of improvement over SAC. The 95% interval [0.74, 0.89] favoured YourAlgorithm, although it ranked below SAC on one task and had wider uncertainty on two additional tasks.



## Experimental Design

### Recommended scale

| Stage | Tasks | Seeds per algorithm per task | Purpose |
|---|---:|---:|---|
| Development | 3–5 | 3–5 | Debugging, screening and rapid iteration. |
| Minimum final evidence | 6–10 | 5 | Defensible only with strong effects and honest uncertainty. |
| Solid benchmark | 10–15 | 10 | Recommended default for a serious benchmark claim. |
| Very strong benchmark | 15–30+ | 10–20 | Broad, high-confidence evidence or modest expected effects. |

!!! tip "Recommended default"
    For a general-purpose RL benchmark, target **10–15 tasks with 10 seeds per algorithm per task**.

### Tasks and seeds answer different questions

- More **seeds** improve precision under training randomness.
- More **tasks** improve evidence that the result generalises across environments.

After reaching roughly ten seeds, adding relevant tasks is often more informative than repeatedly increasing seeds on a narrow benchmark.

### Matched seeds

Use the same seed IDs for all algorithms when feasible. This makes the experimental design balanced and enables the supplementary paired comparisons.

!!! warning "Do not selectively add seeds"
    Set the minimum seed count and any extension rule before inspecting whether a result is nearly favourable. Adding seeds only to rescue an inconclusive comparison introduces researcher degrees of freedom.

### Baselines

A convincing benchmark should include:

- the closest methodological baseline;
- strong current algorithms for the environment family;
- any method explicitly claimed to be improved upon;
- a simpler reference where useful.

### Hyperparameter fairness

Report whether each algorithm used published defaults, equal tuning budgets or independently optimised configurations.

!!! warning
    Do not tune repeatedly on the same final benchmark tasks and then present those tasks as untouched evidence of generalisation.



## Common Mistakes

### Averaging raw rewards across tasks

Reward scales differ, so the result is dominated by large-magnitude environments.

!!! danger
    Do not calculate a pooled benchmark IQM, mean or median from raw task returns.

Use probability of improvement, mean superiority or ranks for cross-task summaries.

### Reporting only the best seed

The best seed measures selection luck, not expected algorithm performance.

### Treating five seeds as universally sufficient

Five seeds may be usable for expensive experiments, but it is a floor rather than a general target. Inspect interval width and instability.

### Treating a p-value as effect magnitude

A small p-value does not imply a large or practically important improvement.

### Calling overlapping intervals "no difference"

An interval overlapping 0.5 means the current data do not resolve the pairwise direction at the selected confidence level. It does not prove equality.

### Calling non-overlapping algorithm IQM intervals a pairwise test

Separate one-sample intervals are not identical to a direct interval on the difference or probability of improvement. Use the direct pairwise output.

### Ignoring metric direction

A runtime or error metric may be lower-is-better. Declaring it as `higher` reverses every pairwise interpretation.

### Over-interpreting rank

An average-rank improvement from 2.0 to 1.8 may represent tiny or large practical differences. Consult probability and task-level magnitudes.

### Treating mean superiority as universal

It changes when the algorithm roster changes. Report the baselines included and use direct comparisons for claims about a named method.



## Command-Line Reference

```text
cares-rl-stats TASKS --output OUTPUT [options]
```

| Option | Default | Meaning |
|---|---:|---|
| `TASKS` | required | Single task directory or directory containing tasks. |
| `--output` | required | Destination root. |
| `--metric COLUMN[:higher|lower]` | `episode_reward:higher` | Evaluation metric; repeatable. |
| `--allow-unmatched-seeds` | off | Permit explicitly independent seed samples. |
| `--early-window-fraction` | `0.25` | Fraction of training used for early AUC. |
| `--final-window-fraction` | `0.10` | Fraction of training used for final AUC. |
| `--bootstrap-samples` | `10000` | Number of bootstrap replicates. |
| `--bootstrap-confidence` | `0.95` | Confidence level. |
| `--random-seed` | `0` | Bootstrap random seed. |
| `--significance-level` | `0.05` | Alpha for supplementary tests and Holm flags. |
| `--no-statistical-figures` | off | Disable cross-task figures. |
| `--no-pdf-report` | off | Disable the guided PDF. |
| `--reference-comparison` | unset | Algorithm to feature as the reference comparison. |
| `--figure-dpi` | `300` | Figure resolution; minimum accepted value is 72. |
| `--primary-performance-metric` | `auc` | One of `auc`, `early_window_auc`, `final_window_auc`. |

!!! note
    `--primary-performance-metric` changes the publication-focused tables and report. All three curve summaries remain in the raw outputs.

--8<-- "include/links.md"