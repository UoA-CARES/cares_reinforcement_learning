--8<-- "include/glossary.md"

#  Frequently Asked Questions

##  Why is IQM preferred over the mean?

RL runs can contain unusually poor or strong seeds. IQM reduces their influence while retaining more distributional information than the median.

##  Why is IQM not divided by episode length?

The evaluation metric is analysed as logged. Episodic return should remain episodic return unless the research question explicitly defines a per-step metric. Dividing after the fact changes the quantity being measured.

##  Why is AUC not normalised by the number of training steps?

Within-task validation requires a common training horizon and evaluation grid, so raw AUC is directly comparable. Normalising would rescale every method on that task by the same constant and would not change ordering or probability of improvement.

##  Why not average IQM across tasks?

Different tasks have different reward scales. A cross-task raw average has no stable interpretation.

##  Does probability of improvement pair matching seeds?

No. It compares every run from one algorithm with every run from the other. Seed matching remains valuable for experiment balance and supplementary paired tests.

##  Is 0.60 a good probability of improvement?

It means the candidate is estimated to win 60% of random run pairs. Whether that is compelling depends on the confidence interval, benchmark scope, task failures and practical cost.

##  What if the confidence interval includes 0.5?

The current data do not clearly resolve which algorithm is more likely to perform better. Report the estimate and interval honestly rather than declaring equality.

##  Why do task wins and probability disagree?

Task wins use observed ranks. Probability of improvement uses all seed-level distributional comparisons. An algorithm can rank first narrowly on many tasks or win fewer tasks with stronger run-level separation.

##  Why are Friedman and Nemenyi supplementary?

They operate on ranks and discard the underlying run distributions and performance magnitude. Direct probability of improvement gives a more interpretable pairwise effect estimate.

##  When is the critical-difference figure missing?

It is generated only when the Friedman test is significant for that evaluation-metric/performance-summary group.

##  Can algorithms have different seeds?

Only with `--allow-unmatched-seeds`. This changes task-level tests to independent Mann–Whitney comparisons and should reflect an intentionally independent experimental design.

##  How many tasks and seeds should I use?

For a solid general benchmark, target 10–15 tasks and 10 seeds per algorithm per task. Increase seeds when variability is high or expected effects are modest.

## Why isn't statistical significance enough?

Statistical significance alone does not describe the magnitude or practical importance of an improvement. The CARES RL Statistical Tool therefore emphasises effect estimates (IQM, Probability of Improvement, Mean Superiority) together with confidence intervals, using hypothesis tests only as supplementary evidence.

--8<-- "include/links.md"