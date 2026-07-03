"""Per-timestep timing at 20 objects for each surface size, with a significance test.

Re-runs the 20-ball case for each surface size (capturing every repetition's raw
time), reports the average wall-clock time per simulation timestep, and tests
whether the differences between surface sizes are statistically significant
(one-way ANOVA + pairwise Welch t-tests).
"""

import time
import numpy as np
from scipy import stats

from benchmark import (
    SURFACE_SIZES,
    TIMESTEPS,
    make_config,
    run_simulation_quiet,
)

NBALL = 20
REPETITIONS = 10


def collect_raw_times():
    # Warm-up to avoid cold-start bias.
    run_simulation_quiet(make_config(*SURFACE_SIZES[0], nball=NBALL))

    raw = {}
    for gx, gy in SURFACE_SIZES:
        label = f"{gx}x{gy}"
        config = make_config(gx, gy, NBALL)
        times = []
        for _ in range(REPETITIONS):
            start = time.perf_counter()
            run_simulation_quiet(config)
            times.append(time.perf_counter() - start)
        raw[label] = np.array(times)
        print(f"{label}: collected {REPETITIONS} runs")
    return raw


def main():
    raw = collect_raw_times()

    # Per-timestep time in milliseconds, per repetition.
    per_step_ms = {label: (t / TIMESTEPS) * 1000.0 for label, t in raw.items()}

    print("\nAverage time per timestep (20 objects):")
    print(f"{'surface':>8}  {'mean (ms)':>10}  {'std (ms)':>9}  {'95% CI (ms)':>18}")
    for label, vals in per_step_ms.items():
        mean = vals.mean()
        sd = vals.std(ddof=1)
        sem = sd / np.sqrt(len(vals))
        ci = stats.t.interval(0.95, len(vals) - 1, loc=mean, scale=sem)
        print(f"{label:>8}  {mean:>10.4f}  {sd:>9.4f}  [{ci[0]:.4f}, {ci[1]:.4f}]")

    groups = list(per_step_ms.values())
    labels = list(per_step_ms.keys())

    # Overall test: is any surface size different?
    f_stat, p_anova = stats.f_oneway(*groups)
    print(f"\nOne-way ANOVA:  F={f_stat:.3f}  p={p_anova:.3e}")
    print(f"  -> {'significant' if p_anova < 0.05 else 'not significant'} at alpha=0.05")

    # Pairwise Welch t-tests (unequal variances), Bonferroni-corrected.
    print("\nPairwise Welch t-tests (Bonferroni-corrected):")
    pairs = [(i, j) for i in range(len(labels)) for j in range(i + 1, len(labels))]
    m = len(pairs)
    for i, j in pairs:
        t_stat, p = stats.ttest_ind(groups[i], groups[j], equal_var=False)
        p_corr = min(p * m, 1.0)
        verdict = "significant" if p_corr < 0.05 else "n.s."
        print(f"  {labels[i]:>6} vs {labels[j]:>6}:  t={t_stat:+.3f}  "
              f"p={p:.3e}  p_bonf={p_corr:.3e}  ({verdict})")


if __name__ == "__main__":
    main()
