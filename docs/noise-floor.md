# The noise floor: how a run decides a gain is real

`gitm run` keeps a candidate only when its measured gain beats the noise of the machine it ran on. This page gives the arithmetic behind that rule, with a worked example from a real H200 run. [`loop.md`](loop.md) describes where the rule sits in the loop. The code is `NoiseFloor` and `mean_noise_factor` in `gitm/optimizer/apply.py`.

## What is measured and what is set

| Part | Decided by | Default |
|---|---|---|
| The floor | measured by the run, every run | none |
| A/A rebuilds and paired rounds per candidate | `--validation N` / `GITM_VALIDATION` | 2 |
| The smallest gain worth keeping | `GITM_MIN_EFFECT` | 1% |
| Margin over the noise of the measured mean | code (`NoiseFloor.threshold`) | 2× |
| Threshold before any A/A sample exists | code (`_THRESHOLD_WITHOUT_FLOOR`) | 2% |

## 1. An A/A sample

The baseline configuration is built, measured, rebuilt and measured again. Each fresh build of the same configuration gives one sample:

$$
d_i = \frac{T_i}{T_{i-1}} - 1
$$

where $T$ is the run's throughput in tokens per second. On a perfectly stable machine every $d_i$ would be 0, so whatever is left over is noise.

Samples come from two places:
- **Calibration**, before the first candidate: the running engine is measured, then N + 1 rebuilds. The first pair is a warm-up and is left out. On Kimi K2.5 the first engine after the opening capture decoded 25% slower than every later build.
- **Every later fresh build of the same baseline**: after a rollback, or between rounds.

A measurement that wasn't the first on its build is skipped (`skipped_not_fresh`). Repeated decodes on one engine (`GITM_AB_REPS`) never count, because they miss the build-to-build part of the noise.

## 2. The floor is the root mean square

$$
\text{floor} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} d_i^{\,2}}
$$

```python
math.sqrt(sum(d * d for d in self.samples) / len(self.samples))
```

An A/A sample has a true mean of zero, so the RMS is its standard deviation, with no degree of freedom spent estimating a mean that is already known. It is also defined from a single sample.

A plain average would let positive and negative samples cancel and report almost no noise. Squaring counts both directions and weights the larger swings more, which errs toward a stricter bar.

## 3. Averaging rounds: the factor

A candidate gets up to N paired rounds. A round's ratio compares one candidate build with one baseline build, so a single round has the floor's spread: factor 1.

Averaging helps less than $1/\sqrt{N}$ suggests, because neighbouring rounds share a build. Serial validation alternates and reuses the running engine: A₁ B₁, then B₁ A₂, then A₂ B₂, and so on. The measured gain is the mean of the rounds' ratios, minus 1. With two rounds:

$$
\bar r = \tfrac{1}{2}\left(\frac{B_1}{A_1} + \frac{B_1}{A_2}\right) - 1
$$

To see its noise, write each build's throughput as its true value times $(1 + \varepsilon)$, where $\varepsilon$ is that build's relative noise: $A_k = a(1+\varepsilon_{A_k})$ and $B_1 = b(1+\varepsilon_{B_1})$. Noise of a fraction of a percent makes products of $\varepsilon$ negligible, so to first order

$$
\frac{B_1}{A_k} \approx \frac{b}{a}\,\big(1 + \varepsilon_{B_1} - \varepsilon_{A_k}\big),
\qquad
\bar r \approx \Big(\frac{b}{a} - 1\Big) + \varepsilon_{B_1} - \tfrac{1}{2}\varepsilon_{A_1} - \tfrac{1}{2}\varepsilon_{A_2}
$$

(with $b/a$ close to 1, the true effect's factor on the noise terms is dropped too). So build $B_1$'s noise has weight 1 and each baseline build's has weight ½. If each build's relative noise has variance $\sigma_b^2$, an A/A sample is the difference of two builds, so $\text{floor}^2 = 2\sigma_b^2$. The spread of the mean is then

$$
\sigma_{\bar r} = \sqrt{\sum_j w_j^2\,\sigma_b^2} = \text{floor}\cdot\sqrt{\tfrac{1}{2}\sum_j w_j^2} = \text{floor}\cdot\sqrt{\tfrac{1}{2}\left(1 + \tfrac14 + \tfrac14\right)} = 0.866\cdot\text{floor}
$$

`mean_noise_factor` adds up the weights for any N:

| Rounds N | Factor | $1/\sqrt{N}$, if rounds were independent |
|---|---|---|
| 1 | 1.000 | 1.000 |
| 2 | 0.866 | 0.707 |
| 3 | 0.745 | 0.577 |
| 4 | 0.661 | 0.500 |
| 6 | 0.553 | 0.408 |

Rounds that rebuild nothing (a hot-swapped knob, or two engines resident at once) re-measure the same two builds, so their factor stays 1.

## 4. The threshold

$$
\text{threshold} = \max\big(\,2 \times \text{floor} \times \text{factor},\ \ \text{min\_effect}\,\big)
$$

The candidate is kept when the mean of its rounds' B/A ratios, minus 1, is at least

$$
\max(\text{threshold},\ \text{noise\_band}), \qquad \text{noise\_band} = \frac{s_A + s_B}{T_A}
$$

where $s_A$ and $s_B$ are the standard deviations of the baseline's and the candidate's repeated decodes on one engine (`GITM_AB_REPS`) and $T_A$ is the baseline throughput. With the default `GITM_AB_REPS=1` there is one decode per engine, so the noise band is 0 and the threshold alone decides. A gain equal to the bar is kept. Until the first A/A sample exists, the threshold is $\max(2\%, \text{min\_effect})$.

The 2× is two standard deviations of the measured mean. If the noise is roughly normal and the floor well estimated, a candidate with no real effect clears the noise term about 2.3% of the time (one-sided). The minimum effect is a separate, practical bar: a gain smaller than it isn't worth changing a deployment for, however clean the measurement.

## Worked example: H200 job 100

Qwen3-8B on one H200, 256 Mooncake conversations, `--validation 2`. The run collected 20 A/A samples (`noise_floor.json`):

| # | $d_i$ | $d_i^2$ (×10⁻⁶) | | # | $d_i$ | $d_i^2$ (×10⁻⁶) |
|---|---|---|---|---|---|---|
| 1 | +0.069% | 0.482 | | 11 | −0.019% | 0.036 |
| 2 | +0.049% | 0.238 | | 12 | +0.122% | 1.490 |
| 3 | +0.050% | 0.254 | | 13 | −0.158% | 2.501 |
| 4 | −0.074% | 0.551 | | 14 | +0.236% | 5.555 |
| 5 | −0.076% | 0.574 | | 15 | −0.288% | 8.274 |
| 6 | +0.140% | 1.957 | | 16 | −0.064% | 0.404 |
| 7 | −0.194% | 3.751 | | 17 | +0.026% | 0.068 |
| 8 | +0.115% | 1.314 | | 18 | +0.067% | 0.445 |
| 9 | −0.070% | 0.493 | | 19 | +0.011% | 0.011 |
| 10 | +0.045% | 0.199 | | 20 | +0.009% | 0.007 |

$$
\sum d_i^2 = 28.603\times10^{-6},\qquad \text{floor} = \sqrt{\frac{28.603\times10^{-6}}{20}} = 0.120\%
$$

For comparison, the plain mean of the same samples is −0.0002%, and the mean of their absolute values is 0.094%.

$$
\text{threshold} = \max(2 \times 0.120\% \times 0.866,\ 1\%) = \max(0.21\%,\ 1\%) = 1\%
$$

`kv_cache_dtype_fp8` measured +11.8% and +12.1% in its two rounds, a mean of **+11.92%**, so it was kept. In the same run, `gpu_memory_utilization` ×1.10 measured +0.30%: above the noise term, below the minimum effect, so it was rolled back.

**On a quiet machine the minimum effect decides, not the noise.** The floor on this H200 was 0.12%; on an RTX 4050 laptop it is about 0.45%, and one MI355X run reached 6% after a single outlier sample. Only a floor above about 0.58% makes the noise term the binding one at two rounds ($2 \times 0.58\% \times 0.866 \approx 1\%$).

## What it doesn't cover

- **Many candidates per run.** Each candidate is tested on its own. A run that tests 10–20 candidates has a higher chance that one of them clears the noise term by luck than any single test does.
- **A small sample.** Early in a run the floor comes from 2–3 samples and can be far off. One outlier moved an MI355X run's floor to 6%.
- **The `gitm` server** decides differently: a paired t-interval at α = 0.05 whose low end must clear the minimum effect, with absolute latency targets.
