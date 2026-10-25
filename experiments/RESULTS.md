# Results summary (for the final report)

All numbers come from `results/*.csv` (errors are relative L2, ×10⁻³ unless stated).
CI = 95% cluster bootstrap over the 12 evaluation trajectories (10,000 resamples).
"±" = standard deviation over 3 training seeds. Buckets are total-variation tertiles of the
128-point input (20 examples each), fixed once and reused on every grid.

---

## Part 1: the promised checks (matched training, 3 seeds, second measure, start-time control)

Both models now train on the **same 38 trajectories** (10 held out for validation), keep
their **best-validation checkpoint**, and are run with **3 seeds**. The numbers barely move
from M3, so the M3 result was not an artefact of the unmatched setup or of one lucky seed.

**Main table** (`table_main.csv`), mean error over seeds:

| Bucket | Model | e₁₂₈ | e₂₅₆ | e₅₁₂ | Growth e₅₁₂−e₁₂₈ |
|---|---|---|---|---|---|
| Smooth | Persistence | 4.23 | 4.23 | 4.23 | – |
| | CNN | 0.25 ± 0.05 | 2.20 ± 0.04 | 3.24 ± 0.04 | 2.99 |
| | FNO | 0.43 ± 0.06 | 0.52 ± 0.12 | 0.55 ± 0.15 | 0.12 |
| Medium | Persistence | 19.15 | 19.15 | 19.15 | – |
| | CNN | 0.74 ± 0.06 | 10.28 ± 0.04 | 14.82 ± 0.02 | 14.07 |
| | FNO | 1.15 ± 0.04 | 1.36 ± 0.22 | 1.43 ± 0.28 | 0.28 |
| Rough | Persistence | 74.22 | 74.26 | 74.27 | – |
| | CNN | 7.93 ± 0.52 | 39.59 ± 0.22 | 57.22 ± 0.03 | 49.30 |
| | FNO | 10.56 ± 0.20 | 10.70 ± 0.20 | 10.72 ± 0.20 | 0.15 |

**Rough-minus-smooth growth contrast** (`table_contrast.csv`):

| Roughness measure | Model | C₂₅₆ [CI] | C₅₁₂ [CI] | seed sd of C₅₁₂ |
|---|---|---|---|---|
| Total variation | CNN | 29.7 [22.7, 37.4] | 46.3 [34.9, 58.7] | 0.48 |
| | FNO | 0.05 [−0.18, 0.29] | 0.03 [−0.26, 0.33] | 0.12 |
| Sharpness | CNN | 19.7 [11.6, 30.8] | 31.3 [18.4, 48.9] | 0.49 |
| | FNO | 0.05 [−0.17, 0.30] | 0.05 [−0.21, 0.34] | 0.16 |
| Max slope | CNN | 29.9 [22.2, 39.2] | 46.1 [33.2, 61.2] | 0.52 |
| | FNO | 0.14 [−0.09, 0.33] | 0.16 [−0.13, 0.40] | 0.02 |

The three roughness measures, all computed on the 128-point input (details in notebook 01, section 2):
1. **Total variation** (primary, pre-registered in M2): sum of all neighbour-to-neighbour jumps.
2. **Sharpness** (main robustness check): RMS slope ÷ RMS fluctuation. Independent of the wave's height,
   roughly a typical wavenumber (≈ 2πk for a sine with k cycles). Measuring it on the 256- or 512-point
   input gives exactly the same buckets.
3. **Max slope** (the check promised in M2): the largest single jump.

- **Positive control works:** the CNN contrast is large and its interval excludes 0, under both measures.
- **FNO:** every interval contains 0, under both measures and at both grids. The FNO's upper bound
  (0.33) is about 1% of the CNN's lower bound (34.9).
- **Sharpness (main robustness check):** ranks the inputs quite differently from total variation
  (Spearman 0.40; 60% same bucket), because it ignores the wave's height, and it is less tied to start time
  (Spearman with start time −0.60 vs −0.82 for total variation). The conclusion does not change.
- **Max slope (promised in M2):** ranks the inputs almost the same way as total variation
  (Spearman 0.90; 73% of inputs land in the same bucket), so it is a weaker check. The conclusion does not change.
- **Accuracy floor:** the FNO is below persistence in every bucket on every grid (6.9× to 16.7× better).
  The CNN falls to 1.3× better than persistence at 512 points (`table_admissible.csv`).

**Start-time control** (`table_starttime.csv`, `table_bucket_by_time.csv`).
Roughness and start time are confounded: Spearman(TV, start time) = −0.82, and every rough input
comes from the first two start times. We regress each example's growth e₅₁₂−e₁₂₈ on
z(log TV), with and without one intercept per start time. With the intercepts, the slope is the
roughness effect *within* a start time.

| Model | Slope, no control | Slope, with start-time intercepts |
|---|---|---|
| CNN | 16.5 [10.8, 56.9] | 7.0 [5.1, 49.9] |
| FNO | 0.08 [−0.10, 0.16] | 0.11 [0.05, 0.59] |

(units: ×10⁻³ growth per standard deviation of log TV)

- CNN: about half of the apparent roughness effect is shared with start time, but a clear
  roughness effect remains within start times.
- FNO: within start times there is a **small positive** roughness association (interval excludes 0).
  It is about 65× smaller than the CNN's. Report it as a real but very small effect, not as zero.
  The intervals are skewed (upper ends are far from the estimate), so don't over-interpret their exact width.

**Decision (M2 rule):** H1 is **not supported** for the FNO. The contrast intervals contain 0 at
256 and 512 points under both measures, and are about 1% of the CNN's. One nuance: after controlling
for start time, the FNO shows a statistically detectable but very small roughness association.

---

## Part 2: the competing explanation ("nothing to fail on")

**The problem.** At a 128-point training grid, the finer grids add almost no new content. The
error of the 512-point target cut off at wavenumber 64 (the *truncation reference*, i.e. the error
of a perfect model that predicts nothing above the training Nyquist) is only 0.11 / 0.44 / 2.47
(smooth / medium / rough). So a flat FNO error could mean "invariant" or "nothing to fail on".

**The test.** Train both models at **64** and **32** points (every 2nd / 4th point of the same
training data; the FNO keeps N/2 modes, the same rule as the main model), then evaluate at every
finer grid up to 512 with the same buckets. Now the finer grids contain a lot that the training
grid cannot see (`table_unresolved.csv`):

| Training grid | Truncation reference: smooth / medium / rough |
|---|---|
| 128 | 0.11 / 0.44 / 2.47 |
| 64 | 0.16 / 1.10 / 6.30 |
| 32 | 0.41 / 9.23 / 25.35 |

**Result 1: the pre-registered contrast** (`table_trainres.csv`, growth e₅₁₂ − e_train):

| Model | Train grid | Growth smooth / medium / rough | Contrast C₅₁₂ [CI] |
|---|---|---|---|
| CNN | 128 | 2.99 / 14.07 / 49.30 | 46.3 [34.9, 58.7] |
| CNN | 64 | 3.52 / 16.26 / 55.79 | 52.3 [40.0, 65.3] |
| CNN | 32 | 3.69 / 16.80 / 52.10 | 48.4 [36.7, 61.4] |
| FNO | 128 | 0.12 / 0.28 / 0.15 | 0.03 [−0.26, 0.33] |
| FNO | 64 | 0.44 / 0.63 / −0.61 | −1.04 [−3.02, 0.37] |
| FNO | 32 | 0.48 / 0.65 / −5.06 | −5.54 [−16.21, 0.40] |

- The CNN control detects a roughness effect at every training grid.
- Even when the unseen content in the rough bucket is 10× larger (32-point training), the FNO's
  contrast **never becomes positive**. The rough bucket's error actually *falls* on finer grids.

**Result 2: why, from the frequency split of the 512-point error** (`table_bandsplit.csv`).
The FNO's error above the training Nyquist, divided by the truncation reference
(1 = the model adds nothing at those scales, < 1 = it recovers some real fine-scale content,
> 1 = it adds wrong fine-scale content):

| FNO trained at | Smooth | Medium | Rough |
|---|---|---|---|
| 128 | 1.96 | 1.36 | 0.77 |
| 64 | 3.48 | 1.09 | 0.74 |
| 32 | 1.62 | 0.44 | 0.51 |

- **Rough inputs:** the FNO *recovers* part of the fine-scale content it never saw in training
  (23% at 128, 26% at 64, 49% at 32). So it uses the extra information on finer grids, which
  is evidence of genuine transfer and not just "nothing to fail on".
- **Smooth inputs:** the FNO *adds* spurious fine-scale content (1.6× to 3.5× the reference).
  This is where its small error growth comes from. It is model-side, not caused by roughness.
- **About half of that growth sits at the domain edges** (notebook 04, figure H). For smooth inputs, half of the FNO's
  squared error lies in the 6% of the domain nearest the edges (x < 0.03 or x > 0.97), and removing those points
  roughly halves its growth in every bucket (0.12 → 0.07, 0.28 → 0.12, 0.16 → 0.07 ×10⁻³). A likely cause: the FNO's
  position input x jumps from 1 back to 0 at the edge while the field is periodic. The rest of the spurious content is
  consistent with aliasing from the point-wise nonlinearities (the motivation for CNO; Bartolucci et al. 2023).
  A periodic position encoding (sin 2πx, cos 2πx) is the obvious fix to test.
- Caveat for the write-up: at coarse training grids, e_train is measured on samples that under-resolve
  rough inputs, so the *growth* measure partly reflects a change in what is being measured. The
  frequency split does not have this problem, which is why it is the stronger evidence here.

**Conclusion for the report.** The FNO's cross-resolution error growth does not depend on input
roughness. The positive control shows the protocol can detect such an effect, and the second
roughness measure and start-time control leave the conclusion essentially unchanged. The
training-resolution experiment rules out the main alternative explanation: when the finer grids hold
substantial content the training grid cannot represent, the FNO still shows no roughness-dependent
growth, and it recovers part of that content on rough inputs. The small non-invariance it does show
comes from spurious fine-scale content on smooth inputs.

---

## Paste-ready sentences

- *Robustness (Sec. 4/5):* "With both models trained on the same 38 trajectories, best-validation
  selection and three seeds, the FNO's rough-minus-smooth growth contrast at 512 points is
  0.03×10⁻³ (95% CI [−0.26, 0.33]×10⁻³; seed sd 0.12×10⁻³), against 46.3×10⁻³ ([34.9, 58.7]) for the CNN.
  The conclusion is unchanged under two alternative roughness measures: an amplitude-normalised sharpness
  (RMS slope over RMS fluctuation; 0.05×10⁻³, CI [−0.21, 0.34]), which ranks the inputs quite differently from
  total variation (Spearman 0.40), and the steepest single jump (0.16×10⁻³, CI [−0.13, 0.40])."
- *Confound (Sec. 5/6):* "Controlling for start time with one intercept per start time, the FNO's
  growth increases by 0.11×10⁻³ per standard deviation of log total variation (95% CI [0.05, 0.59]),
  about 65 times less than the CNN's 7.0×10⁻³."
- *Advanced experiment (Sec. 6):* "Training at 32 points raises the rough bucket's unresolved content
  tenfold (truncation reference 25.4×10⁻³ vs 2.5×10⁻³), yet the FNO's contrast stays non-positive
  (−5.5×10⁻³, CI [−16.2, 0.4]). Its error above the training Nyquist is 0.51 of the truncation
  reference on rough inputs, so it recovers about half of the unseen content, while on smooth inputs it
  adds spurious fine-scale content (1.6–3.5× the reference)."

## Notes

- The FNO used here is trained on 38 trajectories with best-validation selection (the M3 FNO used all 48
  and its final epoch). Its errors are almost the same (0.0041 / 0.0044 / 0.0045 mean at 128/256/512, vs
  0.0040 / 0.0043 / 0.0043 for the old checkpoint), so the matched models can replace the M3 numbers.
- Start-time labels: the kit's `context` column labels the five start times 0, 0.2, 0.4, 0.6, 0.8.
  Fitting the PDE to the data gives Δt = 0.01 per stored step with diffusion ν/π, which would put the
  start times (stored indices 0/40/80/120/160) at t = 0, 0.4, 0.8, 1.2, 1.6. Check this before quoting
  physical start times in the report; "the first two start times" is safe either way.
