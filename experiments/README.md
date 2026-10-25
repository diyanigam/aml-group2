# Final-report experiments

Everything in the final report's results comes from this folder.

```bash
pip install -r requirements.txt   # from the repository root
cd experiments
jupyter notebook
```

Run the notebooks in order. Each one explains what it does and why, step by step:

| Notebook | What it does | Run time (laptop CPU) |
|---|---|---|
| `01_data_and_roughness.ipynb` | Data checks, the three roughness measures and buckets, start-time confound, how much unseen detail each grid adds, and a check of the exact Burgers equation (ν/π, Δt = 0.01) | < 1 min |
| `02a_train_cnn.ipynb` | The CNN explained, then its 9 training runs with **live convergence plots**, an overview of all curves, and a sandbox for trying hyperparameters | ~8 min |
| `02b_train_fno.ipynb` | The same for the FNO, including why it does not use early stopping | ~45 min |
| `03_evaluation.ipynb` | Every table (saved to `results/`) and the two report figures (saved to `figures/`), with each statistic explained | ~20 s |
| `04_figures.ipynb` | A visual tour: the plots from `bucked_based_eval.ipynb` adapted to the final models, plus new figures (edge effect, error spectra, zoom on a front). Nothing is saved | ~1 min |

The training notebooks skip runs whose checkpoint already exists, and show the saved curve instead. To retrain, delete
the matching files in `checkpoints/`. Retraining reproduces the saved models exactly: all 18 came out bit-for-bit
identical to the first training run (on CPU; another machine may differ in the last digits).

**Which roughness measure makes the buckets?** Total variation of the 128-point input, for every result. Sharpness and
max slope only appear as robustness checks of the main test (notebook 03, section 5). In `04_figures` you can switch
the buckets to either of them with one setting.

The same pipeline also exists as plain scripts: `run_training.py` (training), `evaluate.py` (evaluation)
and `common.py` (shared code). They give the same numbers. Use whichever you prefer.

## What was run

**Part 1: matched training and the checks promised in M2/M3**

- CNN and FNO, both trained at 128 points on the **same 38 training / 10 validation
  trajectories** (the CNN notebook's split, `np.random.seed(0)`), each with **3 seeds**
  (0, 1, 2). Both keep their **best-validation checkpoint**.
  Each model keeps its own optimiser and schedule from the group notebooks:
  - CNN: Adam 1e-3, batch 32, halve LR after 20 epochs without improvement, early stop after 40 (max 600).
  - FNO: Adam 1e-3 with weight decay 1e-4, cosine schedule, 500 epochs, batch 16, 64 modes, width 64.
- Roughness buckets: periodic total variation of the 128-point input, split into tertiles
  (20 examples each). The buckets are fixed once and reused on every grid.
- Main robustness check: the same analysis with **sharpness** (RMS slope / RMS fluctuation of the 128-point input),
  which ignores the wave's height and is less tied to start time.
- Check promised in M2: the same analysis with **max slope** `max_i |u_{i+1} - u_i|` as the roughness measure.
- **Start-time control**: regress the growth on z(log TV) with one intercept per start time.
  The slope is then the roughness effect *within* start times.
- Uncertainty: 95% cluster bootstrap over the 12 evaluation trajectories (10,000 resamples).
  Per-example errors are averaged over seeds first. Across-seed sd is reported separately.

**Part 2: training-resolution experiment (the competing explanation)**

At 128 points, less than 1.4e-6 of the input energy lies above the training grid's
Nyquist wavenumber. So the finer grids add almost nothing the model has not seen,
and a flat FNO error could mean "invariant" or "nothing to fail on".
To tell these apart, both models are also trained at **64 and 32 points**
(every 2nd / 4th point of the training data, so no new data is needed).
They are then evaluated at every finer grid up to 512 points, with the same buckets.
The FNO keeps N/2 modes at an N-point training grid (32 at 64, 16 at 32).
That is every wavenumber the training grid resolves except Nyquist, the same rule as the main model.

All evaluation grids are exact subsamples of the 512-point evaluation data
(checked in `common.load_eval`: `eval_128 == eval_512[:, ::4]`).

## Outputs

| File | Content |
|---|---|
| `results/table_main.csv` | Part 1: mean error per bucket at 128/256/512 points (mean ± seed sd), ratio, growth |
| `results/table_contrast.csv` | Part 1: rough-minus-smooth growth contrast with 95% CI, under all three roughness measures |
| `results/table_starttime.csv` | Part 1: roughness effect with and without start-time fixed effects |
| `results/table_bucket_by_time.csv` | Part 1: how many inputs of each bucket come from each start time |
| `results/table_admissible.csv` | Part 1: is each model below persistence on every grid in every bucket? |
| `results/table_unresolved.csv` | Part 2: input energy above the training Nyquist, and the truncation reference error |
| `results/table_trainres.csv` | Part 2: growth to 512 points per bucket and contrast, for training at 32/64/128 |
| `results/table_bandsplit.csv` | Part 2: 512-point error split at the training Nyquist, against the truncation reference |
| `results/per_example_errors.npz` | Every per-example error, for any further analysis |
| `results/summary.json` | The key numbers in machine-readable form |
| `figures/fig_growth_main.*` | Part 1 figure (growth per bucket, 95% CI) |
| `figures/fig_trainres.*` | Part 2 figure: (a) unseen content, (b) FNO growth, (c) FNO fine-scale error vs truncation reference |
| `RESULTS.md` | Plain-language summary of every result, with paste-ready sentences for the report |
| `checkpoints/` | All 18 trained models, their settings and their training history (convergence curves) |

Errors are relative L2 per example. Growth is `e_N - e_train`. The contrast is mean growth
(rough) minus mean growth (smooth). Tables report values ×1e-3.
