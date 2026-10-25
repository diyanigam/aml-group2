# Is Resolution Invariance Uniform?

**Roughness-stratified error growth of a Fourier Neural Operator on 1D Burgers**

Group 2 · Advanced Machine Learning · MSc Artificial Intelligence, University of Groningen · Project PK-2026-011

Diya Nigam · Tejasvi Kanteti · Vaishnav Vishwanath · Ved Bhatt

---

## In short

Neural operators such as the Fourier Neural Operator (FNO) can be trained on one grid and applied to finer ones.
We test whether this transfer works **equally well for smooth and rough inputs**.

- An FNO trained at 128 points on 1D viscous Burgers is evaluated at 256 and 512 points.
- Test inputs are split into thirds by roughness (total variation).
- A grid-locked CNN serves as a model known to fail, and persistence (predict no change) as an accuracy floor.

**Result:** rough inputs are harder to predict, but **not harder to transfer**. The FNO's rough-minus-smooth
difference in error growth at 512 points is 0.03 × 10⁻³ (95% interval [−0.26, 0.32]), against 46.3 × 10⁻³ for the CNN.
This holds across three roughness measures, three seeds, a start-time control, and coarser training grids
(64 and 32 points) where the finer grids contain detail the model never saw.
The FNO's small remaining error growth comes from wrong fine detail, about half of it at the domain edges.

## Where to start

**Everything behind the final report is in [`experiments/`](experiments/).** Start with its
[README](experiments/README.md); the results in prose are in [`experiments/RESULTS.md`](experiments/RESULTS.md).

```bash
pip install -r requirements.txt
cd experiments
jupyter notebook
```

Run the notebooks in order:

| Notebook | What it does | Time (laptop CPU) |
|---|---|---|
| `01_data_and_roughness.ipynb` | Data checks, the three roughness measures, the start-time confound, how much unseen detail each grid adds, and a check of the exact equation | < 1 min |
| `02a_train_cnn.ipynb` | The grid-locked CNN: 9 training runs with live convergence plots and a hyperparameter sandbox | ~8 min |
| `02b_train_fno.ipynb` | The FNO: 9 training runs with live convergence plots and a hyperparameter sandbox | ~45 min |
| `03_evaluation.ipynb` | Every table and both report figures (saved to `results/` and `figures/`) | ~20 s |
| `04_figures.ipynb` | A visual tour: example predictions, errors along the domain, error spectra | ~1 min |

The trained models are included in `experiments/checkpoints/`, so notebooks 01, 03 and 04 run without training.
The training notebooks skip models that already exist; delete a checkpoint to retrain it. Retraining on CPU
reproduces the saved models exactly.

## Repository layout

| Path | Contents |
|---|---|
| [`experiments/`](experiments/) | **The final project**: notebooks, trained models, result tables, figures |
| [`data/`](data/) | The PDEBench 1D Burgers subset (ν = 0.01): training data at 128 points, test data at 128, 256 and 512 points |
| [`cnn_baseline/`](cnn_baseline/) | Development of the grid-locked CNN, including the configuration ablation (`ablation.ipynb`) cited in the report |
| [`FNO/`](FNO/), [`models/FNO_submission/`](models/FNO_submission/), `FNO.ipynb` | Development of the FNO (earlier versions) |
| `bucked_based_eval.ipynb`, `cnn_full_pipeline.ipynb` | Earlier bucket-based evaluation, where the sharpness measure was introduced |
| [`DATA_LICENCE.md`](DATA_LICENCE.md) | Data source and licence |

The folders other than `experiments/` and `data/` document how the models were developed; the final results do
not depend on them.

## Setup at a glance

| | Grid-locked CNN | FNO |
|---|---|---|
| Role | positive control (known to break on finer grids) | model under test |
| Architecture | 4 convolutions (kernel 3, width 64) + linear output, residual | lifting, 4 Fourier layers (width 64), projection |
| Fourier modes | n/a | half the training grid (64 at 128 points) |
| Training | same 38 / 10 trajectory split, best-validation weights, 3 seeds | same |
| Training grids | 128 (main), 64 and 32 | same |

## Team

| Member | Contributions |
|---|---|
| Diya Nigam | Research question, hypothesis and problem formulation; data analysis; total-variation roughness measure and buckets; evaluation design and decision rule; final experiments (matched training, seeds, robustness checks, start-time control, training-resolution experiment, frequency and boundary analysis); review of both models; report |
| Tejasvi Kanteti | Grid-locked CNN: implementation, training and configuration selection |
| Vaishnav Vishwanath | Sharpness roughness measure; bucket-based evaluation of the models |
| Ved Bhatt | FNO: implementation and training |

## Use of AI tools

AI tools were used to reorganise this repository: collecting the final code into `experiments/`, putting the
notebooks in the order they should be run, and documenting them to match the report. AI assistance was also used
when writing the final experiment notebooks. The models (grid-locked CNN and FNO), the roughness measures and the
original evaluation were written by the group members (under `archive/`), and all results were run and checked by the group.

## Data

The Burgers subset is derived from PDEBench (Takamoto et al., 2022, https://doi.org/10.18419/darus-2986),
licensed CC BY 4.0. See [`DATA_LICENCE.md`](DATA_LICENCE.md).
