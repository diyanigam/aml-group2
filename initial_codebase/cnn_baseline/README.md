# CNN Baseline (Grid-Locked CNN)

Negative-control baseline for the FNO resolution study on 1D Burgers.

## Results

| Resolution | CNN rel-L2 | Identity baseline | Improvement |
|-----------|------------|-------------------|-------------|
| 128 | 0.00281 | 0.03253 | 11.57× |
| 256 | 0.01729 | 0.03255 | 1.88× |
| 512 | 0.02508 | 0.03255 | 1.30× |

**Error growth**: 6.15× (128→256), 8.92× (128→512).

## Architecture

4-layer Conv1d, kernel 3, hidden 64, circular padding, residual connection.
Chosen by ablation (see `ablation.ipynb`) over depth 2/4/6/8, kernel 3/5/7,
hidden 32/64/128. Picked smallest model within noise of best validation loss.

## Training

Trajectory-level validation split (38 train / 10 val trajectories).
Early stopping (patience 40), reduce-LR-on-plateau, best-checkpoint saving.
Best val rel-L2 = 0.00217 (13× improvement over identity).

## Files

- `train_final.ipynb` — trains the model, saves `models/cnn_128.pt`
- `eval_cnn.ipynb` — evaluates on 128/256/512, saves per-example errors
- `ablation.ipynb` — hyperparameter ablation over 8 configs
- `plot_history.ipynb` — learning/convergence/precision curves
- `plot_eval.ipynb` — error-vs-resolution and growth-ratio plots
