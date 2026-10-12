# FNO submission

This folder contains the final 1D Fourier Neural Operator (FNO), its evaluation results, and plotting notebooks. The model predicts the next Burgers solution field from the current field and is evaluated at 128, 256, and 512 grid points.

## Files

| File | Purpose |
| --- | --- |
| `FNO_final_model.ipynb` | Defines and trains the final FNO on 240 training examples from all 48 training simulations. Settings: 64 Fourier modes, width 64, 4 layers, 500 epochs, seed 0. The final epoch weights are saved. |
| `fno_modes64_all48_seed0.pt` | Trained model checkpoint, including model weights, configuration, and the 500-epoch training-loss history. |
| `eval_fno.ipynb` | Loads the checkpoint, evaluates the same model at all three resolutions, and saves per-example FNO and identity-baseline errors. |
| `fno_errors.npz` | Evaluation errors: `err_128`, `err_256`, `err_512` for the FNO and `id_128`, `id_256`, `id_512` for the identity baseline. Each array contains 60 example errors. |
| `plot_eval.ipynb` | Plots the mean FNO and identity errors by resolution and the FNO error-growth ratios. Reads `fno_errors.npz`. |
| `plot_history.ipynb` | Plots the training-loss history stored in the checkpoint, including a close-up of the last 100 epochs. There is no validation curve because the final model was trained on all training examples. |
| `plot_outputs.ipynb` | Shows input, true next-step solution, and FNO prediction for one evaluation example at each resolution. Set `EXAMPLE_INDEX` to select another example. |
| `plots_eval.png` | Saved evaluation summary plot. |
| `plots_training.png` | Saved training-history plot. |
| `plots_outputs.png` | Saved example prediction plot. |

The evaluation data files (`data/eval_128.npz`, `data/eval_256.npz`, and `data/eval_512.npz`) are in the project-level `data/` directory, outside this folder.

## Run the notebooks

Use the project Python environment with NumPy, Matplotlib, and PyTorch installed. From the project root, run `eval_fno.ipynb`; its paths to `data/` and the checkpoint are relative to the project root. It writes `fno_errors.npz` to the current working directory.

Run `plot_eval.ipynb` with this folder as the notebook working directory so it reads this folder's `fno_errors.npz`. Run `plot_history.ipynb` with this folder as the working directory so it can load the checkpoint. Both save their PNG next to the notebook.

`plot_outputs.ipynb` locates the project root whether its working directory is the project root or this folder. It loads the evaluation data and checkpoint and saves `plots_outputs.png` in this folder.

To regenerate `fno_errors.npz` in this folder after running evaluation from the project root, save/copy the generated archive into `FNO_submission/` before running `plot_eval.ipynb`.

## Recorded evaluation summary

The checked-in evaluation archive has 60 examples per resolution. Its mean relative L² errors are:

| Resolution | FNO | Identity baseline |
| ---: | ---: | ---: |
| 128 | 0.00404 | 0.03253 |
| 256 | 0.00427 | 0.03255 |
| 512 | 0.00433 | 0.03255 |

These are aggregate evaluation results; `plot_outputs.ipynb` lets you inspect individual prediction curves.
