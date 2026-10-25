"""Train every model used in the final report.

  Part 1 (matched training): CNN and FNO at 128 points, seeds 0, 1, 2.
  Part 2 (training-resolution experiment): CNN and FNO at 64 and 32 points, seeds 0, 1, 2.

All runs use the same 38 training / 10 validation trajectories (fixed_split) and
keep the best-validation checkpoint. The FNO keeps N/2 Fourier modes at an
N-point training grid (64 at 128, 32 at 64, 16 at 32): every wavenumber the
training grid resolves except Nyquist, the same rule as the main model.

Usage:  python run_training.py          (skips runs whose checkpoint exists)
"""
import json, os, time
import torch
from common import CKPT, fixed_split, load_train, train_cnn, train_fno

torch.set_num_threads(2)
SEEDS = [0, 1, 2]
tr, va = fixed_split()

for n_train in (128, 64, 32):
    X, Y = load_train(n_train)
    for model_name in ("cnn", "fno"):
        for seed in SEEDS:
            tag = f"{model_name}_train{n_train}_seed{seed}"
            path = os.path.join(CKPT, tag + ".pt")
            if os.path.exists(path):
                continue
            t0 = time.time()
            if model_name == "cnn":
                model, info = train_cnn(X, Y, tr, va, seed)
                config = dict(hidden=64, kernel_size=3, num_layers=4)
            else:
                model, info = train_fno(X, Y, tr, va, seed, modes=n_train // 2)
                config = dict(modes=n_train // 2, width=64, n_layers=4)
            info.update(model=model_name, n_train=n_train, seed=seed, config=config,
                        seconds=round(time.time() - t0, 1))
            torch.save({"state": model.state_dict(), "config": config, "info": info}, path)
            with open(os.path.join(CKPT, tag + ".json"), "w") as f:
                json.dump(info, f, indent=1)
            print(tag, info, flush=True)
