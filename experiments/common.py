"""Shared code for the final-report experiments (Group 2, PK-2026-011).

Models are copied unchanged from the group notebooks:
  - GridLockedCNN  : cnn_baseline/train_final.ipynb
  - FNO1d          : models/FNO_submission/FNO_final_model.ipynb
Training settings are the ones in those notebooks. The only change is the one
promised in M2/M3: both models are trained on the SAME 38 trajectories and
both keep their best-validation checkpoint (same selection rule).
"""
import os
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
CKPT = os.path.join(HERE, "checkpoints")
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
for d in (CKPT, RES, FIG):
    os.makedirs(d, exist_ok=True)

N_TRAJ, PER_TRAJ = 48, 5          # training set: 48 trajectories x 5 start times
N_EVAL_TRAJ = 12                  # evaluation set: 12 trajectories x 5 start times
GRIDS = [32, 64, 128, 256, 512]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_train(n_grid=128):
    """Training pairs on an n_grid-point grid (n_grid in 32, 64, 128).
    Coarser grids are exact subsamples of the 128-point training data,
    exactly as the kit builds 128 from the 1024-point source (every k-th point)."""
    d = np.load(os.path.join(DATA, "train.npz"))
    s = 128 // n_grid
    return d["field"][:, ::s].astype(np.float32), d["target"][:, ::s].astype(np.float32)


def load_eval():
    """Evaluation pairs on every grid. All grids are exact subsamples of the
    512-point evaluation data (checked: eval_128 == eval_512[:, ::4])."""
    d = np.load(os.path.join(DATA, "eval_512.npz"))
    f, t = d["field"].astype(np.float32), d["target"].astype(np.float32)
    out = {N: (f[:, ::512 // N], t[:, ::512 // N]) for N in GRIDS}
    d128 = np.load(os.path.join(DATA, "eval_128.npz"))
    assert np.array_equal(out[128][0], d128["field"]) and np.array_equal(out[128][1], d128["target"])
    start_label = d128["context"][:, 0]           # kit label (0, .2, .4, .6, .8)
    traj = np.arange(len(f)) // PER_TRAJ           # rows are trajectory-major
    return out, start_label, traj


def fixed_split():
    """Trajectory-level split used by the CNN notebook (np.random.seed(0)):
    10 validation trajectories, 38 training trajectories. Used for EVERY run."""
    rng = np.random.RandomState(0)
    perm = rng.permutation(N_TRAJ)
    val_traj, train_traj = perm[:10], perm[10:]
    rows = lambda ts: np.concatenate([np.arange(t * PER_TRAJ, t * PER_TRAJ + PER_TRAJ) for t in ts])
    return rows(train_traj), rows(val_traj)


# ----------------------------------------------------------------------------
# roughness measures (computed on the 128-point evaluation inputs, fixed for all grids)
# ----------------------------------------------------------------------------
def total_variation(u):
    return np.abs(np.roll(u, -1, axis=1) - u).sum(axis=1)


def max_slope(u):
    return np.abs(np.roll(u, -1, axis=1) - u).max(axis=1)


def sharpness(u):
    """RMS slope / RMS fluctuation (central differences, periodic); used on the 128-point input."""
    u = u.astype(np.float64)
    slope = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 / u.shape[1])
    fluct = u - u.mean(axis=1, keepdims=True)
    return np.sqrt((slope ** 2).mean(axis=1)) / np.sqrt((fluct ** 2).mean(axis=1))


def tertile_buckets(score):
    """0 = smooth, 1 = medium, 2 = rough; 20 examples each."""
    order = np.argsort(score, kind="stable")
    b = np.empty(len(score), dtype=int)
    for k, idx in enumerate(np.array_split(order, 3)):
        b[idx] = k
    return b


# ----------------------------------------------------------------------------
# models (verbatim from the group notebooks)
# ----------------------------------------------------------------------------
class GridLockedCNN(nn.Module):
    def __init__(self, hidden=64, kernel_size=3, num_layers=4):
        super().__init__()
        layers = [nn.Conv1d(1, hidden, kernel_size, padding=kernel_size // 2,
                            padding_mode="circular"), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size // 2,
                                 padding_mode="circular"), nn.ReLU()]
        layers.append(nn.Conv1d(hidden, 1, kernel_size, padding=kernel_size // 2,
                                padding_mode="circular"))
        self.net = nn.Sequential(*layers)

    def forward(self, u):                 # u: (batch, N)
        u = u[:, None, :]
        return (u + self.net(u))[:, 0, :]


def make_input(u):
    batch, n_points = u.shape
    x = torch.linspace(0, 1, n_points + 1, device=u.device)[:-1]
    return torch.stack([u, x.expand(batch, n_points)], dim=-1)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))

    def forward(self, x):
        batch, _, n_points = x.shape
        x_freq = torch.fft.rfft(x)
        out_freq = torch.zeros(batch, self.weights.shape[1], n_points // 2 + 1,
                               dtype=torch.cfloat, device=x.device)
        k = min(self.modes, n_points // 2 + 1)
        out_freq[:, :, :k] = torch.einsum("bix,iox->box", x_freq[:, :, :k], self.weights[:, :, :k])
        return torch.fft.irfft(out_freq, n=n_points)


class FNO1d(nn.Module):
    def __init__(self, modes=64, width=64, n_layers=4):
        super().__init__()
        self.lift = nn.Linear(2, width)
        self.spec = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.w = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.proj = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, u):                 # u: (batch, N) -- wraps make_input
        x = self.lift(make_input(u)).permute(0, 2, 1)
        for fourier, local in zip(self.spec, self.w):
            x = nn.functional.gelu(fourier(x) + local(x))
        return self.proj(x.permute(0, 2, 1)).squeeze(-1)


def rel_l2_torch(pred, target):
    return (torch.linalg.vector_norm(pred - target, dim=-1) / torch.linalg.vector_norm(target, dim=-1)).mean()


def rel_l2(pred, target):
    return np.linalg.norm(pred - target, axis=1) / np.linalg.norm(target, axis=1)


# ----------------------------------------------------------------------------
# training (settings from the group notebooks; selection = best validation)
# ----------------------------------------------------------------------------
def train_cnn(X, Y, tr, va, seed, max_epochs=600):
    set_seed(seed)
    model = GridLockedCNN(hidden=64, kernel_size=3, num_layers=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt, Yt, Xv, Yv = map(torch.tensor, (X[tr], Y[tr], X[va], Y[va]))
    g = torch.Generator().manual_seed(seed)
    best, best_state, wait, lr_wait, ep = np.inf, None, 0, 0, 0
    for ep in range(max_epochs):
        model.train()
        order = torch.randperm(len(Xt), generator=g)
        for s in range(0, len(Xt), 32):
            i = order[s:s + 32]
            loss = rel_l2_torch(model(Xt[i]), Yt[i])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = rel_l2_torch(model(Xv), Yv).item()
        if vl < best - 1e-6:
            best, wait, lr_wait = vl, 0, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1; lr_wait += 1
        if wait >= 40:
            break
        if lr_wait >= 20:
            for pg in opt.param_groups:
                pg["lr"] *= 0.5
            lr_wait = 0
    model.load_state_dict(best_state)
    return model, dict(best_val=best, epochs=ep + 1)


def train_fno(X, Y, tr, va, seed, modes, epochs=500):
    set_seed(seed)
    model = FNO1d(modes=modes, width=64, n_layers=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt, Yt, Xv, Yv = map(torch.tensor, (X[tr], Y[tr], X[va], Y[va]))
    g = torch.Generator().manual_seed(seed)
    best, best_state = np.inf, None
    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(Xt), generator=g)
        for s in range(0, len(Xt), 16):
            i = order[s:s + 16]
            loss = rel_l2_torch(model(Xt[i]), Yt[i])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = rel_l2_torch(model(Xv), Yv).item()
        if vl < best:
            best = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, dict(best_val=best, epochs=epochs)


@torch.no_grad()
def predict(model, u):
    model.eval()
    return model(torch.tensor(u)).numpy()


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------
def cluster_bootstrap(stat_fn, traj, n_boot=10000, seed=0):
    """95% interval of stat_fn(row_indices), resampling whole evaluation trajectories."""
    rng = np.random.default_rng(seed)
    ids = np.unique(traj)
    rows_of = {t: np.flatnonzero(traj == t) for t in ids}
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        vals[b] = stat_fn(np.concatenate([rows_of[t] for t in pick]))
    return np.percentile(vals, [2.5, 97.5])


def contrast(growth, bucket, rows=None):
    """mean growth (rough) - mean growth (smooth) over the given rows."""
    if rows is None:
        rows = np.arange(len(growth))
    g, b = growth[rows], bucket[rows]
    return g[b == 2].mean() - g[b == 0].mean()
