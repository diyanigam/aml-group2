"""Evaluate every checkpoint from run_training.py and write the report tables/figures.

Outputs (results/ and figures/):
  per_example_errors.npz      every model x training grid x seed x eval grid, 60 errors each
  table_main.csv              Part 1: mean error per TV bucket at 128/256/512 (mean over seeds +- seed sd)
  table_contrast.csv          Part 1: rough-minus-smooth growth contrast, TV / sharpness / max-slope buckets, 95% CI
  table_starttime.csv         Part 1: start-time control (regression with start-time fixed effects)
  table_bucket_by_time.csv    Part 1: bucket composition by start time
  table_admissible.csv        Part 1: FNO / CNN below persistence on every grid in every bucket?
  table_trainres.csv          Part 2: growth per bucket and contrast when training at 32 / 64 / 128 points
  table_unresolved.csv        Part 2: input energy above the training Nyquist and truncation reference
  fig_growth_main.(png|pdf)   Part 1 figure
  fig_trainres.(png|pdf)      Part 2 figure
"""
import glob, json, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import (CKPT, RES, FIG, GRIDS, FNO1d, GridLockedCNN, load_eval, predict, rel_l2,
                    total_variation, max_slope, sharpness, tertile_buckets, cluster_bootstrap, contrast)

NB = int(os.environ.get("NB", 10000))
BUCKET_NAMES = ["smooth", "medium", "rough"]
evalset, start_label, traj = load_eval()
u128 = evalset[128][0]
tv, ms = total_variation(u128), max_slope(u128)
B_TV, B_MS = tertile_buckets(tv), tertile_buckets(ms)
B_SH = tertile_buckets(sharpness(u128))
persist = {N: rel_l2(*evalset[N]) for N in GRIDS}

# ---------------------------------------------------------------- predictions
def band_split(err, y, kcut):
    """Split the error at one grid into the part below / at-or-above wavenumber kcut,
    both relative to ||y|| (so low**2 + high**2 = total**2)."""
    Ef, Yn = np.fft.rfft(err.astype(np.float64), axis=1), np.linalg.norm(y, axis=1)
    w = np.ones(Ef.shape[1]); w[1:] = 2                     # rfft -> full-spectrum energy weights
    if err.shape[1] % 2 == 0:
        w[-1] = 1
    P = w * np.abs(Ef) ** 2 / err.shape[1]
    return np.sqrt(P[:, :kcut].sum(1)) / Yn, np.sqrt(P[:, kcut:].sum(1)) / Yn

errs = {}   # (model, n_train, seed) -> {N: (60,)}
hband = {}  # (model, n_train, seed) -> (low, high) error at 512, split at the training Nyquist
for path in sorted(glob.glob(os.path.join(CKPT, "*.pt"))):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    info = ck["info"]
    model = (GridLockedCNN if info["model"] == "cnn" else FNO1d)(**ck["config"])
    model.load_state_dict(ck["state"])
    key = (info["model"], info["n_train"], info["seed"])
    preds = {N: predict(model, evalset[N][0]) for N in GRIDS if N >= info["n_train"]}
    errs[key] = {N: rel_l2(p, evalset[N][1]) for N, p in preds.items()}
    hband[key] = band_split(preds[512] - evalset[512][1], evalset[512][1], info["n_train"] // 2)
np.savez(os.path.join(RES, "per_example_errors.npz"),
         **{f"{m}_train{n}_seed{s}_eval{N}": e for (m, n, s), d in errs.items() for N, e in d.items()},
         **{f"persistence_eval{N}": persist[N] for N in GRIDS}, tv=tv, max_slope=max_slope(u128),
         start_label=start_label, traj=traj)

seeds_of = lambda m, n: sorted(s for (mm, nn, s) in errs if mm == m and nn == n)
def seed_mean(m, n, N):
    return np.mean([errs[(m, n, s)][N] for s in seeds_of(m, n)], axis=0)
def growth(m, n, N, seed=None):
    if seed is None:
        return seed_mean(m, n, N) - seed_mean(m, n, n)
    return errs[(m, n, seed)][N] - errs[(m, n, seed)][n]

def fmt(x, scale=1e3, d=2):
    return f"{x * scale:.{d}f}"

summary = {}
models = [m for m in ("cnn", "fno") if seeds_of(m, 128)]

# ---------------------------------------------------------------- Part 1: main table
rows = ["bucket,model,e128,e128_sd,e256,e256_sd,e512,e512_sd,ratio_512,growth_512,n_seeds (errors x1e-3)"]
for k, b in enumerate(BUCKET_NAMES):
    sel = B_TV == k
    p = [persist[N][sel].mean() for N in (128, 256, 512)]
    rows.append(f"{b},persistence,{fmt(p[0])},,{fmt(p[1])},,{fmt(p[2])},,,,")
    for m in models:
        cells = []
        per_seed = {N: [errs[(m, 128, s)][N][sel].mean() for s in seeds_of(m, 128)] for N in (128, 256, 512)}
        for N in (128, 256, 512):
            cells += [fmt(np.mean(per_seed[N])), fmt(np.std(per_seed[N], ddof=1) if len(per_seed[N]) > 1 else 0)]
        e128, e512 = np.mean(per_seed[128]), np.mean(per_seed[512])
        rows.append(f"{b},{m.upper()},{','.join(cells)},{e512 / e128:.2f},{fmt(e512 - e128)},{len(seeds_of(m, 128))}")
open(os.path.join(RES, "table_main.csv"), "w").write("\n".join(rows) + "\n")

# ---------------------------------------------------------------- Part 1: contrasts (TV and max-slope)
rows = ["measure,model,N,contrast_x1e-3,ci_lo,ci_hi,seed_mean,seed_sd,growth_smooth,growth_medium,growth_rough"]
for measure, B in (("total_variation", B_TV), ("sharpness", B_SH), ("max_slope", B_MS)):
    for m in models:
        for N in (256, 512):
            g = growth(m, 128, N)
            c = contrast(g, B)
            lo, hi = cluster_bootstrap(lambda r: contrast(g, B, r), traj, NB)
            cs = [contrast(growth(m, 128, N, s), B) for s in seeds_of(m, 128)]
            gb = [g[B == k].mean() for k in range(3)]
            rows.append(f"{measure},{m.upper()},{N},{fmt(c, d=3)},{fmt(lo, d=3)},{fmt(hi, d=3)},"
                        f"{fmt(np.mean(cs), d=3)},{fmt(np.std(cs, ddof=1) if len(cs) > 1 else 0, d=3)},"
                        + ",".join(fmt(x, d=3) for x in gb))
            summary[f"contrast_{measure}_{m}_{N}"] = [c, lo, hi]
open(os.path.join(RES, "table_contrast.csv"), "w").write("\n".join(rows) + "\n")

# agreement of the two roughness measures
agree = float((B_TV == B_MS).mean())
rank = lambda x: np.argsort(np.argsort(x))
spearman_tv_ms = float(np.corrcoef(rank(tv), rank(ms))[0, 1])
spearman_tv_t = float(np.corrcoef(rank(tv), rank(start_label + 1e-6 * np.arange(60)))[0, 1])
summary.update(bucket_agreement_tv_vs_maxslope=agree, spearman_tv_maxslope=spearman_tv_ms,
               spearman_tv_starttime=spearman_tv_t)

# ---------------------------------------------------------------- Part 1: start-time control
# growth_512 = a_t (one intercept per start time) + beta * z(log TV) + noise.
# beta is the roughness effect WITHIN start times; compare with the model without a_t.
times = np.unique(start_label)
z = (np.log(tv) - np.log(tv).mean()) / np.log(tv).std()
def beta(g, rows_, fixed_effects):
    cols = [z[rows_]] + ([(start_label[rows_] == t).astype(float) for t in times] if fixed_effects
                         else [np.ones(len(rows_))])
    A = np.stack(cols, 1)
    coef, *_ = np.linalg.lstsq(A, g[rows_], rcond=None)
    return coef[0]
rows = ["model,N,controls,beta_x1e-3_per_sd_logTV,ci_lo,ci_hi"]
for m in models:
    for N in (256, 512):
        g = growth(m, 128, N)
        for fe in (False, True):
            bval = beta(g, np.arange(60), fe)
            lo, hi = cluster_bootstrap(lambda r: beta(g, r, fe), traj, NB)
            rows.append(f"{m.upper()},{N},{'start-time fixed effects' if fe else 'none'},"
                        f"{fmt(bval, d=3)},{fmt(lo, d=3)},{fmt(hi, d=3)}")
            summary[f"beta_{m}_{N}_{'fe' if fe else 'nofe'}"] = [bval, lo, hi]
open(os.path.join(RES, "table_starttime.csv"), "w").write("\n".join(rows) + "\n")

rows = ["start_label,smooth,medium,rough"]
for t in times:
    rows.append(f"{t:.1f}," + ",".join(str(int(((start_label == t) & (B_TV == k)).sum())) for k in range(3)))
open(os.path.join(RES, "table_bucket_by_time.csv"), "w").write("\n".join(rows) + "\n")

# ---------------------------------------------------------------- Part 1: admissibility (accuracy floor)
rows = ["model,bucket,N,mean_error_x1e-3,persistence_x1e-3,times_better,below_persistence"]
for m in models:
    for k, b in enumerate(BUCKET_NAMES):
        for N in (128, 256, 512):
            e, p = seed_mean(m, 128, N)[B_TV == k].mean(), persist[N][B_TV == k].mean()
            rows.append(f"{m.upper()},{b},{N},{fmt(e)},{fmt(p)},{p / e:.1f},{e < p}")
open(os.path.join(RES, "table_admissible.csv"), "w").write("\n".join(rows) + "\n")

# ---------------------------------------------------------------- Part 2: training resolution
def band_limit(u, kmax):
    U = np.fft.rfft(u.astype(np.float64), axis=1); U[:, kmax:] = 0
    return np.fft.irfft(U, n=u.shape[1], axis=1)
x512, y512 = (a.astype(np.float64) for a in evalset[512])
E = np.abs(np.fft.rfft(x512, axis=1)) ** 2
rows = ["n_train,bucket,median_input_energy_above_nyquist,truncation_reference_err_x1e-3"]
for n in (32, 64, 128):
    frac = E[:, n // 2:].sum(1) / E.sum(1)
    trunc = rel_l2(band_limit(y512, n // 2), y512)
    for k, b in enumerate(BUCKET_NAMES):
        rows.append(f"{n},{b},{np.median(frac[B_TV == k]):.2e},{fmt(trunc[B_TV == k].mean(), d=3)}")
        summary[f"unresolved_{n}_{b}"] = float(np.median(frac[B_TV == k]))
        summary[f"trunc_{n}_{b}"] = float(trunc[B_TV == k].mean())
open(os.path.join(RES, "table_unresolved.csv"), "w").write("\n".join(rows) + "\n")

rows = ["model,n_train,n_seeds,e_train_smooth,e_train_rough,growth512_smooth,growth512_medium,growth512_rough,"
        "contrast512_x1e-3,ci_lo,ci_hi,seed_sd,below_persistence_all_grids_and_buckets"]
trainres = {}
for m in ("cnn", "fno"):
    for n in (32, 64, 128):
        if not seeds_of(m, n):
            continue
        g = growth(m, n, 512)
        c = contrast(g, B_TV)
        lo, hi = cluster_bootstrap(lambda r: contrast(g, B_TV, r), traj, NB)
        cs = [contrast(growth(m, n, 512, s), B_TV) for s in seeds_of(m, n)]
        et = seed_mean(m, n, n)
        ok = all(seed_mean(m, n, N)[B_TV == k].mean() < persist[N][B_TV == k].mean()
                 for N in GRIDS if N >= n for k in range(3))
        gb = [g[B_TV == k].mean() for k in range(3)]
        trainres[(m, n)] = dict(gb=gb, gb_ci=[cluster_bootstrap(lambda r, k=k: g[r][B_TV[r] == k].mean(), traj, 2000)
                                              for k in range(3)], c=(c, lo, hi))
        rows.append(f"{m.upper()},{n},{len(seeds_of(m, n))},{fmt(et[B_TV == 0].mean())},{fmt(et[B_TV == 2].mean())},"
                    + ",".join(fmt(x, d=3) for x in gb)
                    + f",{fmt(c, d=3)},{fmt(lo, d=3)},{fmt(hi, d=3)},"
                    f"{fmt(np.std(cs, ddof=1) if len(cs) > 1 else 0, d=3)},{ok}")
        summary[f"trainres_{m}_{n}"] = dict(contrast=[c, lo, hi], growth_by_bucket=gb, below_persistence=ok)
open(os.path.join(RES, "table_trainres.csv"), "w").write("\n".join(rows) + "\n")

# band-split of the 512-point error at the training Nyquist k = n_train/2.
# high-band error vs the truncation reference (= high-band error of a prediction with NO content
# above k): recovered < 0 means the model adds wrong fine-scale content, 1 means it gets all of it.
rows = ["model,n_train,bucket,e512_x1e-3,lowband_err_x1e-3,highband_err_x1e-3,truncation_ref_x1e-3,"
        "fraction_of_unseen_content_recovered"]
for m in ("cnn", "fno"):
    for n in (32, 64, 128):
        if not seeds_of(m, n):
            continue
        lo_b = np.mean([hband[(m, n, s)][0] for s in seeds_of(m, n)], axis=0)
        hi_b = np.mean([hband[(m, n, s)][1] for s in seeds_of(m, n)], axis=0)
        e5 = seed_mean(m, n, 512)
        for k, b in enumerate(BUCKET_NAMES):
            sel = B_TV == k
            tr_ref = summary[f"trunc_{n}_{b}"]
            rec = 1 - hi_b[sel].mean() / tr_ref
            rows.append(f"{m.upper()},{n},{b},{fmt(e5[sel].mean(), d=3)},{fmt(lo_b[sel].mean(), d=3)},"
                        f"{fmt(hi_b[sel].mean(), d=3)},{fmt(tr_ref, d=3)},{rec:.2f}")
            summary[f"recovered_{m}_{n}_{b}"] = float(rec)
open(os.path.join(RES, "table_bandsplit.csv"), "w").write("\n".join(rows) + "\n")
json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1, default=float)

# ---------------------------------------------------------------- figures
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e4e4e4"
BUCKET_COLORS = ["#2a78d6", "#eda100", "#e34948"]     # smooth -> rough: blue -> yellow -> red
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})

def bucket_bars(ax, m, n, N=512, scale=1e3):
    g = growth(m, n, N)
    for k in range(3):
        mean = g[B_TV == k].mean()
        lo, hi = cluster_bootstrap(lambda r: g[r][B_TV[r] == k].mean(), traj, 2000)
        ax.bar(k, mean * scale, width=0.7, color=BUCKET_COLORS[k], edgecolor="white", linewidth=1)
        ax.errorbar(k, mean * scale, yerr=[[max(mean - lo, 0) * scale], [max(hi - mean, 0) * scale]],
                    color=INK, capsize=3, lw=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(["smooth", "med.", "rough"])
    ax.axhline(0, color=MUTED, lw=0.8)

if models:
    fig, axes = plt.subplots(1, len(models), figsize=(3.3, 2.1), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        bucket_bars(ax, m, 128)
        ax.set_title(f"{m.upper()} ({len(seeds_of(m, 128))} seeds)", color=INK, fontsize=9)
    axes[0].set_ylabel(r"$\Delta_{512}$ ($\times10^{-3}$)")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig_growth_main.{ext}"), dpi=300)
    plt.close(fig)

ns = [n for n in (32, 64, 128) if ("fno", n) in trainres]
if ns:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), constrained_layout=True)
    def xaxis(ax):
        ax.set_xscale("log", base=2); ax.set_xticks(ns); ax.set_xticklabels(ns); ax.minorticks_off()
        ax.set_xlabel("training grid (points)")
    # (a) how much the finer grids add: truncation reference error at the training Nyquist
    ax = axes[0]
    for k in range(3):
        ax.plot(ns, [summary[f"trunc_{n}_{BUCKET_NAMES[k]}"] * 1e3 for n in ns],
                "-o", color=BUCKET_COLORS[k], lw=2, ms=4, label=BUCKET_NAMES[k])
    ax.set_yscale("log"); xaxis(ax)
    ax.set_title("(a) Unseen detail", color=INK, fontsize=9)
    ax.set_ylabel(r"truncation ref. ($\times10^{-3}$)")
    # (b) the pre-registered measure: FNO growth e_512 - e_train, 95% CI
    ax = axes[1]
    for k in range(3):
        y = [trainres[("fno", n)]["gb"][k] * 1e3 for n in ns]
        lo = [trainres[("fno", n)]["gb_ci"][k][0] * 1e3 for n in ns]
        hi = [trainres[("fno", n)]["gb_ci"][k][1] * 1e3 for n in ns]
        ax.fill_between(ns, lo, hi, color=BUCKET_COLORS[k], alpha=0.18, lw=0)
        ax.plot(ns, y, "-o", color=BUCKET_COLORS[k], lw=2, ms=4)
    ax.axhline(0, color=MUTED, lw=0.8); xaxis(ax)
    ax.set_title("(b) FNO error growth", color=INK, fontsize=9)
    ax.set_ylabel(r"$e_{512}-e_{\mathrm{train}}$ ($\times10^{-3}$)")
    # (c) FNO fine-scale error relative to predicting no fine-scale content (1 = adds nothing)
    ax = axes[2]
    for k in range(3):
        r = [np.mean([hband[("fno", n, s)][1] for s in seeds_of("fno", n)], axis=0)[B_TV == k].mean()
             / summary[f"trunc_{n}_{BUCKET_NAMES[k]}"] for n in ns]
        ax.plot(ns, r, "-o", color=BUCKET_COLORS[k], lw=2, ms=4)
    ax.axhline(1, color=MUTED, lw=0.8, ls="--")
        xaxis(ax); ax.set_ylim(bottom=0)
    ax.set_title("(c) FNO fine-scale error", color=INK, fontsize=9)
    ax.set_ylabel("ratio to truncation ref.")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig_trainres.{ext}"), dpi=300)
    plt.close(fig)

print(open(os.path.join(RES, "table_main.csv")).read())
print(open(os.path.join(RES, "table_contrast.csv")).read())
print(open(os.path.join(RES, "table_starttime.csv")).read())
print(open(os.path.join(RES, "table_bucket_by_time.csv")).read())
print(open(os.path.join(RES, "table_unresolved.csv")).read())
print(open(os.path.join(RES, "table_trainres.csv")).read())
print(open(os.path.join(RES, "table_bandsplit.csv")).read())
print({k: v for k, v in summary.items() if not k.startswith(("contrast", "beta", "trainres", "unres", "trunc"))})
