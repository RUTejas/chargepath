"""
Research Figures (9 publication-quality plots)
================================================
Fig 1  Predicted vs Actual — all 6 horizons
Fig 2  Ablation: model comparison (MAE/RMSE/MAPE)
Fig 3  Training & validation loss curves
Fig 4  EV demand heatmap (hour × day-of-week)
Fig 5  RMSE per forecast horizon (short vs medium)
Fig 6  Residual error distribution
Fig 7  Probabilistic prediction intervals (Q10/Q50/Q90)
Fig 8  Cross-dataset generalisation (Upgrade 1)
Fig 9  Learned hyperedge type weights (Upgrade 3)
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = "results/figures"
DPI     = 150
os.makedirs(FIG_DIR, exist_ok=True)

MODEL_COLORS = {
    "ST-HGNN v2": "#3b82f6",
    "TFT-lite":   "#22c55e",
    "N-BEATS":    "#f59e0b",
    "LSTM":       "#ef4444",
    "Transformer":"#a78bfa",
}
EDGE_COLORS = ["#3b82f6","#22c55e","#f59e0b","#a78bfa"]

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#0d1117",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#8b949e",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.6,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "legend.fontsize":  8,
    "font.family":      "DejaVu Sans",
})


def _save(fig, name: str) -> str:
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Plot] Saved {path}")
    return path


def _spine(ax):
    for sp in ["top","right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(True, alpha=0.4)


# ── Fig 1 ─────────────────────────────────────────────────────────────────────
def fig1_predictions(preds, targets, model="ST-HGNN v2", sh=3, mh=6):
    n = min(200, len(preds))
    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    fig.suptitle("ST-HGNN v2: Predicted vs Actual EV Demand",
                 fontsize=13, fontweight="bold")
    c = MODEL_COLORS.get(model, "#3b82f6")
    for h in range(mh):
        ax = axes[h//3][h%3]
        ax.plot(targets[:n,h], color="#8b949e", lw=1.5, alpha=0.8, label="Actual")
        ax.plot(preds[:n,h],   color=c, lw=1.5, ls="--", alpha=0.9,
                label=model)
        tag = "Short-term" if h < sh else "Medium-term"
        ax.set_title(f"t+{h+1}h  [{tag}]", fontweight="bold", fontsize=10)
        ax.set_xlabel("Time steps", fontsize=8)
        ax.set_ylabel("kWh", fontsize=8)
        ax.legend(); _spine(ax)
    plt.tight_layout()
    return _save(fig, "fig1_predictions.png")


# ── Fig 2 ─────────────────────────────────────────────────────────────────────
def fig2_ablation(comparison: dict):
    models  = [k for k in comparison if not k.startswith("_")]
    metrics = ["MAE","RMSE","MAPE"]
    x = np.arange(len(metrics)); w = 0.15
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, name in enumerate(models):
        vals = [comparison[name]["overall"][m] for m in metrics]
        bars = ax.bar(x + i*w, vals, w, label=name,
                      color=MODEL_COLORS.get(name,f"C{i}"),
                      alpha=0.85, edgecolor="#0d1117", linewidth=0.7)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6.5,
                    color="#8b949e")
    ax.set_xticks(x + w*2); ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Error"); ax.set_title("Ablation Study — 5 Models",
                                          fontsize=12, fontweight="bold")
    ax.legend(); _spine(ax)
    plt.tight_layout()
    return _save(fig, "fig2_ablation.png")


# ── Fig 3 ─────────────────────────────────────────────────────────────────────
def fig3_loss_curves(hist_path="results/training_history.json"):
    if not os.path.exists(hist_path):
        print("[Plot] No history file, skipping fig3"); return None
    with open(hist_path) as f: hist = json.load(f)
    n = len(hist)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), squeeze=False)
    for ax, (name, h) in zip(axes[0], hist.items()):
        c = MODEL_COLORS.get(name, "#3b82f6")
        ax.plot(h["train"], color=c, lw=1.5, label="Train")
        ax.plot(h["val"],   color=c, lw=1.5, ls="--", alpha=0.6, label="Val")
        ax.set_title(name, fontweight="bold"); ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss"); ax.legend(); _spine(ax)
    fig.suptitle("Training & Validation Loss", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "fig3_loss_curves.png")


# ── Fig 4 ─────────────────────────────────────────────────────────────────────
def fig4_heatmap(hourly_df):
    import pandas as pd
    h = hourly_df.copy()
    if "timestamp" in h.columns:
        h["hour"] = pd.to_datetime(h["timestamp"]).dt.hour
        h["dow"]  = pd.to_datetime(h["timestamp"]).dt.dayofweek
    pivot = h.pivot_table(values="demand", index="hour",
                           columns="dow", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Avg Demand (kWh)")
    ax.set_xticks(range(7))
    ax.set_xticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    ax.set_yticks(range(0,24,2))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(0,24,2)])
    ax.set_xlabel("Day of Week"); ax.set_ylabel("Hour of Day")
    ax.set_title("EV Charging Demand Heatmap (all datasets)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "fig4_heatmap.png")


# ── Fig 5 ─────────────────────────────────────────────────────────────────────
def fig5_horizon_rmse(comparison: dict, mh: int = 6):
    models = [k for k in comparison if not k.startswith("_")]
    fig, ax = plt.subplots(figsize=(9, 4))
    for name in models:
        vals = [comparison[name].get(f"h{h+1}",{}).get("RMSE",0)
                for h in range(mh)]
        ax.plot([f"t+{i+1}h" for i in range(mh)], vals,
                marker="o", color=MODEL_COLORS.get(name,"#8b949e"),
                label=name, lw=2, markersize=7)
    ax.axvline(x="t+3h", color="#8b949e", ls=":", lw=1.2, alpha=0.5)
    ax.text(2.15, ax.get_ylim()[0], "short→medium", fontsize=7,
            color="#8b949e", va="bottom")
    ax.set_xlabel("Forecast Horizon"); ax.set_ylabel("RMSE")
    ax.set_title("RMSE per Forecast Horizon", fontsize=12, fontweight="bold")
    ax.legend(); _spine(ax)
    plt.tight_layout()
    return _save(fig, "fig5_horizon_rmse.png")


# ── Fig 6 ─────────────────────────────────────────────────────────────────────
def fig6_residuals(preds, targets, model="ST-HGNN v2"):
    res = (preds - targets).flatten()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(res, bins=80, color=MODEL_COLORS.get(model,"#3b82f6"),
            alpha=0.8, edgecolor="#0d1117")
    ax.axvline(0, color="#c9d1d9", ls="--", lw=1.5)
    ax.set_xlabel("Residual (Predicted − Actual kWh)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Distribution — {model}",
                 fontsize=12, fontweight="bold")
    _spine(ax); plt.tight_layout()
    return _save(fig, "fig6_residuals.png")


# ── Fig 7 ─────────────────────────────────────────────────────────────────────
def fig7_prediction_intervals(preds, targets, quantiles,
                               model="ST-HGNN v2"):
    """Probabilistic PI visualisation — Q10/Q50/Q90."""
    if quantiles is None:
        print("[Plot] No quantiles, skipping fig7"); return None
    n  = min(150, len(preds))
    h  = 0   # show t+1h horizon
    idx = np.arange(n)
    lo, med, hi = quantiles[:n,h,0], quantiles[:n,h,1], quantiles[:n,h,2]
    act = targets[:n, h]

    # Compute PICP / PINAW for title
    picp  = float(((act >= lo) & (act <= hi)).mean())
    pinaw = float((hi-lo).mean() / (act.max()-act.min()+1e-8))

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(idx, lo, hi, alpha=0.2,
                    color=MODEL_COLORS.get(model,"#3b82f6"),
                    label="80% PI [Q10–Q90]")
    ax.plot(idx, hi,  color=MODEL_COLORS.get(model,"#3b82f6"),
            lw=0.7, alpha=0.5, ls="--")
    ax.plot(idx, lo,  color=MODEL_COLORS.get(model,"#3b82f6"),
            lw=0.7, alpha=0.5, ls="--")
    ax.plot(idx, med, color=MODEL_COLORS.get(model,"#3b82f6"),
            lw=2, label="Q50 (median)")
    ax.plot(idx, act, color="#8b949e", lw=1.5, label="Actual")
    ax.set_title(
        f"Probabilistic Forecast (t+1h)  |  "
        f"PICP = {picp:.3f}  (target ≥ 0.80)  |  "
        f"PINAW = {pinaw:.3f}",
        fontsize=10, fontweight="bold")
    ax.set_xlabel("Time steps"); ax.set_ylabel("kWh")
    ax.legend(); _spine(ax); plt.tight_layout()
    return _save(fig, "fig7_prediction_intervals.png")


# ── Fig 8 ─────────────────────────────────────────────────────────────────────
def fig8_generalisation(gen_results: dict):
    """Cross-dataset generalisation bar chart."""
    if not gen_results:
        print("[Plot] No gen results, skipping fig8"); return None
    names = list(gen_results.keys())
    mae   = [gen_results[n]["MAE"]  for n in names]
    rmse  = [gen_results[n]["RMSE"] for n in names]
    x = np.arange(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x-w/2, mae,  w, label="MAE",
           color="#3b82f6", alpha=0.85, edgecolor="#0d1117")
    ax.bar(x+w/2, rmse, w, label="RMSE",
           color="#22c55e", alpha=0.85, edgecolor="#0d1117")
    for i,(m,r) in enumerate(zip(mae,rmse)):
        ax.text(i-w/2, m+0.02, f"{m:.3f}", ha="center", fontsize=8, color="#8b949e")
        ax.text(i+w/2, r+0.02, f"{r:.3f}", ha="center", fontsize=8, color="#8b949e")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Error (kWh)")
    ax.set_title("ST-HGNN v2 Cross-Dataset Generalisation",
                 fontsize=12, fontweight="bold")
    ax.legend(); _spine(ax); plt.tight_layout()
    return _save(fig, "fig8_generalisation.png")


# ── Fig 9 ─────────────────────────────────────────────────────────────────────
def fig9_edge_weights(model):
    """Learned hyperedge type weights — novel contribution."""
    try:
        w = model.get_learned_edge_weights()
    except Exception:
        print("[Plot] Could not get edge weights, skipping fig9"); return None
    labels = ["Temporal\n(same hour)",
              "Spatial\n(≤0.5 km)",
              "Grid Load\n(same site)",
              "User\n(same driver)"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, w, color=EDGE_COLORS, alpha=0.85, edgecolor="#0d1117",
                  width=0.55)
    for b, v in zip(bars, w):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                f"{v:.3f}", ha="center", fontsize=11, fontweight="bold",
                color="#c9d1d9")
    ax.set_ylabel("Learned Weight (Softplus)")
    ax.set_title("Adaptive Hyperedge Type Weights  [Novel — Upgrade 3]",
                 fontsize=11, fontweight="bold")
    _spine(ax); plt.tight_layout()
    return _save(fig, "fig9_edge_weights.png")


# ── Entry point ───────────────────────────────────────────────────────────────
def generate_all(ablation: dict, comparison: dict, predictions: dict,
                 hourly_df=None, gen_results: dict = None,
                 sh: int = 3, mh: int = 6) -> list:
    paths = []
    main  = "ST-HGNN v2"

    if main in predictions:
        P = predictions[main]["preds"]
        T = predictions[main]["targets"]
        Q = predictions[main].get("quantiles")
        paths.append(fig1_predictions(P, T, main, sh, mh))
        paths.append(fig6_residuals(P, T, main))
        paths.append(fig7_prediction_intervals(P, T, Q, main))
        if main in ablation:
            paths.append(fig9_edge_weights(ablation[main]["model"]))

    if comparison:
        paths.append(fig2_ablation(comparison))
        paths.append(fig5_horizon_rmse(comparison, mh))

    paths.append(fig3_loss_curves())

    if hourly_df is not None:
        paths.append(fig4_heatmap(hourly_df))

    if gen_results:
        paths.append(fig8_generalisation(gen_results))

    return [p for p in paths if p]
