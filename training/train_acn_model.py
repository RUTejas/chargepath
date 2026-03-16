"""
Training Pipeline  ── Q1 Grade
================================
• Trains 5 models: ST-HGNN v2, TFT-lite, N-BEATS, LSTM, Transformer
• Combined loss: Huber (point) + Pinball (quantile)
• Early stopping, ReduceLROnPlateau, Adam, gradient clipping
• Saves checkpoints + training history JSON
• Prepares per-dataset generalisation loaders (Upgrade 1)
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.multi_dataset_loader  import load_all_datasets
from utils.feature_engineering   import (build_feature_matrix, build_sequences,
                                          save_scaler, SHORT_HORIZON, MEDIUM_HORIZON)
from utils.hypergraph_builder     import SpatioTemporalHypergraph
from models.charging_model        import (STHGNNv2, TFTLite, NBeatsLike,
                                          LSTMBaseline, TransformerBaseline,
                                          pinball_loss)

CONFIG = {
    "acn_path":       "data/acndata_sessions.json",
    "seq_len":        24,
    "short_horizon":  SHORT_HORIZON,     # 3
    "medium_horizon": MEDIUM_HORIZON,    # 6
    "d_model":        128,
    "n_hgnn_layers":  2,
    "n_heads":        4,
    "dropout":        0.15,
    "batch_size":     64,
    "lr":             1e-3,
    "weight_decay":   1e-4,
    "epochs":         80,
    "patience":       15,
    "grad_clip":      1.0,
    "train_ratio":    0.70,
    "val_ratio":      0.15,
    "device":         "cuda" if torch.cuda.is_available() else "cpu",
    "ckpt_dir":       "checkpoints",
    "results_dir":    "results",
    "use_weather":    True,
    "lambda_pinball": 0.30,    # weight of quantile loss
    "hg_sample_size": 2000,    # sessions used to build hypergraph
}


def _loss(out: dict, target: torch.Tensor,
          sh: int, criterion, lp: float) -> torch.Tensor:
    return (0.4 * criterion(out["short"],  target[:, :sh])
          + 0.6 * criterion(out["medium"], target)
          + lp  * pinball_loss(out["quantile"], target))


def prepare_data(cfg: dict) -> dict:
    """Load + feature-engineer all datasets, build DataLoaders + hypergraph."""
    datasets = load_all_datasets(cfg["acn_path"])
    combined = datasets["combined"]

    feat = build_feature_matrix(combined, use_weather=cfg["use_weather"])
    save_scaler(feat["scaler"], os.path.join(cfg["ckpt_dir"], "scaler.pkl"))

    X, y = feat["X"], feat["y"]
    Xs, ys = build_sequences(X, y, cfg["seq_len"], cfg["medium_horizon"])
    N = len(Xs)
    n_tr  = int(N * cfg["train_ratio"])
    n_val = int(N * cfg["val_ratio"])

    split = {
        "train": (Xs[:n_tr],             ys[:n_tr]),
        "val":   (Xs[n_tr:n_tr+n_val],   ys[n_tr:n_tr+n_val]),
        "test":  (Xs[n_tr+n_val:],        ys[n_tr+n_val:]),
    }
    loaders = {}
    for sp, (Xsp, ysp) in split.items():
        ds = TensorDataset(torch.tensor(Xsp), torch.tensor(ysp))
        loaders[sp] = DataLoader(
            ds, batch_size=cfg["batch_size"],
            shuffle=(sp == "train"), num_workers=0, pin_memory=False)
        print(f"[Data] {sp:5s}: {len(ds):,} samples")

    # Hypergraph — use a larger sample to get temporal edges
    hg_size = min(cfg["hg_sample_size"], len(combined))
    hg_df   = combined.sample(hg_size, random_state=42)
    hg = SpatioTemporalHypergraph(
            temporal_window_h=1,
            spatial_radius_km=0.5,
            use_user_edges=True).build(hg_df)
    theta = torch.tensor(hg.theta, dtype=torch.float32)

    # Per-dataset generalisation loaders (Upgrade 1)
    gen_loaders = {}
    for name, df_src in datasets.items():
        if name == "combined": continue
        try:
            f2 = build_feature_matrix(df_src, use_weather=cfg["use_weather"])
            if len(f2["y"]) < cfg["seq_len"] + cfg["medium_horizon"] + 20:
                continue
            # Pad / trim features to match combined in_features
            inf = feat["in_features"]
            X2  = f2["X"]
            if X2.shape[1] < inf:
                X2 = np.pad(X2, ((0,0),(0,inf-X2.shape[1])))
            elif X2.shape[1] > inf:
                X2 = X2[:, :inf]
            Xs2, ys2 = build_sequences(X2, f2["y"],
                                        cfg["seq_len"], cfg["medium_horizon"])
            nt = int(len(Xs2)*0.8)
            ds2 = TensorDataset(torch.tensor(Xs2[nt:]), torch.tensor(ys2[nt:]))
            if len(ds2) > 0:
                gen_loaders[name] = DataLoader(ds2, batch_size=cfg["batch_size"],
                                                shuffle=False, num_workers=0)
                print(f"[Data] gen/{name}: {len(ds2):,} samples")
        except Exception as e:
            print(f"[Data] gen/{name} FAILED: {e}")

    return dict(
        loaders=loaders, theta=theta,
        in_features=feat["in_features"],
        station_features=feat["station_features"],
        hourly_df=feat["hourly_df"],
        y_train=ys[:n_tr],
        hg_stats=hg.stats,
        df=combined,
        gen_loaders=gen_loaders,
    )


def train_one(model: nn.Module, loaders: dict,
              theta: torch.Tensor, cfg: dict,
              name: str = "model") -> dict:
    dev = cfg["device"]
    model.to(dev); theta = theta.to(dev)
    opt   = torch.optim.Adam(model.parameters(),
                               lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = ReduceLROnPlateau(opt, "min", patience=5, factor=0.5, verbose=False)
    crit  = nn.HuberLoss(delta=1.0)
    sh    = cfg["short_horizon"]
    lp    = cfg["lambda_pinball"]

    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    ckpt  = os.path.join(cfg["ckpt_dir"], name.replace(" ","_")+".pth")
    best_val, patience_cnt = float("inf"), 0
    history = {"train": [], "val": []}

    nparams = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*56}\n  {name}  |  params={nparams:,}  device={dev}\n{'='*56}")

    for epoch in range(1, cfg["epochs"]+1):
        t0 = time.time()
        model.train(); tr_ls = []
        for Xb, yb in loaders["train"]:
            Xb, yb = Xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = _loss(model(Xb, theta=theta), yb, sh, crit, lp)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step(); tr_ls.append(loss.item())

        model.eval(); vl_ls = []
        with torch.no_grad():
            for Xb, yb in loaders["val"]:
                Xb, yb = Xb.to(dev), yb.to(dev)
                vl_ls.append(
                    _loss(model(Xb, theta=theta), yb, sh, crit, lp).item())

        tl, vl = np.mean(tr_ls), np.mean(vl_ls)
        sched.step(vl)
        history["train"].append(round(float(tl), 5))
        history["val"].append(round(float(vl), 5))

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d}/{cfg['epochs']} | "
                  f"train={tl:.4f}  val={vl:.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.1e}  "
                  f"{time.time()-t0:.1f}s")

        if vl < best_val:
            best_val, patience_cnt = vl, 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                print(f"  Early stop @ ep{epoch}  best_val={best_val:.4f}")
                break

    model.load_state_dict(
        torch.load(ckpt, map_location=dev, weights_only=True))
    print(f"  ✓ best_val={best_val:.4f}  saved → {ckpt}")
    return history


def run_training(cfg: dict = None, progress_cb=None) -> dict:
    cfg = cfg or CONFIG
    cb  = progress_cb or (lambda p, m: None)

    cb(5, "Loading datasets + weather features…")
    data = prepare_data(cfg)
    inf  = data["in_features"]
    sh, mh = cfg["short_horizon"], cfg["medium_horizon"]

    models_def = {
        "ST-HGNN v2": STHGNNv2(inf, cfg["d_model"], cfg["n_hgnn_layers"],
                                 cfg["n_heads"], sh, mh, cfg["dropout"]),
        "TFT-lite":   TFTLite(inf, cfg["d_model"], cfg["n_heads"], sh, mh),
        "N-BEATS":    NBeatsLike(inf, cfg["seq_len"], cfg["d_model"], sh, mh),
        "LSTM":       LSTMBaseline(inf, cfg["d_model"],
                                    short_horizon=sh, medium_horizon=mh),
        "Transformer":TransformerBaseline(inf, cfg["d_model"], cfg["n_heads"],
                                           short_horizon=sh, medium_horizon=mh),
    }

    results, all_hist = {}, {}
    for i, (name, model) in enumerate(models_def.items()):
        cb(15 + i*12, f"Training {name}…")
        hist = train_one(model, data["loaders"],
                         data["theta"], cfg, name)
        results[name] = {"model": model, "history": hist}
        all_hist[name] = hist

    os.makedirs(cfg["results_dir"], exist_ok=True)
    with open(os.path.join(cfg["results_dir"], "training_history.json"), "w") as f:
        json.dump(all_hist, f, indent=2)
    print("[Training] History saved.")

    return {**results, "_data": data, "_cfg": cfg}


if __name__ == "__main__":
    run_training()
