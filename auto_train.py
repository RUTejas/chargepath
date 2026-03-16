"""
auto_train.py — Train-Once, Load-Forever
=========================================
Logic:
  1. App starts → check if checkpoints/trained.lock exists
  2. YES  → load model + results from disk instantly (30 sec)
  3. NO   → train all 5 models, save everything, create lock file
             never retrains again unless lock file is deleted

Adam optimizer is used (torch.optim.Adam).
Results survive server restarts, redeployments, and container rebuilds
as long as the /checkpoints volume is mounted (Railway persistent disk).
"""
import os, json, threading, traceback
from datetime import datetime
import numpy as np

LOCK_FILE    = "checkpoints/trained.lock"
RESULTS_FILE = "checkpoints/auto_results.json"
MODEL_FILE   = "checkpoints/ST-HGNN_v2.pth"


def is_trained() -> bool:
    """True only when training fully completed (lock file exists)."""
    return (os.path.exists(LOCK_FILE) and
            os.path.exists(RESULTS_FILE) and
            os.path.exists(MODEL_FILE))


def load_saved_results() -> dict:
    if not os.path.exists(RESULTS_FILE):
        return {}
    with open(RESULTS_FILE) as f:
        return json.load(f)


def _write_lock(msg: str = ""):
    os.makedirs("checkpoints", exist_ok=True)
    with open(LOCK_FILE, "w") as f:
        json.dump({"trained_at": datetime.now().isoformat(),
                   "note": msg}, f, indent=2)


def _save_all(comparison, gen_results, edge_weights, decisions):
    """Persist all results to disk after successful training."""
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results",     exist_ok=True)
    payload = {
        "comparison":   comparison,
        "gen_results":  gen_results,
        "edge_weights": edge_weights,
        "decisions":    decisions,
        "saved_at":     datetime.now().isoformat(),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    with open("results/evaluation_metrics.json", "w") as f:
        json.dump(comparison, f, indent=2)
    with open("results/generalisation.json", "w") as f:
        json.dump(gen_results, f, indent=2)
    with open("results/decision_outputs.json", "w") as f:
        json.dump(decisions, f, indent=2)
    print("[AutoTrain] Results saved to disk ✓")


# ── FULL TRAINING (runs once only) ─────────────────────────────────────────
def _run_training(STATE, STORE):
    try:
        STATE.update(status="training", progress=5,
                     message="First-time setup: training AI models (Adam optimizer)…",
                     started_at=datetime.now().isoformat(), error=None)

        from training.train_acn_model    import run_training, CONFIG
        from experiments.evaluate_model  import evaluate_all, evaluate_generalisation
        from experiments.plot_results    import generate_all
        from decision_system.redirect_logic import DecisionEngine
        import torch

        cfg = CONFIG.copy()
        def cb(p, m): STATE.update(progress=p, message=m)

        # ── Train all 5 models with Adam ──────────────────────────────────
        cb(8, "Training ST-HGNN v2 with Adam optimizer…")
        results  = run_training(cfg, progress_cb=cb)
        data     = results["_data"]
        ablation = {k:v for k,v in results.items() if not k.startswith("_")}

        # ── Evaluate ─────────────────────────────────────────────────────
        cb(76, "Evaluating 5 models…")
        comparison, predictions = evaluate_all(
            ablation, data["loaders"]["test"],
            data["theta"], cfg["device"],
            cfg["short_horizon"], cfg["medium_horizon"])

        # ── Cross-dataset generalisation ──────────────────────────────────
        cb(83, "Cross-dataset generalisation…")
        main_model = results["ST-HGNN v2"]["model"]
        gen_results = evaluate_generalisation(
            main_model, data["gen_loaders"],
            data["theta"], cfg["device"])

        # ── Research figures ──────────────────────────────────────────────
        cb(88, "Generating 9 research figures…")
        generate_all(ablation, comparison, predictions,
                     hourly_df=data["hourly_df"],
                     gen_results=gen_results,
                     sh=cfg["short_horizon"], mh=cfg["medium_horizon"])

        # ── Decision engine ───────────────────────────────────────────────
        cb(93, "Building decision engine…")
        engine = DecisionEngine()
        engine.fit(data["y_train"])

        dev   = cfg["device"]
        theta = data["theta"].to(dev)
        sdf   = data["station_features"]
        ds_   = data["loaders"]["test"].dataset

        demos = []
        idxs  = np.random.choice(len(ds_), min(12, len(ds_)), replace=False)
        for idx in idxs:
            Xb, _ = ds_[int(idx)]
            with torch.no_grad():
                out = main_model(Xb.unsqueeze(0).to(dev), theta=theta)
            sp = out["short"].squeeze().cpu().numpy()
            mp = out["medium"].squeeze().cpu().numpy()
            qp = out["quantile"].squeeze().cpu().numpy()
            row = sdf.sample(1).iloc[0]
            d = engine.decide(str(row["station_id"]), sp, mp, sdf,
                              float(row["latitude"]), float(row["longitude"]),
                              q10=qp[:,0], q90=qp[:,2])
            demos.append({
                "station_id":  str(row["station_id"]),
                "action":      d.action,
                "confidence":  d.confidence,
                "reason":      d.reason,
                "short_pred":  d.short_pred,
                "medium_pred": d.medium_pred,
                "demand_trend":d.demand_trend,
                "redirect_to": d.redirect_to,
                "redirect_km": d.redirect_dist_km,
                "q10": [round(float(v),2) for v in qp[:,0]],
                "q50": [round(float(v),2) for v in qp[:,1]],
                "q90": [round(float(v),2) for v in qp[:,2]],
            })

        try:    ew = main_model.get_learned_edge_weights()
        except: ew = [1.0, 1.5, 2.0, 0.8]

        # ── Save everything to disk ───────────────────────────────────────
        cb(97, "Saving model and results to disk…")
        _save_all(comparison, gen_results, ew, demos)
        _write_lock("Training complete — Adam optimizer — never retrain")

        # ── Populate STORE ────────────────────────────────────────────────
        STORE.update(
            results=results, data=data,
            comparison=comparison, predictions=predictions,
            gen_results=gen_results, decisions=demos,
            edge_weights=ew, cfg=cfg,
            engine=engine, model=main_model,
            theta=theta, device=dev,
            station_features=sdf,
        )

        STATE.update(status="ready", progress=100,
                     message="Ready ✓  —  model saved, will load instantly next time",
                     finished_at=datetime.now().isoformat())
        print("[AutoTrain] ✓ Done. Lock file written. Will never retrain.")

    except Exception as e:
        STATE.update(status="error", progress=0,
                     message=str(e), error=traceback.format_exc())
        print("[AutoTrain] ERROR:\n", traceback.format_exc())


# ── FAST LOAD FROM DISK (all subsequent starts) ───────────────────────────
def _load_from_disk(STATE, STORE):
    try:
        import torch
        from training.train_acn_model       import CONFIG
        from decision_system.redirect_logic import DecisionEngine
        from utils.multi_dataset_loader     import load_all_datasets
        from utils.feature_engineering      import build_feature_matrix, build_sequences
        from utils.hypergraph_builder       import SpatioTemporalHypergraph
        from torch.utils.data               import TensorDataset, DataLoader
        from models.charging_model          import STHGNNv2

        STATE.update(status="loading", progress=10,
                     message="Loading saved model from disk…")

        saved = load_saved_results()
        cfg   = CONFIG.copy()
        dev   = cfg["device"]

        STATE.update(progress=25, message="Rebuilding feature matrix…")
        ds   = load_all_datasets(cfg["acn_path"])
        feat = build_feature_matrix(ds["combined"], use_weather=cfg["use_weather"])
        Xs, ys = build_sequences(feat["X"], feat["y"],
                                  cfg["seq_len"], cfg["medium_horizon"])
        n_tr  = int(len(Xs) * cfg["train_ratio"])
        n_val = int(len(Xs) * cfg["val_ratio"])
        test_ds = TensorDataset(
            torch.tensor(Xs[n_tr+n_val:]),
            torch.tensor(ys[n_tr+n_val:]))
        test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"],
                                  shuffle=False, num_workers=0)

        STATE.update(progress=50, message="Rebuilding hypergraph…")
        hg_df = ds["combined"].sample(min(2000, len(ds["combined"])),
                                       random_state=42)
        hg    = SpatioTemporalHypergraph(1, 0.5, True).build(hg_df)
        theta = torch.tensor(hg.theta, dtype=torch.float32)

        STATE.update(progress=70, message="Loading Adam-trained model weights…")
        model = STHGNNv2(feat["in_features"], cfg["d_model"],
                         cfg["n_hgnn_layers"], cfg["n_heads"],
                         cfg["short_horizon"], cfg["medium_horizon"],
                         cfg["dropout"])
        model.load_state_dict(
            torch.load(MODEL_FILE, map_location=dev, weights_only=True))
        model.to(dev).eval()

        engine = DecisionEngine()
        engine.fit(ys[:n_tr])

        STORE.update(
            comparison   = saved.get("comparison",   {}),
            gen_results  = saved.get("gen_results",  {}),
            edge_weights = saved.get("edge_weights"),
            decisions    = saved.get("decisions",    []),
            cfg          = cfg,
            model        = model,
            theta        = theta.to(dev),
            device       = dev,
            engine       = engine,
            station_features = feat["station_features"],
            data = {
                "loaders":   {"test": test_loader},
                "theta":     theta,
                "hourly_df": feat["hourly_df"],
                "y_train":   ys[:n_tr],
                "gen_loaders": {},
            },
        )
        STATE.update(status="ready", progress=100,
                     message="Ready ✓  —  loaded from saved checkpoint (instant)",
                     finished_at=datetime.now().isoformat())
        print("[AutoTrain] ✓ Loaded from disk instantly.")

    except Exception as e:
        # Disk load failed → fallback to retrain
        print(f"[AutoTrain] Disk load failed ({e}) → retraining…")
        # Remove broken lock so training runs clean
        for f in [LOCK_FILE, RESULTS_FILE]:
            try: os.remove(f)
            except: pass
        STATE.update(status="idle", progress=0, message="Retraining…")
        threading.Thread(target=_run_training, args=(STATE, STORE),
                         daemon=True).start()


# ── PUBLIC ENTRY POINT ─────────────────────────────────────────────────────
def start(STATE: dict, STORE: dict):
    """
    Called once at app startup. Non-blocking.
    - Already trained? → load from disk (~30 sec)
    - Never trained?   → train now (once, saves forever)
    """
    if is_trained():
        print("[AutoTrain] Checkpoint found → loading from disk (no retraining)")
        t = threading.Thread(target=_load_from_disk,
                             args=(STATE, STORE), daemon=True)
    else:
        print("[AutoTrain] No checkpoint → training now (Adam optimizer, runs once)")
        t = threading.Thread(target=_run_training,
                             args=(STATE, STORE), daemon=True)
    t.start()


def force_retrain(STATE: dict, STORE: dict):
    """
    Optional: call this to force a fresh retrain (e.g. after adding new data).
    Deletes the lock file then retrains.
    """
    for f in [LOCK_FILE, RESULTS_FILE, MODEL_FILE]:
        try: os.remove(f)
        except: pass
    print("[AutoTrain] Lock removed → forcing retrain")
    threading.Thread(target=_run_training, args=(STATE, STORE),
                     daemon=True).start()
