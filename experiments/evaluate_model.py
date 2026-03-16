"""
Evaluation  ── Q1-Grade Metrics
================================
Point:        MAE · RMSE · MAPE · R²
Probabilistic:PICP (coverage, target ≥ 0.80 for 80% PI)
              PINAW (normalised interval width — lower = sharper)
Cross-dataset:per-source MAE/RMSE for generalisation claim (Upgrade 1)
"""
import numpy as np
import torch
from torch.utils.data import DataLoader


def _point(p: np.ndarray, t: np.ndarray) -> dict:
    mae  = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t)**2)))
    mape = float(np.mean(
        np.abs((p - t) / np.clip(np.abs(t), 1e-6, None))) * 100)
    ss_r = float(np.sum((p - t)**2))
    ss_t = float(np.sum((t - t.mean())**2))
    r2   = float(1 - ss_r / (ss_t + 1e-8))
    return dict(MAE=round(mae,4), RMSE=round(rmse,4),
                MAPE=round(mape,2), R2=round(r2,4))


def _prob(q: np.ndarray, t: np.ndarray) -> dict:
    """q [N,H,3]  t [N,H]  → PICP, PINAW"""
    lo, hi  = q[:,:,0], q[:,:,2]
    covered = float(((t >= lo) & (t <= hi)).mean())
    width   = float((hi - lo).mean())
    rng     = float(t.max() - t.min()) + 1e-8
    return dict(PICP=round(covered,4), PINAW=round(width/rng,4))


def evaluate_model(model, loader: DataLoader,
                   theta: torch.Tensor, device: str,
                   sh: int = 3, mh: int = 6) -> tuple:
    model.eval().to(device); theta = theta.to(device)
    pts, tgts, qps = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            out = model(Xb.to(device), theta=theta)
            pts.append(out["medium"].cpu().numpy())
            tgts.append(yb.numpy())
            if "quantile" in out:
                qps.append(out["quantile"].cpu().numpy())

    P = np.vstack(pts)
    T = np.vstack(tgts)
    Q = np.concatenate(qps, axis=0) if qps else None

    result = {}
    result["overall"]    = _point(P, T)
    result["short_avg"]  = _point(P[:,:sh], T[:,:sh])
    result["medium_avg"] = _point(P, T)
    for h in range(mh):
        result[f"h{h+1}"] = _point(P[:,h], T[:,h])
    if Q is not None:
        result["probabilistic"] = _prob(Q, T)
    return result, P, T, Q


def evaluate_all(ablation: dict, test_loader: DataLoader,
                 theta: torch.Tensor, device: str,
                 sh: int = 3, mh: int = 6) -> tuple:
    comparison, predictions = {}, {}
    print(f"\n{'─'*65}")
    print(f"  {'Model':<22} MAE    RMSE   MAPE%  R²     PICP   PINAW")
    print(f"{'─'*65}")
    for name, res in ablation.items():
        if name.startswith("_"): continue
        m, P, T, Q = evaluate_model(
            res["model"], test_loader, theta, device, sh, mh)
        comparison[name]  = m
        predictions[name] = {"preds": P, "targets": T, "quantiles": Q}
        pb = m.get("probabilistic", {})
        print(f"  {name:<22} "
              f"{m['overall']['MAE']:<7.4f}"
              f"{m['overall']['RMSE']:<7.4f}"
              f"{m['overall']['MAPE']:<7.2f}"
              f"{m['overall']['R2']:<7.4f}"
              f"{pb.get('PICP','—'):<7} "
              f"{pb.get('PINAW','—')}")
    print(f"{'─'*65}")
    return comparison, predictions


def evaluate_generalisation(model, gen_loaders: dict,
                             theta: torch.Tensor, device: str,
                             sh: int = 3, mh: int = 6) -> dict:
    """Upgrade 1: test ST-HGNN v2 on each held-out dataset."""
    gen_results = {}
    print("\n[Generalisation] ST-HGNN v2 cross-dataset evaluation:")
    for name, loader in gen_loaders.items():
        try:
            m, _, _, _ = evaluate_model(model, loader, theta, device, sh, mh)
            gen_results[name] = m["overall"]
            pb = m.get("probabilistic", {})
            print(f"  {name:<20} MAE={m['overall']['MAE']:.4f}  "
                  f"RMSE={m['overall']['RMSE']:.4f}  "
                  f"R²={m['overall']['R2']:.4f}  "
                  f"PICP={pb.get('PICP','—')}")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
    return gen_results
