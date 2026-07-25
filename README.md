# EV Charging AI — Q1 Research System
### ST-HGNN v2 with 5 Q1-Grade Upgrades

---

## Quick Start

```bash
bash run.sh          # Linux / macOS
run.bat              # Windows (double-click)
```

Open **http://localhost:5000** → click **Train All Models**

---

## The 5 Q1 Upgrades

### Upgrade 1 — Multi-Dataset (Generalisation)
| Dataset | Sessions | Stations | Source |
|---|---|---|---|
| ACN-Data | 15,976 | 54 | Real (user-provided, Caltech/JPL) |
| Palo Alto EV | 36,000 | 27 | Replica of data.cityofpaloalto.org |
| Boulder CO EV | 28,000 | 26 | Replica of open-data.bouldercolorado.gov |
| Synthetic Chicago | 10,000 | 20 | Controlled 3rd-site |
| **Combined** | **89,976** | **127** | |

### Upgrade 2 — Weather Exogenous Features
6 hourly features from Open-Meteo API (free, no key):
`temp_c · humidity_pct · precip_mm · wind_kmh · cloud_pct · is_daytime`
Realistic synthetic fallback if offline. Cached in `data/weather_cache/`.

### Upgrade 3 — Learnable Hyperedge Weights (Novel)
4 edge types, each with a trainable scalar weight (nn.Parameter):
```
W_temporal  W_spatial  W_grid  W_user  ← trained end-to-end
```
Research claim: "Adaptive hyperedge type weighting via gradient descent"

### Upgrade 4 — Probabilistic Forecasting
Output: Q10 / Q50 / Q90 per horizon step → [B, H, 3]
Metrics: **PICP** (target ≥ 0.80) · **PINAW** (lower = sharper intervals)
Loss: Huber (point) + Pinball (quantile), λ=0.30

### Upgrade 5 — Strong Baselines
| Model | Reference |
|---|---|
| TFT-lite | Lim et al. (2021) IJoF |
| N-BEATS-like | Oreshkin et al. (2020) ICLR |
| LSTM | Classic baseline |
| Transformer | Vanilla self-attention |

---

## Architecture

```
Input [B, T=24, F=31]   ← 25 temporal + 6 weather features
  ↓
Input Projection → [B, T, 128]
  ↓
LearnableHGConv × 2     ← Novel: W = nn.Parameter (Upgrade 3)
  ↓
Temporal Multi-head Attention (4 heads)
  ↓
Feed-Forward (GELU)
  ↓
ctx = x[:, -1, :]       ← last timestep
  ├─ Short head  → [B, 3]     t+1h, t+2h, t+3h
  ├─ Medium head → [B, 6]     t+1h … t+6h
  └─ Quantile head → [B, 6, 3]  Q10/Q50/Q90 (Upgrade 4)
```

---

## Output Figures (9 total)

| Fig | Content |
|---|---|
| 1 | Predicted vs Actual — all 6 horizons |
| 2 | Ablation study — 5 models |
| 3 | Training & validation loss curves |
| 4 | Demand heatmap (hour × day-of-week) |
| 5 | RMSE per forecast horizon |
| 6 | Residual error distribution |
| 7 | Probabilistic intervals (PICP/PINAW) |
| 8 | Cross-dataset generalisation |
| 9 | Learned hyperedge weights |

---

## Target Q1 Journals

| Journal | IF | Realistic with this system |
|---|---|---|
| Applied Energy | 11.2 | ✅ Yes |
| IEEE Trans. Smart Grid | 9.0 | ✅ Yes |
| Energy | 9.0 | ✅ Yes |

---

## Requirements

- Python 3.8+
- PyTorch (CPU version works, ~200 MB)
- Flask, NumPy, Pandas, Scikit-learn, Matplotlib, SciPy
- No paid APIs — 100% local, 100% free
