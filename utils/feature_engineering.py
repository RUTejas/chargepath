"""
Feature Engineering  ── 31 Features (25 temporal + 6 weather)
==============================================================
Groups:
  [0:8]   Cyclical temporal + binary flags
  [8:15]  Lag features: 1,2,3,6,12,24,48h
  [15:23] Rolling mean + std (windows 3,6,12,24h)
  [23:25] Differencing (diff_1, diff_24)
  [25:31] Weather exogenous: temp, humidity, precip, wind, cloud, daytime
"""
import os, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SHORT_HORIZON  = 3
MEDIUM_HORIZON = 6
SEQ_LEN        = 24
LAG_STEPS      = [1, 2, 3, 6, 12, 24, 48]
ROLL_WINDOWS   = [3, 6, 12, 24]
WEATHER_COLS   = ["temp_c", "humidity_pct", "precip_mm",
                  "wind_kmh", "cloud_pct", "is_daytime"]


def build_feature_matrix(df: pd.DataFrame,
                          use_weather: bool = True) -> dict:
    df = df.copy().sort_values("connection_time").reset_index(drop=True)

    # ── Hourly demand aggregation ─────────────────────────────────────────────
    hourly = (df.groupby(["date", "hour"])["energy_delivered"]
                .sum().reset_index()
                .rename(columns={"energy_delivered": "demand"}))
    hourly["timestamp"] = pd.to_datetime(
        hourly["date"].astype(str) + " " +
        hourly["hour"].astype(str) + ":00:00")
    hourly = hourly.sort_values("timestamp").reset_index(drop=True)

    # Fill missing hours
    full_idx = pd.date_range(hourly["timestamp"].min(),
                              hourly["timestamp"].max(), freq="h")
    hourly = (hourly.set_index("timestamp")
                    .reindex(full_idx, fill_value=0.0)
                    .reset_index()
                    .rename(columns={"index": "timestamp"}))

    hourly["hour"]        = hourly["timestamp"].dt.hour
    hourly["day_of_week"] = hourly["timestamp"].dt.dayofweek
    hourly["month"]       = hourly["timestamp"].dt.month
    hourly["is_weekend"]  = (hourly["day_of_week"] >= 5).astype(int)
    hourly["is_peak"]     = hourly["hour"].isin([13,14,15,16,17]).astype(int)

    # ── Cyclical encodings ────────────────────────────────────────────────────
    for col, period in [("hour",24), ("day_of_week",7), ("month",12)]:
        hourly[f"{col}_sin"] = np.sin(2*np.pi*hourly[col]/period)
        hourly[f"{col}_cos"] = np.cos(2*np.pi*hourly[col]/period)

    # ── Lag features ─────────────────────────────────────────────────────────
    for lag in LAG_STEPS:
        hourly[f"lag_{lag}"] = hourly["demand"].shift(lag)

    # ── Rolling statistics ────────────────────────────────────────────────────
    for w in ROLL_WINDOWS:
        hourly[f"roll_mean_{w}"] = hourly["demand"].shift(1).rolling(w).mean()
        hourly[f"roll_std_{w}"]  = (hourly["demand"].shift(1)
                                    .rolling(w).std().fillna(0))

    # ── Differencing ──────────────────────────────────────────────────────────
    hourly["diff_1"]  = hourly["demand"].diff(1).fillna(0)
    hourly["diff_24"] = hourly["demand"].diff(24).fillna(0)

    # ── Weather exogenous (Upgrade 2) ─────────────────────────────────────────
    weather_cols_used = []
    if use_weather:
        try:
            from utils.weather_features import build_hourly_weather
            hw = build_hourly_weather(df)
            hw["timestamp"] = pd.to_datetime(hw["timestamp"])
            hourly = hourly.merge(hw, on="timestamp", how="left")
            for c in WEATHER_COLS:
                if c in hourly.columns:
                    hourly[c] = hourly[c].ffill().bfill().fillna(0)
                    weather_cols_used.append(c)
            print(f"[Features] Weather attached: {weather_cols_used}")
        except Exception as e:
            print(f"[Features] Weather skipped ({e})")

    hourly = hourly.dropna().reset_index(drop=True)

    base_cols = (
        ["hour_sin","hour_cos","day_of_week_sin","day_of_week_cos",
         "month_sin","month_cos","is_weekend","is_peak"] +
        [f"lag_{l}"       for l in LAG_STEPS]  +
        [f"roll_mean_{w}" for w in ROLL_WINDOWS] +
        [f"roll_std_{w}"  for w in ROLL_WINDOWS] +
        ["diff_1","diff_24"]
    )
    all_cols = base_cols + weather_cols_used

    X  = hourly[all_cols].values.astype(np.float32)
    y  = hourly["demand"].values.astype(np.float32)
    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    print(f"[Features] X={Xs.shape}  y={y.shape}  "
          f"n_features={len(all_cols)}  weather={'YES' if weather_cols_used else 'NO'}")

    return {
        "X":               Xs,
        "y":               y,
        "feature_names":   all_cols,
        "weather_cols":    weather_cols_used,
        "scaler":          sc,
        "station_features":_station_features(df),
        "hourly_df":       hourly,
        "timestamps":      hourly["timestamp"].values,
        "in_features":     len(all_cols),
    }


def _station_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("station_id").agg(
        avg_energy    =("energy_delivered","mean"),
        total_sessions=("session_id","count"),
        avg_duration  =("duration","mean"),
        peak_hour     =("hour", lambda x: int(x.mode()[0])),
        latitude      =("latitude","first"),
        longitude     =("longitude","first"),
        site_id       =("site_id","first"),
    ).reset_index()


def build_sequences(X: np.ndarray, y: np.ndarray,
                    seq_len: int = SEQ_LEN,
                    horizon: int = MEDIUM_HORIZON) -> tuple:
    Xs, ys = [], []
    for i in range(len(X) - seq_len - horizon + 1):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len:i+seq_len+horizon])
    Xs = np.array(Xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    print(f"[Sequences] X={Xs.shape}  y={ys.shape}  horizon={horizon}h")
    return Xs, ys


def save_scaler(sc, path="checkpoints/scaler.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"wb") as f: pickle.dump(sc, f)

def load_scaler(path="checkpoints/scaler.pkl"):
    with open(path,"rb") as f: return pickle.load(f)
