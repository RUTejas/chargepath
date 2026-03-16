"""Thin wrapper — load_acn_sessions is now inside multi_dataset_loader."""
import json, os
import numpy as np
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "acndata_sessions.json")

def load_acn_sessions(filepath: str = None) -> pd.DataFrame:
    from utils.multi_dataset_loader import _load_acn
    path = filepath or os.path.abspath(DATA_PATH)
    return _load_acn(path)

def get_dataset_summary(df: pd.DataFrame) -> dict:
    return {
        "total_sessions":   int(len(df)),
        "total_energy_kwh": round(float(df["energy_delivered"].sum()), 1),
        "unique_stations":  int(df["station_id"].nunique()),
        "unique_users":     int(df["user_id"].nunique()),
        "date_start":       str(df["date"].min()),
        "date_end":         str(df["date"].max()),
        "avg_energy_kwh":   round(float(df["energy_delivered"].mean()), 2),
        "avg_duration_h":   round(float(df["duration"].mean()), 2),
        "peak_hour":        int(df.groupby("hour")["energy_delivered"].sum().idxmax()),
        "peak_dow":         int(df.groupby("day_of_week")["energy_delivered"].sum().idxmax()),
    }
