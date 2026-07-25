"""Haversine-based nearest station finder."""
import numpy as np
import pandas as pd
from typing import Optional


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1); dl = np.radians(lon2 - lon1)
    a  = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def find_nearest(lat: float, lon: float,
                 station_df: pd.DataFrame,
                 exclude: Optional[str] = None,
                 top_k: int = 3) -> pd.DataFrame:
    df = station_df.copy()
    if exclude:
        df = df[df["station_id"] != exclude]
    df = df.copy()
    df["distance_km"] = df.apply(
        lambda r: haversine_km(lat, lon,
                               float(r["latitude"]),
                               float(r["longitude"])), axis=1)
    return df.nsmallest(top_k, "distance_km").reset_index(drop=True)
