"""
Weather Feature Integration  ── Upgrade 2: Exogenous Features
==============================================================
Source: Open-Meteo Historical Weather API (https://open-meteo.com)
  • Free, no API key, CC-BY 4.0 licence
  • ERA5 reanalysis, hourly, from 1940, any location

Features added per hour (6 total):
  temp_c        — temperature (°C)          major EV demand driver
  humidity_pct  — relative humidity (%)
  precip_mm     — precipitation (mm)        rain reduces outdoor use
  wind_kmh      — wind speed (km/h)
  cloud_pct     — cloud cover (%)
  is_daytime    — 0/1 sunlight indicator

Fallback: if API unreachable (offline env), realistic synthetic
weather is generated from seasonal + diurnal physics models.
Results are cached to data/weather_cache/ to avoid re-fetching.
"""
import os, json
import numpy as np
import pandas as pd

CACHE_DIR = "data/weather_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

WEATHER_COLS = [
    "temp_c", "humidity_pct", "precip_mm",
    "wind_kmh", "cloud_pct", "is_daytime",
]

# Lat / lon / timezone per dataset source
SITE_META = {
    "acn":               (34.1377, -118.1253, "America%2FLos_Angeles"),
    "palo_alto":         (37.4419, -122.1430, "America%2FLos_Angeles"),
    "boulder":           (40.0150, -105.2705, "America%2FDenver"),
    "synthetic_chicago": (41.8827, -87.6233,  "America%2FChicago"),
    "default":           (34.1377, -118.1253, "America%2FLos_Angeles"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────
def attach_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge hourly weather into session-level DataFrame.
    Adds WEATHER_COLS to every row matched to connection_time floor(h).
    """
    df = df.copy()
    src_col = "dataset_source"
    if src_col not in df.columns:
        df[src_col] = "default"

    parts = []
    for src, grp in df.groupby(src_col):
        lat, lon, tz = SITE_META.get(src, SITE_META["default"])
        start = str(grp["connection_time"].min().date())
        end   = str(grp["connection_time"].max().date())
        wdf   = _get_weather(lat, lon, tz, start, end, src)

        grp = grp.copy()
        grp["_hkey"] = pd.to_datetime(grp["connection_time"]).dt.floor("h")
        wdf.index    = pd.to_datetime(wdf.index)
        merged = grp.merge(wdf[WEATHER_COLS], how="left",
                           left_on="_hkey", right_index=True)
        parts.append(merged)

    out = pd.concat(parts, ignore_index=True).drop(columns=["_hkey"], errors="ignore")
    for c in WEATHER_COLS:
        if c in out.columns:
            out[c] = out[c].ffill().bfill().fillna(0)
    return out


def build_hourly_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Return hourly weather DataFrame aligned to demand timestamps."""
    if WEATHER_COLS[0] not in df.columns:
        df = attach_weather(df)
    df["_ts"] = pd.to_datetime(df["connection_time"]).dt.floor("h")
    hw = (df.groupby("_ts")[WEATHER_COLS].mean()
            .reset_index().rename(columns={"_ts": "timestamp"}))
    return hw


# ─────────────────────────────────────────────────────────────────────────────
# Internal: fetch or generate weather
# ─────────────────────────────────────────────────────────────────────────────
def _get_weather(lat, lon, tz, start, end, key) -> pd.DataFrame:
    cache_file = os.path.join(CACHE_DIR, f"{key}_{start}_{end}.csv")
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"[Weather] Cache hit: {key} ({start}→{end})")
        return df

    # Try Open-Meteo
    try:
        import urllib.request
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={start}&end_date={end}"
            f"&hourly=temperature_2m,relative_humidity_2m,"
            f"precipitation,wind_speed_10m,cloud_cover,is_day"
            f"&timezone={tz}&wind_speed_unit=kmh"
        )
        with urllib.request.urlopen(url, timeout=12) as r:
            data = json.loads(r.read().decode())
        h = data["hourly"]
        df = pd.DataFrame({
            "temp_c":       h["temperature_2m"],
            "humidity_pct": h["relative_humidity_2m"],
            "precip_mm":    h["precipitation"],
            "wind_kmh":     h["wind_speed_10m"],
            "cloud_pct":    h["cloud_cover"],
            "is_daytime":   h["is_day"],
        }, index=pd.to_datetime(h["time"]))
        df = df.ffill().bfill().fillna(0)
        df.to_csv(cache_file)
        print(f"[Weather] API fetched: {key} {len(df)} hours → cached")
        return df
    except Exception as e:
        print(f"[Weather] API unavailable ({type(e).__name__}). Using synthetic.")

    # Synthetic fallback
    return _synthetic(lat, lon, start, end, key, cache_file)


def _synthetic(lat, lon, start, end, key, cache_file) -> pd.DataFrame:
    """
    Physics-based synthetic weather:
      Temperature: seasonal sine + diurnal sine + Gaussian noise
      Precipitation: Poisson rain events at realistic rates
      Humidity: inversely correlated with temperature
      Cloud cover: seasonal variation + noise
    """
    idx = pd.date_range(start, end, freq="h")
    n   = len(idx)
    rng = np.random.default_rng(abs(hash(key)) % (2**31))

    doy  = idx.dayofyear.values.astype(float)
    hour = idx.hour.values.astype(float)

    # Seasonal mean temperature based on latitude
    t_mean   = 18.0 - abs(lat - 35.0) * 0.4
    t_season = t_mean + 10.0 * np.sin(2*np.pi*(doy - 80)/365)
    t_daily  = 5.0  * np.sin(2*np.pi*(hour - 6)/24)
    temp     = t_season + t_daily + rng.normal(0, 1.8, n)

    # Humidity: anticorrelated with temperature
    hum  = np.clip(72 - 0.9*(temp - t_mean) + rng.normal(0, 8, n), 15, 100)

    # Precipitation (Poisson events, ~8 rain-days/month)
    rain_ev = rng.random(n) < 0.11
    precip  = np.where(rain_ev, rng.exponential(2.5, n), 0.0)

    # Wind
    wind = np.clip(rng.gamma(2.0, 5.5, n), 0, 90)

    # Cloud cover (seasonal + noise)
    cloud = np.clip(42 + 18*np.sin(2*np.pi*doy/365) + rng.normal(0, 22, n), 0, 100)

    # Daytime flag (simple threshold)
    is_day = ((hour >= 6) & (hour <= 19)).astype(float)

    df = pd.DataFrame({
        "temp_c":       np.round(temp, 1),
        "humidity_pct": np.round(hum, 1),
        "precip_mm":    np.round(precip, 2),
        "wind_kmh":     np.round(wind, 1),
        "cloud_pct":    np.round(cloud, 1),
        "is_daytime":   is_day,
    }, index=idx)
    df.index.name = "datetime"
    df.to_csv(cache_file)
    print(f"[Weather] Synthetic generated: {key} {len(df)} hours → cached")
    return df
