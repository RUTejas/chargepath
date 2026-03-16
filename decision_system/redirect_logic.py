"""
Adaptive Decision Engine
=========================
Uses ST-HGNN v2 quantile predictions (Q10/Q50/Q90) for uncertainty-aware decisions.
Thresholds fitted from training data percentiles (not hardcoded).

Actions: CHARGE_HERE · REDIRECT · DELAY · HOME_CHARGE
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
from decision_system.nearest_station import find_nearest


@dataclass
class Decision:
    action:          str
    station_id:      str
    reason:          str
    confidence:      float
    short_pred:      List[float]
    medium_pred:     List[float]
    demand_trend:    str = "stable"
    redirect_to:     Optional[str]   = None
    redirect_dist_km:Optional[float] = None
    wait_hours:      Optional[float] = None


class DecisionEngine:
    def __init__(self, low_pct: float = 40.0, high_pct: float = 72.0):
        self.low_pct  = low_pct
        self.high_pct = high_pct
        self.low_thr  = None
        self.high_thr = None

    def fit(self, demand: np.ndarray):
        self.low_thr  = float(np.percentile(demand, self.low_pct))
        self.high_thr = float(np.percentile(demand, self.high_pct))
        print(f"[Decision] Thresholds → low={self.low_thr:.2f} kWh  "
              f"high={self.high_thr:.2f} kWh")

    def decide(self, station_id: str,
               short_pred: np.ndarray,
               medium_pred: np.ndarray,
               station_df: pd.DataFrame,
               station_lat: float,
               station_lon: float,
               home_available: bool = True,
               q10: np.ndarray = None,
               q90: np.ndarray = None) -> Decision:

        assert self.low_thr is not None, "Call fit() first"

        sp  = [round(float(v), 2) for v in short_pred]
        mp  = [round(float(v), 2) for v in medium_pred]
        d1  = sp[0]
        slope = (sp[-1] - sp[0]) / max(len(sp)-1, 1)
        trend = "rising" if slope > 0.5 else ("falling" if slope < -0.5 else "stable")

        # Uncertainty range (from quantiles if available)
        uncertainty = ""
        if q10 is not None and q90 is not None:
            lo, hi = float(q10[0]), float(q90[0])
            uncertainty = f" [Q10={lo:.1f}, Q90={hi:.1f}]"

        nearby = find_nearest(station_lat, station_lon, station_df,
                               exclude=station_id, top_k=3)

        # ── Rule 1: Low demand ────────────────────────────────────────────────
        if d1 < self.low_thr:
            conf = round(min(0.97, 0.60 + (self.low_thr-d1)/max(self.low_thr,1)*0.4), 3)
            return Decision(
                action="CHARGE_HERE", station_id=station_id,
                reason=(f"Demand low ({d1:.1f} kWh < {self.low_thr:.1f} threshold)"
                        f"{uncertainty}. Trend: {trend}. Optimal charging window."),
                confidence=conf, short_pred=sp, medium_pred=mp,
                demand_trend=trend)

        # ── Rule 2: High now, drops soon ──────────────────────────────────────
        if d1 >= self.high_thr and len(sp) >= 2 and sp[1] < self.low_thr:
            return Decision(
                action="DELAY", station_id=station_id,
                reason=(f"Peak now ({d1:.1f} kWh) → drops to {sp[1]:.1f} kWh "
                        f"in ~1h. Waiting recommended."),
                confidence=0.78, short_pred=sp, medium_pred=mp,
                demand_trend="falling", wait_hours=1.0)

        # ── Rule 3: High → redirect ───────────────────────────────────────────
        if d1 >= self.high_thr:
            conf = round(min(0.95, 0.55 + (d1-self.high_thr)/max(self.high_thr,1)*0.4), 3)
            if len(nearby) > 0:
                best = nearby.iloc[0]
                return Decision(
                    action="REDIRECT", station_id=station_id,
                    reason=(f"High demand ({d1:.1f} kWh ≥ {self.high_thr:.1f})"
                            f"{uncertainty}. "
                            f"Nearest alternative: {best['station_id']} "
                            f"({best['distance_km']:.2f} km)."),
                    confidence=conf, short_pred=sp, medium_pred=mp,
                    demand_trend=trend,
                    redirect_to=str(best["station_id"]),
                    redirect_dist_km=round(float(best["distance_km"]), 3))
            if home_available:
                return Decision(
                    action="HOME_CHARGE", station_id=station_id,
                    reason=(f"High demand ({d1:.1f} kWh), no nearby alternative. "
                            f"Home charging recommended."),
                    confidence=0.88, short_pred=sp, medium_pred=mp,
                    demand_trend=trend)
            return Decision(
                action="DELAY", station_id=station_id,
                reason=f"High demand ({d1:.1f} kWh). No alternatives. Wait ~1–2h.",
                confidence=0.55, short_pred=sp, medium_pred=mp,
                demand_trend=trend, wait_hours=1.5)

        # ── Rule 4: Moderate ──────────────────────────────────────────────────
        return Decision(
            action="CHARGE_HERE", station_id=station_id,
            reason=(f"Moderate demand ({d1:.1f} kWh). "
                    f"Charging feasible{uncertainty}."),
            confidence=0.68, short_pred=sp, medium_pred=mp,
            demand_trend=trend)
