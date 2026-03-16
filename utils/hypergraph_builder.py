"""
Spatio-Temporal Hypergraph  ── Upgrade 3 (supports learnable weights)
======================================================================
4 hyperedge types with initial weights (later overridden by nn.Parameter):
  0  Temporal  — sessions within same 1-hour bucket (window=1h)
  1  Spatial   — stations within spatial_radius_km (Haversine)
  2  Grid      — sessions sharing dataset site / cluster
  3  User      — repeat sessions by the same user

Also returns edge_type_ids [E] so the model can apply
per-type learnable weights (Upgrade 3).
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

try:
    import torch; _TORCH = True
except ImportError:
    _TORCH = False


class SpatioTemporalHypergraph:

    def __init__(self, temporal_window_h: int = 1,
                 spatial_radius_km: float = 0.5,
                 use_user_edges: bool = True):
        self.temporal_window_h  = temporal_window_h
        self.spatial_radius_km  = spatial_radius_km
        self.use_user_edges     = use_user_edges
        self.H              = None   # [N, E]  incidence matrix
        self.W              = None   # [E]     initial edge weights
        self.theta          = None   # [N, N]  propagation matrix
        self.edge_type_ids  = None   # [E]     int type per edge
        self.stats          = {}

    # ─────────────────────────────────────────────────────────────────────────
    def build(self, df: pd.DataFrame) -> "SpatioTemporalHypergraph":
        df = df.copy().reset_index(drop=True)
        n  = len(df)
        print(f"[Hypergraph] Building {n} nodes …")

        edges, weights, type_ids = [], [], []

        te = self._temporal(df)
        for e in te:
            edges.append(e); weights.append(1.0); type_ids.append(0)

        se = self._spatial(df)
        for e in se:
            edges.append(e); weights.append(1.5); type_ids.append(1)

        ge = self._grid(df)
        for e in ge:
            edges.append(e); weights.append(2.0); type_ids.append(2)

        ue = []
        if self.use_user_edges and "user_id" in df.columns:
            ue = self._user(df)
            for e in ue:
                edges.append(e); weights.append(0.8); type_ids.append(3)

        m = len(edges)
        if m == 0:
            # Degenerate case: make identity-like structure
            edges    = [[i] for i in range(n)]
            weights  = [1.0]*n
            type_ids = [0]*n
            m = n

        H = np.zeros((n, m), dtype=np.float32)
        for j, edge in enumerate(edges):
            for v in edge:
                if 0 <= v < n:
                    H[v, j] = 1.0

        W = np.array(weights, dtype=np.float32)
        self.H            = H
        self.W            = W
        self.edge_type_ids = np.array(type_ids, dtype=np.int64)
        self.theta        = _propagate(H, W)
        self.stats = dict(nodes=n, edges=m, temporal=len(te),
                          spatial=len(se), grid=len(ge), user=len(ue))
        print(f"[Hypergraph] N={n}  E={m}  "
              f"T={len(te)} S={len(se)} G={len(ge)} U={len(ue)}")
        return self

    # ─────────────────────────────────────────────────────────────────────────
    def _temporal(self, df: pd.DataFrame) -> list:
        edges = []
        ct_col = "connection_time"
        if ct_col not in df.columns:
            return edges
        df = df.copy()
        df["_tb"] = pd.to_datetime(df[ct_col]).dt.floor(
            f"{self.temporal_window_h}h")
        for _, g in df.groupby("_tb"):
            if len(g) >= 2:
                edges.append(g.index.tolist())
        return edges

    def _spatial(self, df: pd.DataFrame) -> list:
        edges = []
        if "latitude" not in df.columns or "longitude" not in df.columns:
            return edges
        st = (df[["station_id","latitude","longitude"]]
              .drop_duplicates("station_id").reset_index(drop=True))
        if len(st) < 2:
            return edges
        coords  = st[["latitude","longitude"]].values
        dist_km = cdist(coords, coords) * 111.0   # deg → km approx
        for i in range(len(st)):
            nb_idx  = np.where(dist_km[i] <= self.spatial_radius_km)[0]
            if len(nb_idx) < 2: continue
            nb_sids = set(st.iloc[nb_idx]["station_id"].values)
            sess_idx = df[df["station_id"].isin(nb_sids)].index.tolist()
            if len(sess_idx) >= 2:
                edges.append(sess_idx)
        return edges

    def _grid(self, df: pd.DataFrame) -> list:
        edges = []
        col = ("site_id" if "site_id" in df.columns
               else "dataset_source" if "dataset_source" in df.columns
               else None)
        if col is None:
            return edges
        for _, g in df.groupby(col):
            if len(g) >= 2:
                edges.append(g.index.tolist())
        return edges

    def _user(self, df: pd.DataFrame) -> list:
        edges = []
        for uid, g in df.groupby("user_id"):
            if str(uid).upper() in ("ANON","NONE","NAN",""):
                continue
            if 2 <= len(g) <= 80:
                edges.append(g.index.tolist())
        return edges

    def to_torch(self, device: str = "cpu") -> dict:
        if not _TORCH:
            raise ImportError("torch not installed")
        import torch
        return {
            "H":            torch.tensor(self.H, dtype=torch.float32, device=device),
            "W":            torch.tensor(self.W, dtype=torch.float32, device=device),
            "theta":        torch.tensor(self.theta, dtype=torch.float32, device=device),
            "edge_type_ids":torch.tensor(self.edge_type_ids, dtype=torch.long, device=device),
        }


def _propagate(H: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Θ = D_v^{-1} · H · W · D_e^{-1} · H^T"""
    Dv = H.sum(axis=1)
    De = H.sum(axis=0)
    Dv_inv = np.where(Dv > 0, 1.0/Dv, 0.0)
    De_inv = np.where(De > 0, 1.0/De, 0.0)
    HW     = H * W[np.newaxis,:]
    HWDe   = HW * De_inv[np.newaxis,:]
    theta  = Dv_inv[:,np.newaxis] * (HWDe @ H.T)
    return theta.astype(np.float32)
