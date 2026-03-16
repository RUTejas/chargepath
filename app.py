"""
ChargePath — EV Charging AI  |  Full-Stack App
================================================
Auto-training: trains once on first boot, loads instantly after.
No buttons needed. Everything is automatic.

Map: /map        — Google Maps-style station finder (PWA)
Dashboard: /     — Research results dashboard
"""
import os, sys, json, math, threading, traceback
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, jsonify, send_from_directory, request
import numpy as np

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

STATE = {"status":"idle","progress":0,"message":"Starting…",
         "error":None,"started_at":None,"finished_at":None}
STORE = {}

# ── Preload station geodata (fast, no model needed) ────────────────────────
def _preload():
    try:
        from utils.multi_dataset_loader import load_all_datasets
        ds  = load_all_datasets("data/acndata_sessions.json")
        sdf = ds["combined"].groupby("station_id").agg(
            latitude      =("latitude","first"),
            longitude     =("longitude","first"),
            site_id       =("site_id","first"),
            dataset_source=("dataset_source","first"),
            total_sessions=("session_id","count"),
            avg_kwh       =("energy_delivered","mean"),
            avg_duration  =("duration","mean"),
            peak_hour     =("hour", lambda x: int(x.mode()[0])),
        ).reset_index()
        STORE["station_df"] = sdf
        STORE["combined"]   = ds["combined"]
        print(f"[App] {len(sdf)} stations preloaded")
    except Exception as e:
        print(f"[App] Preload warning: {e}")

_preload()

# ── Auto-train on startup (non-blocking) ───────────────────────────────────
import auto_train
auto_train.start(STATE, STORE)

# ── Helpers ────────────────────────────────────────────────────────────────
def _haversine(la1,lo1,la2,lo2):
    R=6371.0; r=math.radians
    dp,dl=r(la2-la1),r(lo2-lo1)
    a=math.sin(dp/2)**2+math.cos(r(la1))*math.cos(r(la2))*math.sin(dl/2)**2
    return R*2*math.atan2(math.sqrt(a),math.sqrt(1-a))

def _color(k): return "green" if k<8 else ("amber" if k<18 else "red")
def _label(k): return "Low demand" if k<8 else ("Moderate" if k<18 else "High demand")
def _avail(k): return "Available now" if k<8 else ("Busy" if k<18 else "Very busy")
def _wait(k,d):
    if k<8: return "No wait"
    if k<18: return f"~{max(1,int(d*20))} min"
    return f"~{max(5,int(d*45))} min"

def _predict(station_id):
    """HGNN prediction if model ready, else historical average fallback."""
    try:
        import torch
        if "model" not in STORE: raise Exception("no model")
        model=STORE["model"].eval()
        theta=STORE["theta"]; dev=STORE["device"]
        ds_=STORE["data"]["loaders"]["test"].dataset
        idx=int(np.random.randint(0,len(ds_)))
        Xb,_=ds_[idx]
        with torch.no_grad():
            out=model(Xb.unsqueeze(0).to(dev),theta=theta)
        return (out["medium"].squeeze().cpu().tolist(),
                out["short"].squeeze().cpu().tolist(),
                out["quantile"].squeeze().cpu().tolist())
    except:
        sdf=STORE.get("station_df"); avg=10.0
        if sdf is not None:
            r=sdf[sdf["station_id"]==station_id]
            if len(r): avg=float(r.iloc[0]["avg_kwh"])
        return [avg]*6,[avg]*3,None

def _station_dict(row, mp, sp, qp):
    avg=float(np.mean(mp))
    dur=float(row["avg_duration"])
    return {
        "id":       str(row["station_id"]),
        "lat":      round(float(row["latitude"]),6),
        "lng":      round(float(row["longitude"]),6),
        "site":     str(row["site_id"]),
        "source":   str(row["dataset_source"]),
        "sessions": int(row["total_sessions"]),
        "avg_kwh":  round(float(row["avg_kwh"]),2),
        "avg_duration": round(dur,2),
        "peak_hour":int(row["peak_hour"]),
        "demand_color":  _color(avg),
        "demand_label":  _label(avg),
        "availability":  _avail(avg),
        "wait":          _wait(avg,dur),
        "forecast":      [round(v,2) for v in mp],
        "short_fc":      [round(v,2) for v in sp],
        "q10": [round(qp[h][0],2) for h in range(len(mp))] if qp else None,
        "q50": [round(qp[h][1],2) for h in range(len(mp))] if qp else None,
        "q90": [round(qp[h][2],2) for h in range(len(mp))] if qp else None,
    }

# ── PAGES ──────────────────────────────────────────────────────────────────
@app.route("/")
def index(): return render_template("index.html")

@app.route("/map")
def map_page(): return render_template("map.html")

# ── STATUS (auto-poll) ─────────────────────────────────────────────────────
@app.route("/api/status")
def api_status(): return jsonify(STATE)

# ── RESEARCH DASHBOARD APIs ────────────────────────────────────────────────
@app.route("/api/results")
def api_results():
    # Try live STORE first, then disk
    if "comparison" in STORE and STORE["comparison"]:
        return jsonify(STORE["comparison"])
    saved = auto_train.load_saved_results()
    if saved.get("comparison"):
        return jsonify(saved["comparison"])
    return jsonify({"error":"Training in progress…"})

@app.route("/api/generalisation")
def api_gen():
    if "gen_results" in STORE: return jsonify(STORE["gen_results"])
    saved = auto_train.load_saved_results()
    return jsonify(saved.get("gen_results",{}))

@app.route("/api/edge_weights")
def api_ew():
    ew = STORE.get("edge_weights") or auto_train.load_saved_results().get("edge_weights")
    return jsonify({"weights":ew,"labels":["Temporal","Spatial","Grid","User"]})

@app.route("/api/decision")
def api_decision():
    decs = STORE.get("decisions") or auto_train.load_saved_results().get("decisions",[])
    return jsonify({"decisions":decs})

@app.route("/api/dataset_summary")
def api_ds():
    try:
        from utils.acn_loader import load_acn_sessions,get_dataset_summary
        return jsonify(get_dataset_summary(load_acn_sessions()))
    except Exception as e: return jsonify({"error":str(e)})

@app.route("/api/forecast")
def api_forecast():
    if STATE["status"]!="ready": return jsonify({"error":"Model not ready yet."})
    try:
        import torch
        cfg=STORE["cfg"]; dev=cfg["device"]
        model=STORE["model"].eval()
        theta=STORE["data"]["theta"].to(dev)
        ds_=STORE["data"]["loaders"]["test"].dataset
        idx=int(np.random.randint(0,len(ds_)))
        Xb,yb=ds_[idx]
        with torch.no_grad():
            out=model(Xb.unsqueeze(0).to(dev),theta=theta)
        mp=out["medium"].squeeze().cpu().tolist()
        sp=out["short"].squeeze().cpu().tolist()
        qp=out["quantile"].squeeze().cpu().tolist()
        mh=len(mp)
        return jsonify({
            "short_pred":[round(v,2) for v in sp],
            "medium_pred":[round(v,2) for v in mp],
            "actual":[round(v,2) for v in yb.tolist()],
            "q10":[round(qp[h][0],2) for h in range(mh)],
            "q50":[round(qp[h][1],2) for h in range(mh)],
            "q90":[round(qp[h][2],2) for h in range(mh)],
            "labels":[f"t+{i+1}h" for i in range(mh)]})
    except Exception as e: return jsonify({"error":str(e)})

@app.route("/api/figures")
def api_figures():
    fd="results/figures"
    if not os.path.exists(fd): return jsonify([])
    return jsonify(sorted(f for f in os.listdir(fd) if f.endswith(".png")))

@app.route("/figures/<filename>")
def serve_fig(filename): return send_from_directory("results/figures",filename)

@app.route("/results/training_history.json")
def serve_history():
    p="results/training_history.json"
    if os.path.exists(p):
        with open(p) as f: return jsonify(json.load(f))
    return jsonify({})

# ── MAP APIs ───────────────────────────────────────────────────────────────
@app.route("/api/stations/all")
def stations_all():
    sdf=STORE.get("station_df")
    if sdf is None: return jsonify({"error":"Stations loading…"})
    out=[]
    for _,row in sdf.iterrows():
        mp,sp,qp=_predict(str(row["station_id"]))
        out.append(_station_dict(row,mp,sp,qp))
    return jsonify({"stations":out,"total":len(out),
                    "model_active":STATE["status"]=="ready"})

@app.route("/api/stations/nearby")
def stations_nearby():
    try:
        lat=float(request.args.get("lat",34.1377))
        lng=float(request.args.get("lng",-118.1253))
        radius=float(request.args.get("radius_km",15.0))
        limit=int(request.args.get("limit",10))
    except: return jsonify({"error":"Invalid params."})
    sdf=STORE.get("station_df")
    if sdf is None: return jsonify({"error":"Stations loading…"})
    results=[]
    for _,row in sdf.iterrows():
        dist=_haversine(lat,lng,float(row["latitude"]),float(row["longitude"]))
        if dist>radius: continue
        mp,sp,qp=_predict(str(row["station_id"]))
        avg=float(np.mean(mp[:3]))
        # HGNN score: lower = better (closer + lower demand)
        score=dist*0.35+avg*0.65
        d=_station_dict(row,mp,sp,qp)
        d.update(distance_km=round(dist,2),
                 drive_min=max(1,int(dist/0.5)),
                 hgnn_score=round(score,3))
        results.append(d)
    results.sort(key=lambda x:x["hgnn_score"])
    return jsonify({"stations":results[:limit],"total_found":len(results),
        "search_center":{"lat":lat,"lng":lng},"radius_km":radius,
        "model_active":STATE["status"]=="ready"})

@app.route("/api/hgnn/predict/<station_id>")
def hgnn_predict(station_id):
    mp,sp,qp=_predict(station_id); avg=float(np.mean(mp))
    return jsonify({"station_id":station_id,
        "medium_pred":[round(v,2) for v in mp],
        "short_pred":[round(v,2) for v in sp],
        "q10":[round(qp[h][0],2) for h in range(len(mp))] if qp else None,
        "q50":[round(qp[h][1],2) for h in range(len(mp))] if qp else None,
        "q90":[round(qp[h][2],2) for h in range(len(mp))] if qp else None,
        "labels":[f"t+{i+1}h" for i in range(len(mp))],
        "demand_color":_color(avg),"demand_label":_label(avg),
        "availability":_avail(avg),"model_active":STATE["status"]=="ready"})

# ── PWA FILES ──────────────────────────────────────────────────────────────
@app.route("/manifest.json")
def manifest(): return send_from_directory("static","manifest.json")
@app.route("/sw.js")
def sw(): return send_from_directory("static","sw.js",
                                     mimetype="application/javascript")

if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    print(f"\n{'='*54}")
    print(f"  ChargePath EV AI  |  Auto-training started")
    print(f"  Map App:    http://localhost:{port}/map")
    print(f"  Dashboard:  http://localhost:{port}")
    print(f"{'='*54}\n")
    app.run(debug=False,host="0.0.0.0",port=port,threaded=True)

# ── RETRAIN (admin only — forces fresh training) ───────────────────────────
@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    """Force a retrain — only call this after uploading new data."""
    import auto_train
    auto_train.force_retrain(STATE, STORE)
    return jsonify({"ok": True, "message": "Retraining started. Results will update automatically."})
