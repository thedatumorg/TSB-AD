"""Convert the OpenTelemetry AIOps benchmark feature tables (Zenodo 10.5281/zenodo.19462083)
into TSB-AD-M series, one CSV per (signal, target service).

Labeling rule: a 60-second window is anomalous when a run in the campaign manifest targets
that service (for cascade runs, either of the two services named in the cascade) and the
window timestamp falls in [fault_start, fault_end]. The 5-minute cooldown after each fault
(fault_end, cooldown_end] is labeled normal by default (COOLDOWN=normal). COOLDOWN=drop removes
those windows instead, which is the cooldown exclusion used in the article's evaluation, but it
also joins consecutive fault windows of the same service into one block because the campaign
ran all faults on a service back to back. Windows during runs that target other services stay
normal. Series with no anomalous window are skipped.

Output name: [index]_OTel_id_[id]_WebService_tr_[train]_1st_[first anomaly].csv
Columns: features 0..N-1, then Label. tr = number of windows before FI_START (the baseline).

usage: convert_otel_to_tsbad_m.py FEATURES_DIR MANIFEST_JSON TIMESTAMPS_ENV OUT_DIR [START_INDEX] [COOLDOWN]
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

SRC, MANIFEST, TS_ENV, OUT = (Path(a) for a in sys.argv[1:5])
START_INDEX = int(sys.argv[5]) if len(sys.argv) > 5 else 201
COOLDOWN = sys.argv[6] if len(sys.argv) > 6 else "normal"
META = {"service", "timestamp", "phase", "label", "fault_type", "run_id", "rep"}

CASCADES = {  # from campaign_config.py, both services named in each cascade
    "cascade-otel-catalog-latency-recmd-cpu": ("productcatalogservice", "recommendationservice"),
    "cascade-otel-checkout-cpu-payment-mem": ("checkoutservice", "paymentservice"),
    "cascade-ss-catalogue-latency-orders-cpu": ("catalogue", "orders"),
    "cascade-ss-frontend-latency-carts-mem": ("front-end", "carts"),
}
# (signal, testbed, target service) -> service name in the parquet
SERIES = [
    ("trace", "otel-demo", "checkoutservice", "checkoutservice"),
    ("trace", "otel-demo", "frontendproxy", "frontendproxy"),
    ("trace", "otel-demo", "paymentservice", "paymentservice"),
    ("trace", "otel-demo", "productcatalogservice", "productcatalogservice"),
    ("trace", "otel-demo", "recommendationservice", "recommendationservice"),
    ("trace", "sockshop", "carts", "carts"),
    ("metrics", "otel-demo", "checkoutservice", "checkoutservice"),
    ("metrics", "otel-demo", "frontendproxy", "frontendproxy"),
    ("metrics", "otel-demo", "paymentservice", "paymentservice"),
    ("metrics", "otel-demo", "productcatalogservice", "productcatalogservice"),
    ("metrics", "otel-demo", "recommendationservice", "recommendationservice"),
    ("metrics", "sockshop", "carts", "carts"),
    ("metrics", "sockshop", "catalogue", "catalogue"),
    ("metrics", "sockshop", "front-end", "front-end"),
    ("metrics", "sockshop", "orders", "orders"),
    ("log", "otel-demo", "checkoutservice", "otel-demo-checkoutservice"),
    ("log", "otel-demo", "paymentservice", "otel-demo-paymentservice"),
    ("log", "otel-demo", "recommendationservice", "otel-demo-recommendationservice"),
    ("log", "sockshop", "carts", "carts"),
    ("log", "sockshop", "catalogue", "catalogue"),
    ("log", "sockshop", "front-end", "front-end"),
]
FILES = {"trace": "trace_features.parquet", "metrics": "metrics_features.parquet", "log": "log_features.parquet"}

env = dict(l.strip().split("=", 1) for l in open(TS_ENV) if "=" in l)
FI_START = pd.Timestamp(env["FI_START"])
runs = pd.DataFrame(json.load(open(MANIFEST))["runs"])
runs = runs[runs.status == "completed"].copy()
for c in ("fault_start", "fault_end", "cooldown_end"):
    runs[c] = pd.to_datetime(runs[c], utc=True)
runs["targets"] = [
    CASCADES[r.cascade_name] if r.is_cascade else (r.service,) for r in runs.itertuples()
]

OUT.mkdir(parents=True, exist_ok=True)
tables = {s: pd.read_parquet(SRC / f) for s, f in FILES.items()}
idx, sid, rows = START_INDEX, 1, []
for signal, testbed, target, name in SERIES:
    df = tables[signal]
    d = df[df.service == name].sort_values("timestamp").reset_index(drop=True)
    feats = [c for c in df.columns if c not in META]
    ts = d["timestamp"]
    lab = np.zeros(len(d), dtype=int)
    drop = np.zeros(len(d), dtype=bool)
    for r in runs[runs.testbed == testbed].itertuples():
        if target not in r.targets:
            continue
        lab |= ((ts >= r.fault_start) & (ts <= r.fault_end)).to_numpy()
        drop |= ((ts > r.fault_end) & (ts <= r.cooldown_end)).to_numpy()
    n_runs = sum(target in t for t in runs[runs.testbed == testbed].targets)
    keep = ~drop if COOLDOWN == "drop" else np.ones(len(d), dtype=bool)
    d, lab = d[keep].reset_index(drop=True), lab[keep]
    if not lab.any():
        print("skip (no anomalous window):", signal, testbed, target)
        continue
    first = int(np.argmax(lab == 1))
    tr = int((d["timestamp"] < FI_START).sum())
    X = d[feats].astype(float)
    X.columns = [str(i) for i in range(len(feats))]
    X["Label"] = lab
    fname = f"{idx:03d}_OTel_id_{sid}_WebService_tr_{tr}_1st_{first}.csv"
    X.to_csv(OUT / fname, index=False)
    segs = int((np.diff(np.r_[0, lab, 0]) == 1).sum())
    rows.append(dict(file_name=fname, signal=signal, testbed=testbed, service=target,
                     n=len(lab), n_features=len(feats), train_index=tr, first_anomaly=first,
                     runs_targeting=n_runs, anomaly_segments=segs, anomaly_windows=int(lab.sum()),
                     prevalence=round(lab.mean(), 4), cooldown_windows=int(drop.sum()), cooldown_mode=COOLDOWN,
                     nan_cells=int(X.isna().sum().sum()),
                     size_kb=round((OUT / fname).stat().st_size / 1024, 1)))
    idx += 1; sid += 1
man = pd.DataFrame(rows)
man.to_csv(OUT / "OTel_series_summary.csv", index=False)
pd.set_option("display.width", 250)
print(man.to_string())
print("series:", len(man), "total MB:", round(man.size_kb.sum() / 1024, 1))
