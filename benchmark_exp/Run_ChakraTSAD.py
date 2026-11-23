"""TSB-AD runner for ChakraTSAD (YHat Labs), a hosted model served through an API.

ChakraTSAD is served at api.yhatlabs.com; this script sends each series with its clean-prefix length and writes the
per-point scores, then evaluates them with TSB-AD's official metrics. Nothing in the request identifies the dataset or
the file; the key is read from the environment at call time.

    pip install "pyarrow==20.0.0"                # long series are uploaded as parquet; newer pyarrow needs numpy 2, TSB-AD numpy<2
    export YHAT_API_KEY=...                      # evaluation key (scoped to the benchmark model id)
    python benchmark_exp/Run_ChakraTSAD.py --set U --dataset_dir Datasets/TSB-AD-U/ --file_list Datasets/File_List/TSB-AD-U-Eva.csv --save
    python benchmark_exp/Run_ChakraTSAD.py --set M --dataset_dir Datasets/TSB-AD-M/ --file_list Datasets/File_List/TSB-AD-M-Eva.csv --save

A rerun reuses the scores already saved in <score_dir>/ChakraTSAD/ and calls the API only for the series still missing.
run_ChakraTSAD_Semisupervised(data_train, data_test) is the runner in TSB-AD's semi-supervised form, for use in other harnesses.

Env: YHAT_API_BASE (default https://api.yhatlabs.com), YHAT_MODEL (default chakra-tsad-tsb, the frozen benchmark
configuration), YHAT_TIMEOUT_S (default 900). A series is sent inline as JSON up to INLINE_MAX_CELLS values; a longer one is
uploaded once as parquet (POST /v1/uploads, columns = channels) and referenced by id; a call past the API's direct-call size
is sent as an async job ("async": true) and polled at GET /v1/jobs/{id}. Output: <score_dir>/ChakraTSAD/<file>.npy per
series and <score_dir>/ChakraTSAD/Uni_ChakraTSAD.csv (or Multi_) with the official columns (file, Time, AUC-PR, AUC-ROC,
VUS-PR, VUS-ROC, Standard-F1, PA-F1, Event-based-F1, R-based-F1, Affiliation-F). Time = client wall time including the
network round trip (and, for a job, the queue wait).
"""

import argparse
import gzip
import hashlib
import io
import json
import logging
import os
import random
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank

VERSION = "1.3"
NAME = "ChakraTSAD"
API_BASE = os.environ.get("YHAT_API_BASE", "https://api.yhatlabs.com").rstrip("/")
MODEL = os.environ.get("YHAT_MODEL", "chakra-tsad-tsb")
TIMEOUT_S = float(os.environ.get("YHAT_TIMEOUT_S", "900"))
MAX_TRIES = 12  # with the 120 s cap: about 12 minutes of retries, enough to outlast a busy key
MAX_DELAY_S = 120.0
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
INLINE_MAX_CELLS = (
    500_000  # the API takes 1,000,000 values / 16 MB inline; JSON of 500k floats stays under both
)
ASYNC_MIN_CELLS = (
    10_000_000  # the API's built-in direct-call size in full mode; larger calls must be async
)
POLL_S = 10.0


def api_key() -> str:
    key = os.environ.get("YHAT_API_KEY")
    if not key:
        raise SystemExit("YHAT_API_KEY is not set")
    return key


def request(
    method: str, url: str, body: dict | bytes | None = None, headers: dict | None = None
) -> dict:
    # bounded retries on transient statuses and network errors; JSON bodies go gzipped
    h = {
        "Authorization": f"Bearer {api_key()}",
        "User-Agent": f"tsb-ad-chakratsad/{VERSION}",
        **(headers or {}),
    }
    raw: bytes | None
    if isinstance(body, dict):
        raw = gzip.compress(json.dumps(body).encode())
        h.update({"Content-Type": "application/json", "Content-Encoding": "gzip"})
    else:
        raw = body
    for attempt in range(1, MAX_TRIES + 1):
        req = urllib.request.Request(url, data=raw, headers=h, method=method)
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                text = r.read().decode()
                return json.loads(text) if text else {}
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_STATUS or attempt == MAX_TRIES:
                raise
            delay = min(MAX_DELAY_S, 2.0**attempt + random.random())
            try:  # the server's Retry-After, when it gives one, wins
                delay = min(MAX_DELAY_S, max(1.0, float(e.headers.get("Retry-After") or "")))
            except ValueError:
                pass
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == MAX_TRIES:
                raise
            delay = min(MAX_DELAY_S, 2.0**attempt + random.random())
        logging.warning("retry %d in %.0fs", attempt, delay)
        time.sleep(delay)
    raise RuntimeError("unreachable")


def check_pyarrow() -> None:
    # series past INLINE_MAX_CELLS values are uploaded as parquet: stop at the start, not series by series
    try:
        import pyarrow as pa

        pa.table({"c0": np.zeros(2)})
    except Exception as e:  # noqa: BLE001 - any failure to load means the long series cannot be sent
        raise SystemExit(
            f"pyarrow is needed to send long series and does not load ({e}). "
            'With TSB-AD\'s numpy<2: pip install "pyarrow==20.0.0"'
        )


def post_json(url: str, body: dict) -> dict:
    return request("POST", url, body)


def upload_series(data: np.ndarray) -> dict:
    # a long series goes up once as a parquet table (columns = channels, rows = time steps), stored under its content hash;
    # the call then references it as {"format": "table", "id": ...}
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({f"c{j}": data[:, j] for j in range(data.shape[1])})
    buf = io.BytesIO()
    pq.write_table(table, buf)
    blob = buf.getvalue()
    meta = post_json(
        f"{API_BASE}/v1/uploads",
        {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob), "format": "parquet"},
    )
    if not meta.get("exists"):
        up = meta["upload"]
        url = up["url"] if up["url"].startswith("http") else API_BASE + up["url"]
        headers = {**up.get("headers", {}), "Content-Type": "application/vnd.apache.parquet"}
        if up.get("auth", True):
            request(up.get("method", "PUT"), url, blob, headers)
        else:  # a signed storage link: no API key on it
            req = urllib.request.Request(
                url, data=blob, headers=headers, method=up.get("method", "PUT")
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                r.read()
    return {"format": "table", "id": meta["id"]}


def wait_job(job_id: str) -> dict:
    while True:
        job = request("GET", f"{API_BASE}/v1/jobs/{job_id}")
        if job["status"] == "succeeded":
            return job["result"]
        if job["status"] in ("failed", "canceled"):
            raise RuntimeError(f"job {job_id} {job['status']}: {job.get('error')}")
        time.sleep(POLL_S)


def detect(data: np.ndarray, train_len: int) -> np.ndarray:
    n, c = data.shape
    item: dict = {"id": "s0", "train_len": int(train_len)}
    if n * c > INLINE_MAX_CELLS:
        item["targets"] = upload_series(data)
    elif c == 1:
        item["values"] = data[:, 0].tolist()
    else:
        item["targets"] = {f"c{j}": data[:, j].tolist() for j in range(c)}
    body: dict = {"model": MODEL, "series": [item]}
    if n * c > ASYNC_MIN_CELLS:
        body["async"] = True
    try:
        reply = post_json(f"{API_BASE}/v1/series/detect", body)
    except urllib.error.HTTPError as e:
        if e.code != 413 or body.get("async"):
            raise
        reply = post_json(f"{API_BASE}/v1/series/detect", {**body, "async": True})  # async_required
    if reply.get("object") == "job" or ("status" in reply and "scores" not in reply):
        reply = wait_job(reply["id"])
    scores = np.asarray(reply["scores"][0]["scores"], dtype=np.float64)
    if len(scores) != n:
        raise RuntimeError(f"API returned {len(scores)} scores for {n} points")
    return scores


def run_ChakraTSAD_Semisupervised(data_train, data_test, **HP):
    """TSB-AD's semi-supervised runner interface (benchmark_exp/README.md): data_train is the anomaly-free prefix and data_test
    the full series, as Run_Detector_U/M.py pass them; returns one score per point of data_test. A data_test that does not start
    with data_train is scored after it, and only its own points are returned. The model takes no hyper-parameters (HP unused)."""
    train = np.asarray(data_train, dtype=float)
    test = np.asarray(data_test, dtype=float)
    train = train.reshape(-1, 1) if train.ndim == 1 else train
    test = test.reshape(-1, 1) if test.ndim == 1 else test
    n = len(train)
    if len(test) >= n and np.array_equal(test[:n], train):
        return detect(test, n)
    return detect(np.vstack([train, test]), n)[n:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="U", choices=["U", "M"])
    ap.add_argument("--dataset_dir", default="Datasets/TSB-AD-U/")
    ap.add_argument("--file_list", default="Datasets/File_List/TSB-AD-U-Eva.csv")
    ap.add_argument("--score_dir", default="eval/score")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    check_pyarrow()
    out = os.path.join(a.score_dir, NAME)
    os.makedirs(out, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(out, f"000_run_{NAME}.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    files = pd.read_csv(a.file_list)["file_name"].values
    if a.limit:
        files = files[: a.limit]
    rows = []
    for filename in files:
        stem = filename.split(".")[0]
        df = pd.read_csv(os.path.join(a.dataset_dir, filename)).dropna()
        data = df.iloc[:, :-1].values.astype(float)
        label = df["Label"].astype(int).to_numpy()
        train_len = int(stem.split("_")[-3])
        window = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
        npy = os.path.join(out, stem + ".npy")
        t0 = time.time()
        if os.path.exists(npy):
            score = np.load(npy)
        else:
            try:
                score = run_ChakraTSAD_Semisupervised(data[:train_len], data)
            except Exception as e:  # noqa: BLE001 — recorded, the run continues
                logging.error("At %s: %s", filename, e)
                print(filename, "ERROR", e, flush=True)
                continue
            np.save(npy, score)
        run_time = time.time() - t0
        logging.info(
            "Success at %s | Time cost: %.3fs at length %d", filename, run_time, len(label)
        )
        if a.save:
            m = get_metrics(score, label, slidingWindow=window)
            rows.append([filename, run_time, *m.values()])
            cols = ["file", "Time", *m.keys()]
            pd.DataFrame(rows, columns=cols).to_csv(
                os.path.join(out, f"{'Uni' if a.set == 'U' else 'Multi'}_{NAME}.csv"), index=False
            )
            print(filename, {k: round(v, 4) for k, v in m.items()}, flush=True)


if __name__ == "__main__":
    main()
