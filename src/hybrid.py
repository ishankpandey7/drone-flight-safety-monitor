"""
STEP 5b --- HYBRID: forecaster ki physics + Mahalanobis ka math.
================================================================

Ab tak jo pata chala:

  * Mahalanobis baseline sabse accha hai (AUC 0.90)
  * Forecaster ne seekha hai ki drone ka agla pal kaisa hona chahiye,
    lekin uske 19 errors ko main RMS se ek number bana raha tha

RMS me ek badi kami hai: wo errors ke AAPSI RISHTE ignore kar deta hai.

    Asli baat ye hai ki healthy flight me bhi errors ek DHAANCHE me aate
    hain. Jaise: tez maneuver me gyro aur accel dono ka error saath badhta
    hai --- ye normal hai. Par agar sirf gyro ka error badhe aur accel ka
    na badhe, to kuch gadbad hai.

    RMS dono ko ek jaisa dekhta hai. Mahalanobis farq kar sakta hai,
    kyunki wo covariance jaanta hai.

Toh hybrid ye hai:
    forecaster se 19-dim error vector nikalo   (seekhi hui physics)
    us par Mahalanobis lagao                   (sahi multivariate math)

Model dobara train nahi hota. Sirf errors ko jodne ka tareeka badalta hai.
"""

import json
from pathlib import Path

import numpy as np
import torch

import common
from common import FS
from forecaster import (LSTMForecaster, downsample, to_windows,
                        HIST, HORIZON, SEQ_STRIDE, DS)
from baseline import MahalanobisDetector, windowize

ROOT = Path(__file__).resolve().parent.parent

ck = torch.load(ROOT / "outputs" / "forecaster.pt", weights_only=False)
model = LSTMForecaster(ck["n_ch"])
model.load_state_dict(ck["state"])
model.eval()
mu, sd, n_ch = ck["mu"], ck["sd"], ck["n_ch"]

d = np.load(ROOT / "data" / "processed" / "flights.npz", allow_pickle=True)
meta = json.load(open(ROOT / "data" / "processed" / "meta.json"))
fmeta = [m for m in meta if m["split"] == "test_faulty"]


def prep(X):
    return to_windows((downsample(X) - mu) / sd)


@torch.no_grad()
def pred_err(X):
    """(N, T, C) -> (N, W, C)  har window me har channel ka prediction error."""
    H, F = prep(X)
    N, W = H.shape[:2]
    xh = torch.tensor(H.reshape(-1, HIST, n_ch).astype(np.float32))
    yf = torch.tensor(F.reshape(-1, HORIZON, n_ch).astype(np.float32))
    out = []
    for i in range(0, len(xh), 512):
        p = model(xh[i:i + 512])
        out.append(((p - yf[i:i + 512]) ** 2).mean(dim=1).numpy())
    return np.concatenate(out).reshape(N, W, n_ch)


print()
print("Forecaster ke prediction errors nikal rahe hain...")
E = {k: pred_err(d[k]) for k in ["train", "val", "test_healthy", "test_faulty"]}
W = E["train"].shape[1]
print(f"  {E['train'].shape[0]} train flights x {W} windows x {n_ch} channels")

# window ka end time
span_s = (HIST + HORIZON) / (FS / DS)
wt = np.array([(s * SEQ_STRIDE) / (FS / DS) + span_s for s in range(W)])

# ---------------------------------------------------------------- variant 1
# sirf forecaster errors -> Mahalanobis
det = MahalanobisDetector().fit(E["train"].reshape(-1, n_ch))


def sc(F):
    n, w, c = F.shape
    return det.score(F.reshape(-1, c)).reshape(n, w)


res_a = common.report("HYBRID A  --  forecaster errors -> Mahalanobis",
                      sc(E["val"]), sc(E["test_healthy"]), sc(E["test_faulty"]),
                      fmeta, wt)

# ---------------------------------------------------------------- variant 2
# baseline ke window stats + forecaster errors, dono ek saath
print("  Baseline features bhi jod rahe hain...")
Fb = {k: windowize(d[k]) for k in E}
Wb = min(W, Fb["train"].shape[1])


def combo(k):
    # dono ko ek hi lambai pe kaat lo (baseline window j ka end time
    # forecaster window j se 1 second pehle hai --- itna farq detection
    # lag me kuch nahi badalta)
    return np.concatenate([Fb[k][:, :Wb, :], E[k][:, :Wb, :]], axis=-1)


C = {k: combo(k) for k in E}
det2 = MahalanobisDetector().fit(C["train"].reshape(-1, C["train"].shape[-1]))


def sc2(F):
    n, w, c = F.shape
    return det2.score(F.reshape(-1, c)).reshape(n, w)


res_b = common.report(
    f"HYBRID B  --  baseline stats (38) + forecaster errors ({n_ch}) -> Mahalanobis",
    sc2(C["val"]), sc2(C["test_healthy"]), sc2(C["test_faulty"]), fmeta, wt[:Wb])

# ---------------------------------------------------------------- save
for tag, res, s in [("hybrid_a", res_a, (sc(E["val"]), sc(E["test_healthy"]), sc(E["test_faulty"]), wt)),
                    ("hybrid_b", res_b, (sc2(C["val"]), sc2(C["test_healthy"]), sc2(C["test_faulty"]), wt[:Wb]))]:
    np.savez(ROOT / "outputs" / f"{tag}_scores.npz",
             val=s[0], test_healthy=s[1], test_faulty=s[2],
             window_times=s[3], threshold=res["threshold"])
    with open(ROOT / "outputs" / f"{tag}_result.json", "w") as f:
        json.dump(res, f, indent=1)

# ---------------------------------------------------------------- summary
KINDS = ["motor_degradation", "prop_damage", "battery_sag", "imu_drift"]
rows = [("baseline", "Mahalanobis (stats)"),
        ("lstm", "LSTM autoencoder"),
        ("forecaster", "LSTM forecaster (RMS)"),
        ("hybrid_a", "Hybrid A (fc -> Mahal)"),
        ("hybrid_b", "Hybrid B (stats + fc)")]

print()
print("=" * 96)
print("  SAB KUCH EK SAATH   (deployable --- threshold val se)")
print("=" * 96)
print(f"  {'detector':<26}{'recall':>9}{'FA':>7}   " + "".join(f"{k[:9]:>11}" for k in KINDS))
print("  " + "-" * 92)
for tag, name in rows:
    p = ROOT / "outputs" / f"{tag}_result.json"
    if not p.exists():
        continue
    r = json.load(open(p))
    print(f"  {name:<26}{100 * r['recall']:>8.0f}%{100 * r['false_alarm']:>6.0f}%   " +
          "".join(f"{100 * r['per_kind'][k]:>10.0f}%" for k in KINDS))
print("  " + "-" * 92)
print()
