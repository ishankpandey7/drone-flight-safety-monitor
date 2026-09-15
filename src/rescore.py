"""
Trained model ko dobara score karo --- alag alag aggregator ke saath.

Model dobara train karne ki zaroorat NAHI hai. Model theek tha; galti sirf
is baat me thi ki 19 channels ke z-scores ko EK number me kaise badla jaaye.

Teen tarike:

  mean(z)  --- jo maine pehle kiya tha
               [21, 0, 0, ..., 0]  ->  1.1     <- ek channel ka chillana dub gaya
               [10, 10, 10, 0...]  ->  1.6

  rms(z)   --- squared z ka average, phir sqrt.  Yahi math Mahalanobis
               bhi karta hai (squared z ka SUM). Squaring bade deviation
               ko bhaari bana deti hai, isliye dilution nahi hota.
               [21, 0, 0, ..., 0]  ->  4.8
               [10, 10, 10, 0...]  ->  4.0     <- ab ordering ulti ho gayi, sahi tarah se

  max(z)   --- sabse zyada sensitive, par sabse shor bhara bhi

RMS kyun chuna: ye test dekh kar nahi chuna. Ye baseline ka HI math hai.
Aur humein design se pata hai ki har fault ek YA DO channels me bolta hai,
saare 19 me nahi --- aise me average lena hamesha galat hai.
"""

import json
from pathlib import Path

import numpy as np
import torch

import common
from lstm_model import LSTMAutoencoder, downsample, to_sequences, SEQ, DS

ROOT = Path(__file__).resolve().parent.parent

ck = torch.load(ROOT / "outputs" / "lstm_model.pt", weights_only=False)
model = LSTMAutoencoder(ck["n_ch"])
model.load_state_dict(ck["state"])
model.eval()
mu, sd, n_ch = ck["mu"], ck["sd"], ck["n_ch"]

d = np.load(ROOT / "data" / "processed" / "flights.npz", allow_pickle=True)
meta = json.load(open(ROOT / "data" / "processed" / "meta.json"))
fmeta = [m for m in meta if m["split"] == "test_faulty"]
T_raw = d["train"].shape[1]
wt = common.window_times(T_raw)


def prep(X):
    return to_sequences((downsample(X) - mu) / sd).astype(np.float32)


@torch.no_grad()
def per_channel_err(S):
    N, W = S.shape[:2]
    flat = torch.tensor(S.reshape(-1, SEQ, n_ch))
    out = []
    for i in range(0, len(flat), 512):
        xb = flat[i:i + 512]
        out.append(((model(xb) - xb) ** 2).mean(dim=1).numpy())
    return np.concatenate(out).reshape(N, W, n_ch)


print()
print("Per-channel errors nikal rahe hain...")
E_tr = per_channel_err(prep(d["train"]))
E_va = per_channel_err(prep(d["val"]))
E_th = per_channel_err(prep(d["test_healthy"]))
E_tf = per_channel_err(prep(d["test_faulty"]))

e_mu = E_tr.reshape(-1, n_ch).mean(0)
e_sd = E_tr.reshape(-1, n_ch).std(0) + 1e-8


def z(E):
    return (E - e_mu) / e_sd


AGG = {
    "mean(z)  [purana]": lambda Z: Z.mean(-1),
    "rms(z)   [naya]":   lambda Z: np.sqrt((Z ** 2).mean(-1)),
    "max(z)":            lambda Z: Z.max(-1),
}

results = {}
for name, fn in AGG.items():
    res = common.report(f"LSTM AUTOENCODER  --  aggregator: {name}",
                        fn(z(E_va)), fn(z(E_th)), fn(z(E_tf)), fmeta, wt)
    results[name] = res
    if name.startswith("rms"):
        np.savez(ROOT / "outputs" / "lstm_scores.npz",
                 val=fn(z(E_va)), test_healthy=fn(z(E_th)),
                 test_faulty=fn(z(E_tf)), window_times=wt,
                 threshold=res["threshold"])
        with open(ROOT / "outputs" / "lstm_result.json", "w") as f:
            json.dump(res, f, indent=1)

# ------------------------------------------------------------------ summary
b = json.load(open(ROOT / "outputs" / "baseline_result.json"))
KINDS = ["motor_degradation", "prop_damage", "battery_sag", "imu_drift"]

print()
print("=" * 96)
print("  SAB KUCH EK SAATH")
print("=" * 96)
hdr = f"  {'':<22}{'BASELINE':>12}" + "".join(f"{n.split()[0]:>14}" for n in AGG)
print(hdr)
print("  " + "-" * (len(hdr) - 2))
print(f"  {'recall':<22}{100 * b['recall']:>11.0f}%" +
      "".join(f"{100 * results[n]['recall']:>13.0f}%" for n in AGG))
print(f"  {'false alarm':<22}{100 * b['false_alarm']:>11.0f}%" +
      "".join(f"{100 * results[n]['false_alarm']:>13.0f}%" for n in AGG))
print("  " + "-" * (len(hdr) - 2))
for k in KINDS:
    print(f"  {k:<22}{100 * b['per_kind'][k]:>11.0f}%" +
          "".join(f"{100 * results[n]['per_kind'][k]:>13.0f}%" for n in AGG))
print("  " + "-" * (len(hdr) - 2))
print()
