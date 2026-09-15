"""
STEP 4b --- LSTM FORECASTER  (autoencoder ki jagah)
===================================================

Autoencoder kyun haara, ek line me:

    Usse input dobara banana tha --- yani JAWAB uske saamne rakha tha.
    Voltage 14.2 dikha, voltage 14.2 likh diya. Copy. Error zero.
    Usne kabhi ye poocha hi nahi ki "kya 14.2 normal hai?"

Forecaster me jawab chheen liya jaata hai:

    Pichhle 8 second do. Agla 1 second PREDICT karo.
    Model ne healthy flights se physics seekhi hai:
       "is load pe voltage itni raftaar se girta hai"
       "is altitude command pe PWM itna uthta hai"
       "is roll rate pe agla roll angle itna hoga"
    Fault me ye rishte tootte hain -> prediction galat -> error.

    Aur model copy nahi kar sakta, kyunki jo predict karna hai wo usne
    dekha hi nahi hai.

Yahi wajah hai ki dynamical systems (drone, engine, turbine, dil ki dhadkan)
me forecasting almost hamesha reconstruction se behtar anomaly detector hai.

Sab kuch baaki BILKUL same rakha hai --- wahi 8s window, wahi 2s stride,
wahi threshold logic, wahi RMS aggregation. Sirf architecture badla hai,
taaki farq sirf usi ka ho.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import common
from common import FS, WIN_S, STRIDE_S

ROOT = Path(__file__).resolve().parent.parent

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

DS         = 5                          # 50 Hz -> 10 Hz
HIST       = int(WIN_S * FS / DS)       # 80 steps = 8 second ka itihaas
HORIZON    = 10                         # agla 1 second predict karo
SEQ_STRIDE = int(STRIDE_S * FS / DS)    # 20 steps = 2 second
HIDDEN     = 96
LAYERS     = 2
EPOCHS     = 250
BATCH      = 256
LR         = 1e-3
PATIENCE   = 20

# NOTE: ek window ab HIST + HORIZON = 90 steps (9 second) leti hai.
# Baseline 8 second leta tha. Fair rehne ke liye hum HIST ko 70 kar ke
# 70+10 = 80 bhi kar sakte the, par 1 second ka farq detection lag me
# kuch nahi badalta aur history chhoti karne se model kamzor hota. Isliye
# HIST 80 rakha aur window_times me 1 second ka extra maan liya gaya hai.


def downsample(X, k=DS):
    N, T, C = X.shape
    T2 = (T // k) * k
    return X[:, :T2, :].reshape(N, T2 // k, k, C).mean(axis=2)


def to_windows(X):
    """(N, T, C) -> hist (N, W, HIST, C), future (N, W, HORIZON, C)"""
    N, T, C = X.shape
    span = HIST + HORIZON
    starts = list(range(0, T - span + 1, SEQ_STRIDE))
    H = np.stack([X[:, s:s + HIST, :] for s in starts], axis=1)
    F = np.stack([X[:, s + HIST:s + span, :] for s in starts], axis=1)
    return H, F


class LSTMForecaster(nn.Module):
    def __init__(self, n_ch, hidden=HIDDEN, layers=LAYERS, horizon=HORIZON):
        super().__init__()
        self.horizon, self.n_ch = horizon, n_ch
        self.lstm = nn.LSTM(n_ch, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, horizon * n_ch),
        )

    def forward(self, x):                      # (B, HIST, C)
        _, (h, _) = self.lstm(x)
        y = self.head(h[-1])                   # (B, HORIZON*C)
        return y.view(-1, self.horizon, self.n_ch)


def main():
    d = np.load(ROOT / "data" / "processed" / "flights.npz", allow_pickle=True)
    meta = json.load(open(ROOT / "data" / "processed" / "meta.json"))
    fmeta = [m for m in meta if m["split"] == "test_faulty"]
    CH = [str(c) for c in d["channels"]]

    Xtr = downsample(d["train"])
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-8

    def prep(X):
        H, F = to_windows((downsample(X) - mu) / sd)
        return H.astype(np.float32), F.astype(np.float32)

    Htr, Ftr = prep(d["train"])
    Hva, Fva = prep(d["val"])
    Hth, Fth = prep(d["test_healthy"])
    Htf, Ftf = prep(d["test_faulty"])
    n_ch = Htr.shape[-1]

    print()
    print(f"  har window : {HIST} steps ka itihaas (8s)  ->  {HORIZON} steps ka future (1s)")
    print(f"  train      : {Htr.shape[0]} flights x {Htr.shape[1]} windows")

    def flat(A):
        return torch.tensor(A.reshape(-1, A.shape[2], n_ch))

    xtr, ytr = flat(Htr), flat(Ftr)
    xva, yva = flat(Hva), flat(Fva)

    model = LSTMForecaster(n_ch)
    print(f"  parameters : {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    lossfn = nn.MSELoss()

    print()
    print("Training (sirf healthy flights pe)...")
    best, best_state, bad = float("inf"), None, 0
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(xtr))
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            j = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossfn(model(xtr[j]), ytr[j])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(j)
        tr_loss = tot / len(perm)

        model.eval()
        with torch.no_grad():
            va_loss = sum(lossfn(model(xva[i:i + BATCH]), yva[i:i + BATCH]).item() * len(xva[i:i + BATCH])
                          for i in range(0, len(xva), BATCH)) / len(xva)
        sched.step(va_loss)

        if va_loss < best - 1e-6:
            best, bad = va_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

        if ep % 10 == 0 or ep == 1 or bad >= PATIENCE:
            print(f"  epoch {ep:3d}  train {tr_loss:.5f}   val {va_loss:.5f}"
                  f"  {'*' if bad == 0 else ' '}  ({time.time() - t0:.0f}s)")
        if bad >= PATIENCE:
            print(f"  early stop")
            break

    model.load_state_dict(best_state)
    model.eval()
    print(f"  best val loss : {best:.5f}")

    # -------------------------------------------------- scoring
    @torch.no_grad()
    def per_channel_err(H, F):
        N, W = H.shape[:2]
        xh, yf = flat(H), flat(F)
        out = []
        for i in range(0, len(xh), 512):
            p = model(xh[i:i + 512])
            out.append(((p - yf[i:i + 512]) ** 2).mean(dim=1).numpy())
        return np.concatenate(out).reshape(N, W, n_ch)

    E_tr = per_channel_err(Htr, Ftr)
    e_mu = E_tr.reshape(-1, n_ch).mean(0)
    e_sd = E_tr.reshape(-1, n_ch).std(0) + 1e-8

    def score(H, F):
        Z = (per_channel_err(H, F) - e_mu) / e_sd
        return np.sqrt((Z ** 2).mean(-1))            # RMS --- wahi jo baseline karta hai

    Sv = score(Hva, Fva)
    Sh = score(Hth, Fth)
    Sf = score(Htf, Ftf)

    # window ka END time: HIST khatam + HORIZON poora hone ke baad hi faisla
    span_s = (HIST + HORIZON) / (FS / DS)
    wt = np.array([(s * SEQ_STRIDE) / (FS / DS) + span_s for s in range(Sv.shape[1])])

    res = common.report("LSTM FORECASTER  --  8s itihaas se 1s ka future",
                        Sv, Sh, Sf, fmeta, wt)

    # -------------------------------------------------- comparison
    b = json.load(open(ROOT / "outputs" / "baseline_result.json"))
    a = json.load(open(ROOT / "outputs" / "lstm_result.json"))
    KINDS = ["motor_degradation", "prop_damage", "battery_sag", "imu_drift"]
    print("  " + "-" * 74)
    print(f"  {'':<22}{'BASELINE':>14}{'AUTOENCODER':>15}{'FORECASTER':>15}")
    print("  " + "-" * 74)
    print(f"  {'recall':<22}{100 * b['recall']:>13.0f}%{100 * a['recall']:>14.0f}%{100 * res['recall']:>14.0f}%")
    print(f"  {'false alarm':<22}{100 * b['false_alarm']:>13.0f}%{100 * a['false_alarm']:>14.0f}%{100 * res['false_alarm']:>14.0f}%")
    print("  " + "-" * 74)
    for k in KINDS:
        print(f"  {k:<22}{100 * b['per_kind'][k]:>13.0f}%"
              f"{100 * a['per_kind'][k]:>14.0f}%{100 * res['per_kind'][k]:>14.0f}%")
    print("  " + "-" * 74)
    print()

    torch.save({"state": best_state, "mu": mu, "sd": sd, "n_ch": n_ch,
                "e_mu": e_mu, "e_sd": e_sd, "hist": HIST, "horizon": HORIZON},
               ROOT / "outputs" / "forecaster.pt")
    np.savez(ROOT / "outputs" / "forecaster_scores.npz",
             val=Sv, test_healthy=Sh, test_faulty=Sf,
             window_times=wt, threshold=res["threshold"])
    with open(ROOT / "outputs" / "forecaster_result.json", "w") as f:
        json.dump(res, f, indent=1)
    print("  saved -> outputs/forecaster.pt")


if __name__ == "__main__":
    main()
