"""
STEP 4 --- LSTM Autoencoder.
============================

IDEA (ek line me):
    Model ko 8-second ki flight clip do, usko 24 numbers me nichodne ko kaho,
    phir unhi 24 se poori clip wapas banane ko kaho. Sirf HEALTHY clips pe
    train karo. Jo clip wo theek se dobara na bana paaye --- wo anomaly hai.

Kyun kaam karta hai:
    19 channels x 80 timesteps = 1520 numbers. Unhe 24 me nichodna ratt maar
    kar mumkin hi nahi hai. Model ko FLIGHT KI PHYSICS seekhni padegi:
    "altitude badhti hai to chaaron PWM saath badhte hain", "roll aata hai to
    do motor upar do neeche". Yahi rishte hain jo fault me tootte hain.

Kyun sirf healthy pe train:
    Model healthy flight ka expert ban jaata hai. Faulty clip uske tajurbe se
    bahar hoti hai --- wo use theek se rebuild nahi kar paata. Aur khaas baat:
    isse wo fault bhi pakde jaate hain jo humne kabhi sikhaye hi nahi.

Fair comparison ke liye baseline se bilkul milaya gaya hai:
    * wahi 8-second window
    * wahi 2-second stride
    * wahi threshold logic (common.py)
    * wahi K_CONSEC = 3 alarm rule
Farq sirf ek hai: baseline har window ko 38 numbers me nichodta hai
(mean + std), LSTM poora time-series dekhta hai.
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

# ---- sab kuch reproducible rakho ----
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---- config ----
DS        = 5                          # 50 Hz -> 10 Hz (har 5 sample ka average)
SEQ       = int(WIN_S * FS / DS)       # 80 steps = 8 seconds
SEQ_STRIDE = int(STRIDE_S * FS / DS)   # 20 steps = 2 seconds
LATENT    = 24                         # bottleneck --- yahi model ko "samajhne" pe majboor karta hai
HIDDEN    = 64
EPOCHS    = 250
BATCH     = 256
LR        = 1e-3
PATIENCE  = 20


# ------------------------------------------------------------------ data

def downsample(X, k=DS):
    """(N, T, C) -> (N, T/k, C), har k samples ka average.

    Average lete hain, har k-va sample uthate nahi. Simple skip karne se
    tez hilne wale signal (vibration) galat tarike se alias ho jaate hain
    --- unki jagah ek jhootha dheema pattern dikhne lagta hai.
    """
    N, T, C = X.shape
    T2 = (T // k) * k
    return X[:, :T2, :].reshape(N, T2 // k, k, C).mean(axis=2)


def to_sequences(X):
    """(N, T, C) -> (N, W, SEQ, C) --- har flight ki overlapping clips."""
    N, T, C = X.shape
    starts = range(0, T - SEQ + 1, SEQ_STRIDE)
    return np.stack([X[:, s:s + SEQ, :] for s in starts], axis=1)


# ------------------------------------------------------------------ model

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_ch, hidden=HIDDEN, latent=LATENT):
        super().__init__()
        # ENCODER: poori clip padhta hai, aakhir me ek summary vector deta hai
        self.enc = nn.LSTM(n_ch, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)      # <-- bottleneck

        # DECODER: sirf us summary se poori clip wapas banata hai
        self.from_latent = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_ch)

    def forward(self, x):                     # x: (B, SEQ, C)
        _, (h, _) = self.enc(x)               # h: (1, B, hidden)
        z = self.to_latent(h[-1])             # (B, latent)   <-- yahan sab kuch nichud gaya

        # decoder ko har timestep pe wahi ek summary di jaati hai.
        # Uske paas original clip dekhne ka koi rasta nahi --- sirf z hai.
        h0 = torch.tanh(self.from_latent(z))          # (B, hidden)
        seq = h0.unsqueeze(1).repeat(1, x.size(1), 1)  # (B, SEQ, hidden)
        y, _ = self.dec(seq)
        return self.out(y)


# ------------------------------------------------------------------ train

def main():
    d = np.load(ROOT / "data" / "processed" / "flights.npz", allow_pickle=True)
    meta = json.load(open(ROOT / "data" / "processed" / "meta.json"))
    fmeta = [m for m in meta if m["split"] == "test_faulty"]
    T_raw = d["train"].shape[1]

    print()
    print("Data taiyaar kar rahe hain...")
    Xtr = downsample(d["train"])
    Xva = downsample(d["val"])
    Xth = downsample(d["test_healthy"])
    Xtf = downsample(d["test_faulty"])

    # Normalize: mean/std SIRF train se. Val ya test se nikalna leakage hai.
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-8

    def prep(X):
        return to_sequences((X - mu) / sd).astype(np.float32)

    Str, Sva_seq = prep(Xtr), prep(Xva)
    Sth_seq, Stf_seq = prep(Xth), prep(Xtf)
    n_ch = Str.shape[-1]

    print(f"  train sequences : {Str.shape[0]} flights x {Str.shape[1]} clips "
          f"x {Str.shape[2]} steps x {n_ch} channels")
    print(f"  har clip         : {SEQ} steps @ 10 Hz = {SEQ / (FS / DS):.0f} seconds")
    print(f"  bottleneck       : {SEQ * n_ch} numbers -> {LATENT} "
          f"({SEQ * n_ch / LATENT:.0f}x compression)")

    flat_tr = torch.tensor(Str.reshape(-1, SEQ, n_ch))
    flat_va = torch.tensor(Sva_seq.reshape(-1, SEQ, n_ch))

    model = LSTMAutoencoder(n_ch)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model parameters : {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    lossfn = nn.MSELoss()

    print()
    print("Training (sirf healthy flights pe)...")
    best, best_state, bad = float("inf"), None, 0
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(flat_tr))
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            xb = flat_tr[perm[i:i + BATCH]]
            opt.zero_grad()
            loss = lossfn(model(xb), xb)
            loss.backward()
            # gradient clipping: LSTM me gradients kabhi kabhi phat jaate hain
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb)
        tr_loss = tot / len(perm)

        model.eval()
        with torch.no_grad():
            va_loss = sum(lossfn(model(flat_va[i:i + BATCH]), flat_va[i:i + BATCH]).item()
                          * len(flat_va[i:i + BATCH])
                          for i in range(0, len(flat_va), BATCH)) / len(flat_va)
        sched.step(va_loss)

        if va_loss < best - 1e-5:
            best, bad = va_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

        if ep % 5 == 0 or ep == 1 or bad >= PATIENCE:
            print(f"  epoch {ep:3d}  train {tr_loss:.5f}   val {va_loss:.5f}"
                  f"   {'*' if bad == 0 else ''}   ({time.time() - t0:.0f}s)")
        if bad >= PATIENCE:
            print(f"  early stop --- {PATIENCE} epochs se val loss nahi sudhra")
            break

    model.load_state_dict(best_state)
    model.eval()
    print(f"  best val loss : {best:.5f}")

    # -------------------------------------------------- scoring
    #
    # YAHAN EK BADA SABAK HAI.
    #
    # Pehle main saare channels ka error seedha AVERAGE kar raha tha. Wo galat
    # tha, aur is wajah se model haar gaya tha:
    #
    #   * gyro_x ka healthy error 0.99 hai --- total ka 8.6%
    #   * voltage ka healthy error 0.056 hai --- total ka sirf 0.49%
    #
    #   Matlab shor machaane wale channels score pe kabza kar lete hain.
    #   battery_sag me voltage ka error 40 GUNA badh jaata tha, par 0.49% ka
    #   40 guna bhi average ko mushkil se hilata tha --- aur fault chhoot jaata.
    #
    # Fix: har channel ka error pehle USKE APNE healthy baseline se naapo,
    # phir jodo. Ab har channel ko barabar vote milta hai --- voltage ka 40x
    # utna hi bhaari hai jitna vibration ka 520x.
    #
    # Baseline mu/sd TRAIN se aate hain, val/test se nahi (warna leakage).

    @torch.no_grad()
    def per_channel_err(S):
        """(N, W, SEQ, C) -> (N, W, C)   har channel ka alag error."""
        N, W = S.shape[:2]
        flat = torch.tensor(S.reshape(-1, SEQ, n_ch))
        errs = []
        for i in range(0, len(flat), 512):
            xb = flat[i:i + 512]
            errs.append(((model(xb) - xb) ** 2).mean(dim=1).numpy())   # time pe average
        return np.concatenate(errs).reshape(N, W, n_ch)

    E_tr = per_channel_err(Str)
    e_mu = E_tr.reshape(-1, n_ch).mean(0)
    e_sd = E_tr.reshape(-1, n_ch).std(0) + 1e-8

    def score(S):
        Z = (per_channel_err(S) - e_mu) / e_sd     # har channel apne paimane pe
        # RMS, na ki mean. mean() me ek channel ka chillana 18 khamosh channels
        # me dub jaata hai. Squaring bade deviation ko bhaari bana deti hai ---
        # aur yahi math Mahalanobis bhi karta hai. (rescore.py me teeno
        # aggregators ka farq naapa gaya hai.)
        return np.sqrt((Z ** 2).mean(axis=-1))

    Sv, Sh, Sf = score(Sva_seq), score(Sth_seq), score(Stf_seq)
    wt = common.window_times(T_raw)

    res = common.report("LSTM AUTOENCODER  --  8s clips, 24-dim bottleneck",
                        Sv, Sh, Sf, fmeta, wt)

    # -------------------------------------------------- comparison
    bpath = ROOT / "outputs" / "baseline_result.json"
    if bpath.exists():
        b = json.load(open(bpath))
        print("  " + "-" * 66)
        print(f"  {'':<22}{'BASELINE':>12}{'LSTM':>12}{'farq':>12}")
        print("  " + "-" * 66)
        print(f"  {'recall':<22}{100 * b['recall']:>11.0f}%{100 * res['recall']:>11.0f}%"
              f"{100 * (res['recall'] - b['recall']):>+11.0f}%")
        print(f"  {'false alarm':<22}{100 * b['false_alarm']:>11.0f}%"
              f"{100 * res['false_alarm']:>11.0f}%"
              f"{100 * (res['false_alarm'] - b['false_alarm']):>+11.0f}%")
        for k in ["motor_degradation", "prop_damage", "battery_sag", "imu_drift"]:
            bv, lv = b["per_kind"][k], res["per_kind"][k]
            print(f"  {k:<22}{100 * bv:>11.0f}%{100 * lv:>11.0f}%{100 * (lv - bv):>+11.0f}%")
        print("  " + "-" * 66)
        print()

    torch.save({"state": best_state, "mu": mu, "sd": sd, "e_mu": e_mu, "e_sd": e_sd,
                "n_ch": n_ch, "seq": SEQ, "latent": LATENT},
               ROOT / "outputs" / "lstm_model.pt")
    np.savez(ROOT / "outputs" / "lstm_scores.npz",
             val=Sv, test_healthy=Sh, test_faulty=Sf,
             window_times=wt, threshold=res["threshold"])
    with open(ROOT / "outputs" / "lstm_result.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"  saved -> outputs/lstm_model.pt")


if __name__ == "__main__":
    main()
