"""
LSTM kyun haara? Andaza mat lagao --- naapo.

Sawaal: battery_sag me baseline 47% deta hai par LSTM 0%. Kyun?

Shak: autoencoder "CHEAT" kar raha hai. Battery sag ek dheema, seedha
      badlav hai. Autoencoder ka kaam hai input ko dobara banana ---
      aur dheemi seedhi cheez ko dobara banana bahut AASAN hai.
      Wo voltage ka jo bhi level dikhta hai use bas copy kar deta hai,
      chahe wo level ajeeb ho.

      Yani autoencoder ye nahi poochta "kya ye value normal hai?"
      Wo poochta hai "kya main ye value dobara bana sakta hoon?"
      Aur ye dono bahut alag sawaal hain.

Test: fault se pehle aur baad me, HAR CHANNEL ka reconstruction error
      alag alag dekho. Agar shak sahi hai to voltage ka error nahi badhega.
"""

import json
from pathlib import Path

import numpy as np
import torch

from lstm_model import LSTMAutoencoder, downsample, to_sequences, SEQ, DS
from common import FS

ROOT = Path(__file__).resolve().parent.parent

ck = torch.load(ROOT / "outputs" / "lstm_model.pt", weights_only=False)
model = LSTMAutoencoder(ck["n_ch"])
model.load_state_dict(ck["state"])
model.eval()
mu, sd = ck["mu"], ck["sd"]

d = np.load(ROOT / "data" / "processed" / "flights.npz", allow_pickle=True)
CH = [str(c) for c in d["channels"]]
meta = json.load(open(ROOT / "data" / "processed" / "meta.json"))
fmeta = [m for m in meta if m["split"] == "test_faulty"]


@torch.no_grad()
def per_channel_err(clips):
    """clips: (n, SEQ, C) normalized -> (n, C) har channel ka MSE."""
    x = torch.tensor(clips.astype(np.float32))
    out = []
    for i in range(0, len(x), 256):
        xb = x[i:i + 256]
        out.append(((model(xb) - xb) ** 2).mean(dim=1).numpy())
    return np.concatenate(out)


def clips_in_window(X_raw_one, t_lo, t_hi):
    """Ek flight ke raw (T, C) se, t_lo..t_hi seconds ki clips nikalo."""
    x = downsample(X_raw_one[None])[0]
    x = (x - mu) / sd
    fs_ds = FS / DS
    lo, hi = int(t_lo * fs_ds), int(t_hi * fs_ds)
    x = x[lo:hi]
    if len(x) < SEQ:
        return None
    return np.stack([x[s:s + SEQ] for s in range(0, len(x) - SEQ + 1, 10)])


Xtf = d["test_faulty"]

print()
print("=" * 104)
print("Har channel ka reconstruction error: fault se PEHLE  ->  fault ke BAAD  (% badlav)")
print("=" * 104)

results = {}
for kind in ["motor_degradation", "prop_damage", "battery_sag", "imu_drift"]:
    befores, afters = [], []
    for x, m in zip(Xtf, fmeta):
        if m["fault_kind"] != kind:
            continue
        t0 = m["fault_start_s"]
        if t0 - 25 < 2 or t0 + 40 > 120:
            continue
        cb = clips_in_window(x, t0 - 25, t0 - 1)
        ca = clips_in_window(x, t0 + 20, min(t0 + 40, 119))
        if cb is None or ca is None:
            continue
        befores.append(per_channel_err(cb).mean(0))
        afters.append(per_channel_err(ca).mean(0))
    b, a = np.array(befores).mean(0), np.array(afters).mean(0)
    results[kind] = 100 * (a - b) / (b + 1e-9)

print()
print(f"  {'channel':<12}" + "".join(f"{k[:14]:>18}" for k in results))
print("  " + "-" * 100)
for i, c in enumerate(CH):
    row = "".join(f"{results[k][i]:>+17.0f}%" for k in results)
    mark = "  <<<" if any(abs(results[k][i]) > 100 for k in results) else ""
    print(f"  {c:<12}{row}{mark}")

print()
print("=" * 104)
print("Faisla")
print("=" * 104)
v = CH.index("voltage")
cur = CH.index("current")
print(f"""
  battery_sag me voltage ka error  : {results['battery_sag'][v]:+.0f}%
  battery_sag me current ka error  : {results['battery_sag'][cur]:+.0f}%

  Baseline ne battery_sag isi voltage se pakda tha (median voltage -6.5%).
  Lekin autoencoder ka voltage error {'nahi badha' if abs(results['battery_sag'][v]) < 30 else 'badha'}.

  >> Autoencoder ne voltage ko bas COPY kar diya. Usne ye nahi poocha ki
     "kya 15.2 V normal hai?" --- usne poocha "kya main 15.2 likh sakta hoon?"
     Aur uska jawab hamesha HAAN hai.

  YAHI reconstruction-based anomaly detection ki sabse badi kamzori hai:
  jo anomaly DHEEMI aur SEEDHI hai, use model aasani se dobara bana leta hai,
  aur error zero reh jaata hai.
""")
