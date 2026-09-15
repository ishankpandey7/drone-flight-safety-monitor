"""
Model banane se PEHLE ka sabse zaroori sawaal:

    Kya signal data me MAUJOOD hai?

Agar nahi hai, to koi bhi model, kitna bhi bada, kuch nahi kar payega ---
aur main ghante barbaad karunga ye sochte hue ki "training theek se nahi
ho rahi". Ye check pehle hona chahiye.

Trick: har faulty flight ko DOOSRI flights se nahi, APNE AAP se compare karo
(fault se pehle wala hissa vs fault ke baad wala hissa). Isse drone-to-drone
variation apne aap cancel ho jaati hai.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
d = np.load(ROOT / "data" / "processed" / "flights.npz", allow_pickle=True)
meta = json.load(open(ROOT / "data" / "processed" / "meta.json"))

CH = list(d["channels"])
FS = 50.0                                    # 50 Hz
idx = {c: i for i, c in enumerate(CH)}
PWM = [idx[f"pwm_{i}"] for i in range(1, 5)]

Xf = d["test_faulty"]
fmeta = [m for m in meta if m["split"] == "test_faulty"]
assert len(fmeta) == len(Xf), f"{len(fmeta)} != {len(Xf)}"


def window_stats(x, lo, hi):
    """x: (T, C). lo/hi seconds me."""
    w = x[int(lo * FS):int(hi * FS)]
    p = w[:, PWM]
    return {
        "pwm_spread": float(np.median(p.max(-1) - p.min(-1))),
        "pwm_mean":   float(p.mean()),
        "vibe":       float(np.median(w[:, idx["vibe_x"]])),
        "voltage":    float(np.median(w[:, idx["voltage"]])),
        "current":    float(np.median(w[:, idx["current"]])),
        "gyro_x":     float(np.std(w[:, idx["gyro_x"]])),
    }


print()
print("=" * 100)
print("Har faulty flight ko APNE aap se compare karo: fault se 25s PEHLE  vs  fault ke 20-40s BAAD")
print("=" * 100)
print()

by_kind = {}
for x, m in zip(Xf, fmeta):
    t0 = m["fault_start_s"]
    # "fault ke baad" wali window CRASH SE PEHLE khatam honi chahiye. Crash ke
    # baad drone palat raha hai --- wahan har channel pagal dikhega, aur ye
    # table jhoothe taur pe shaandaar aa jayega.
    crash = m.get("crash_s", -1.0)
    hi = (crash - 1.0) if crash > 0 else (len(x) / FS - 1.0)
    lo_after = t0 + 5
    if t0 - 25 < 5 or hi - lo_after < 8:      # window flight ke andar honi chahiye
        continue
    before = window_stats(x, t0 - 25, t0 - 1)
    after  = window_stats(x, lo_after, min(t0 + 35, hi))
    by_kind.setdefault(m["fault_kind"], []).append((before, after, m))

hdr = f"  {'fault type':<20} {'n':>3}  " + "  ".join(f"{k:>12}" for k in
      ["pwm_spread", "pwm_mean", "vibe", "voltage", "current", "gyro_x"])
print(hdr)
print("  " + "-" * (len(hdr) - 2))

for kind, rows in by_kind.items():
    deltas = {}
    for key in ["pwm_spread", "pwm_mean", "vibe", "voltage", "current", "gyro_x"]:
        b = np.array([r[0][key] for r in rows])
        a = np.array([r[1][key] for r in rows])
        # relative change, % me
        deltas[key] = 100 * np.mean((a - b) / (np.abs(b) + 1e-9))
    print(f"  {kind:<20} {len(rows):>3}  " +
          "  ".join(f"{deltas[k]:>+11.1f}%" for k in
                    ["pwm_spread", "pwm_mean", "vibe", "voltage", "current", "gyro_x"]))

print()
print("  (har number = fault ke baad wo cheez kitne % badli, us flight ke apne baseline se)")

# ------------------------------------------------------------------ separability
print()
print("=" * 100)
print("Ab asli sawaal: kya ek SIMPLE detector in flights ko alag kar sakta hai?")
print("=" * 100)
print()

Xh = d["test_healthy"]


def flight_score(x, crash=-1.0):
    """Sabse naive score: 8-second window me motors ka farq, uska max.

    crash ke baad ka data kaat do --- warna ye naive detector bhi 100% dikhega,
    kyunki palta hua drone pakadna koi kamaal nahi hai.
    """
    if crash > 0:
        x = x[:int(crash * FS)]
    p = x[:, PWM]
    s = p.max(-1) - p.min(-1)
    k = int(8 * FS)
    if len(s) < k:
        return 0.0
    sm = np.convolve(s, np.ones(k) / k, mode="valid")
    return float(sm.max())


sh = np.array([flight_score(x) for x in Xh])
sf = np.array([flight_score(x, m.get("crash_s", -1.0)) for x, m in zip(Xf, fmeta)])

thr = np.percentile(sh, 95)                   # 5% false alarm par set karo
recall = float(np.mean(sf > thr))
print(f"  Healthy flights ka score : {sh.mean():.4f}  (95th pct = {thr:.4f})")
print(f"  Faulty  flights ka score : {sf.mean():.4f}")
print()
print(f"  5% false-alarm rate par is naive detector ki recall : {100 * recall:.0f}%")
print(f"  Yani 60 me se {int(recall * 60)} faulty flights pakdi, {60 - int(recall * 60)} chhoot gayi.")

print()
print("  Fault type ke hisaab se:")
kinds = [m["fault_kind"] for m in fmeta]
for kind in ["motor_degradation", "prop_damage", "battery_sag", "imu_drift"]:
    sel = np.array([k == kind for k in kinds])
    r = float(np.mean(sf[sel] > thr))
    bar = "#" * int(r * 20)
    print(f"    {kind:<20} {100 * r:5.0f}%  {bar}")
print()
