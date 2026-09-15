"""
Sanity check: kya fault ka nishaan data me sach me aaya?

Ye script SIRF isliye hai ki hum apni aankhon se confirm kar lein ki
simulator jhooth nahi bol raha. Yahan koi ML nahi hai.

NOTE: min/max se compare karna bekaar hai --- ek hi transient spike poori
range ko 0..1 bana deta hai, aur phir "range ke andar hai" ka koi matlab
nahi bachta. Isliye hum percentile use karte hain.
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
healthy = pd.read_csv(RAW / "demo_healthy.csv")
broken  = pd.read_csv(RAW / "demo_motor_fault.csv")
dm = json.load(open(RAW / "demo_meta.json"))

FAULT_T = dm["motor_fault"]["fault_start_s"]
CRASH_T = dm["motor_fault"]["crash_s"]
PWM = ["pwm_1", "pwm_2", "pwm_3", "pwm_4"]

# Faulty flight CRASH_T pe gir gayi. Uske baad ka data frozen hai (aakhri
# value repeat). Usko kaat do --- warna har number jhootha aayega.
if CRASH_T > 0:
    broken = broken[broken.t <= CRASH_T].copy()
    print()
    print(f"  NOTE: faulty flight {CRASH_T:.0f}s pe crash hui "
          f"({dm['motor_fault']['crash_reason']}).")
    print("  Sirf uske PEHLE ka data dekh rahe hain --- crash ke baad ki")
    print("  telemetry se koi bhi nateeja nikalna khud ko dhoka dena hai.")

# "fault ke baad" wali window: fault shuru hone ke 5s baad se crash se 1s pehle tak
AFT_LO = FAULT_T + 5
AFT_HI = (CRASH_T - 1) if CRASH_T > 0 else broken.t.max()

print()
print("=" * 92)
print("Pehle: simulator kitna aggressive hai?")
print("=" * 92)
print()

vals = healthy[PWM].values
sat = 100 * np.mean((vals <= 0.001) | (vals >= 0.999))
print(f"  Healthy flight me motors kitne % time saturate hue : {sat:.2f}%")
print(f"  (thoda saturation normal hai --- tez turn ke waqt asli drone me bhi hota hai)")

print()
print("=" * 92)
print("SAWAL 1: Kya kharab motor ka PWM AKELA dekhne pe 'ajeeb' lagta hai?")
print("=" * 92)

lo, hi = np.percentile(healthy[PWM].values, [1, 99])
print(f"\n  Healthy PWM ka normal band (1st-99th pct) : {lo:.3f}  se  {hi:.3f}")

late = broken[broken.t >= AFT_LO]
print(f"  Fault ke baad kharab motor ka average      : {late.pwm_3.mean():.3f}")
print(f"  Fault ke baad uska median                  : {late.pwm_3.median():.3f}")

inside = lo <= late.pwm_3.mean() <= hi
print()
if inside:
    print("  >> Kharab motor ka PWM ab bhi NORMAL BAND ke andar hai.")
    print("     Matlab: 'agar pwm_3 > X toh alarm' wala rule ya toh ise miss karega,")
    print("     ya har tez turn pe jhootha alarm bajayega. Threshold kaafi nahi hai.")
else:
    print("  >> PWM normal band se bahar nikal gaya --- is case me simple threshold")
    print("     bhi kaam kar jata. (Isliye baaki fault types zyada interesting hain.)")

print()
print("=" * 92)
print("SAWAL 2: Kya motors ka aapsi RISHTA tootta hai?")
print("=" * 92)
print()


def summarize(df, label, t_lo, t_hi):
    w = df[(df.t >= t_lo) & (df.t < t_hi)]
    m = w[PWM].mean()
    others = m[["pwm_1", "pwm_2", "pwm_4"]].mean()
    rel = 100 * (m["pwm_3"] - others) / others
    print(f"  {label:<38} m1={m.pwm_1:.3f} m2={m.pwm_2:.3f} "
          f"m3={m.pwm_3:.3f} m4={m.pwm_4:.3f}  | m3 vs baaki: {rel:+6.1f}%")
    return rel


summarize(healthy, "HEALTHY  (t=20-50s)", 20, 50)
summarize(broken,  "FAULTY   (t=20-50s, fault se PEHLE)", 20, 50)
rel_after = summarize(broken, f"FAULTY   (t={AFT_LO:.0f}-{AFT_HI:.0f}s, fault ke BAAD)", AFT_LO, AFT_HI)

print()
print(f"  >> Fault ke baad kharab motor {rel_after:.1f}% zyada mehnat kar raha hai.")
print("     Ye number kisi ek channel me nahi likha --- ye CHAAR channels ke")
print("     rishte se nikalta hai. Yahi baat model ko seekhni hai.")

print()
print("=" * 92)
print("SAWAL 3: Kitna jaldi pata chal sakta tha? (lead time)")
print("=" * 92)
print()

# har second pe relative asymmetry nikalo
b = broken.copy()
b["rel"] = 100 * (b.pwm_3 - b[["pwm_1", "pwm_2", "pwm_4"]].mean(axis=1)) \
                / b[["pwm_1", "pwm_2", "pwm_4"]].mean(axis=1)
b["rel_smooth"] = b["rel"].rolling(100, min_periods=1).mean()   # 2 second ka average

h = healthy.copy()
h["rel"] = 100 * (h.pwm_3 - h[["pwm_1", "pwm_2", "pwm_4"]].mean(axis=1)) \
                / h[["pwm_1", "pwm_2", "pwm_4"]].mean(axis=1)
noise_band = h["rel"].rolling(100, min_periods=1).mean().abs().quantile(0.99)

print(f"  Healthy flight me ye asymmetry kabhi {noise_band:.1f}% se upar nahi jaati.")

# Healthy flight me KHUD kitni baar alarm bajta? (ye false alarm rate hai)
h_trips = (h["rel"].rolling(100, min_periods=1).mean().abs() > noise_band) & (h.t > 10)
print(f"  Healthy flight me hi ye rule {100 * h_trips.mean():.1f}% samay alarm bajata hai.")

trip = b[(b.rel_smooth > noise_band) & (b.t > 10)]
print()
if len(trip):
    first = trip.t.iloc[0]
    print(f"  Faulty flight me pehla alarm : {first:.1f}s")
    print(f"  Fault sach me shuru hua       : {FAULT_T:.1f}s")
    print()
    if first < FAULT_T:
        print(f"  >> JHOOTHA ALARM. Rule ne fault se {FAULT_T - first:.1f} second PEHLE")
        print("     alarm baja diya --- jab drone bilkul theek tha. Wo bas ek tez")
        print("     maneuver tha. Aise detector pe koi pilot bharosa nahi karega:")
        print("     do-teen jhoothe alarm ke baad wo use band kar dega.")
    else:
        print(f"  >> Detection lag: {first - FAULT_T:.1f} seconds")
else:
    print("  Kabhi cross nahi hui --- signal noise me dab gaya (miss).")

print()
for col, nice in [("vibe_x", "vibration"), ("roll", "roll angle"), ("current", "current draw")]:
    hv = healthy[(healthy.t >= AFT_LO) & (healthy.t <= AFT_HI)][col].mean()
    bv = broken[(broken.t >= AFT_LO) & (broken.t <= AFT_HI)][col].mean()
    print(f"  {nice:<14} healthy={hv:+8.3f}  faulty={bv:+8.3f}  farq={bv - hv:+8.3f}")
print()
