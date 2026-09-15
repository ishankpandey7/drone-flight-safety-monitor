"""
Fault signature ko aankhon se dekhne ke liye plot.

Teen panel:
  A) Healthy flight ke 4 motors     -> spikes aate hain, par CHAARON saath chalte hain
  B) Faulty flight ke 4 motors      -> motor 3 akela alag ho jata hai
  C) Dono ka "spread" (max - min)   -> yahan dikhta hai ki threshold kyun impossible hai
"""

import matplotlib
matplotlib.use("Agg")                      # bina window ke seedha file me save karo

import json
from pathlib import Path

import matplotlib.pyplot as plt
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

# Crash ke baad ka data frozen hai --- plot me dikhane ka koi fayda nahi,
# aur statistics me lena to bilkul galat hai.
T_END = (CRASH_T + 2) if CRASH_T > 0 else broken.t.max()
broken  = broken[broken.t <= T_END].copy()
healthy = healthy[healthy.t <= T_END].copy()

COL = {"pwm_1": "#5B8DEF", "pwm_2": "#57B894",
       "pwm_3": "#E5484D", "pwm_4": "#C9A227"}

fig, ax = plt.subplots(3, 1, figsize=(13.5, 9.5), sharex=True,
                       gridspec_kw={"height_ratios": [1, 1, 1.15], "hspace": 0.28})

# ------------------------------------------------------------------ Panel A
for c in PWM:
    ax[0].plot(healthy.t, healthy[c], lw=0.8, alpha=0.85,
               color=COL[c], label=c.replace("pwm_", "Motor "))
ax[0].set_title("A.  HEALTHY flight  —  spikes aate hain, lekin chaaron motor SAATH chalte hain",
                fontsize=12, fontweight="bold", loc="left", pad=8)
ax[0].set_ylabel("PWM")
ax[0].legend(ncol=4, fontsize=9, loc="upper right", framealpha=0.9)
ax[0].set_ylim(-0.05, 1.05)

# ------------------------------------------------------------------ Panel B
for c in PWM:
    is_bad = (c == "pwm_3")
    ax[1].plot(broken.t, broken[c],
               lw=1.6 if is_bad else 0.8,
               alpha=1.0 if is_bad else 0.45,
               color=COL[c], zorder=3 if is_bad else 2,
               label=("Motor 3  (kharab)" if is_bad else c.replace("pwm_", "Motor ")))
ax[1].axvline(FAULT_T, color="#E8A33D", ls="--", lw=1.5, alpha=0.9)
ax[1].text(FAULT_T + 0.7, 0.04, "fault shuru", color="#E8A33D",
           fontsize=10, fontweight="bold")
if CRASH_T > 0:
    ax[1].axvline(CRASH_T, color="#B91C1C", lw=2.2)
    ax[1].text(CRASH_T + 0.7, 0.04, "CRASH", color="#B91C1C",
               fontsize=10, fontweight="bold")
ax[1].set_title("B.  FAULTY flight  —  motor 3 akela upar khisak jaata hai",
                fontsize=12, fontweight="bold", loc="left", pad=8)
ax[1].set_ylabel("PWM")
ax[1].legend(ncol=4, fontsize=9, loc="upper right", framealpha=0.9)
ax[1].set_ylim(-0.05, 1.05)

# ------------------------------------------------------------------ Panel C
def spread(df):
    s = df[PWM].max(axis=1) - df[PWM].min(axis=1)
    return s.rolling(50, min_periods=1).median()          # 1 second ka median

h_sp, b_sp = spread(healthy), spread(broken)
h_p99 = (healthy[PWM].max(axis=1) - healthy[PWM].min(axis=1)).quantile(0.99)
aft = broken[(broken.t >= FAULT_T + 5) & (broken.t <= (CRASH_T - 1 if CRASH_T > 0 else T_END))]
b_med = (aft[PWM].max(axis=1) - aft[PWM].min(axis=1)).median()

# Dono flights ka seed same hai, isliye fault se pehle wale spikes bilkul
# overlap karte hain. Healthy ko mota aur neeche rakho taaki dono dikhein.
ax[2].plot(healthy.t, h_sp, color="#57B894", lw=4.0, alpha=0.55, zorder=2,
           label="Healthy — motors ka farq")
ax[2].plot(broken.t,  b_sp, color="#E5484D", lw=1.6, zorder=3,
           label="Faulty — motors ka farq")
ax[2].axvline(FAULT_T, color="#E8A33D", ls="--", lw=1.5, alpha=0.9)
if CRASH_T > 0:
    ax[2].axvline(CRASH_T, color="#B91C1C", lw=2.2)

# Naive feature threshold ko KAB paar karta hai? Yahi asli sawaal hai ---
# "paar karta hai ya nahi" nahi, "crash se kitne pehle karta hai".
cross_t = None
mask = broken.t > FAULT_T
if (b_sp[mask] > h_p99).any():
    cross_t = float(broken.t[mask][b_sp[mask] > h_p99].iloc[0])

ax[2].axhline(h_p99, color="#2B2B2B", ls=":", lw=1.5)
ax[2].text(2, h_p99 + 0.008,
           f"threshold yahan rakhna padega ({h_p99:.2f}) — "
           f"warna healthy flights pe hi alarm bajega",
           fontsize=9.5, color="#2B2B2B", fontweight="bold", va="bottom")

if cross_t and CRASH_T > 0:
    lead = CRASH_T - cross_t
    y_ann = h_p99 * 1.45
    ax[2].axvspan(cross_t, CRASH_T, color="#B91C1C", alpha=0.15, zorder=0)
    ax[2].annotate("", xy=(CRASH_T, y_ann), xytext=(cross_t, y_ann),
                   arrowprops=dict(arrowstyle="<->", lw=2.2, color="#B91C1C"))
    # Text ARROW KE BAAYIN taraf, warna wo right edge se bahar nikal jaata hai
    ax[2].annotate(f"threshold paar hua...  sirf {lead:.1f}s ki warning  ",
                   xy=(cross_t, y_ann), xytext=(cross_t - 2, y_ann),
                   ha="right", va="center", fontsize=11.5,
                   fontweight="bold", color="#B91C1C")

ax[2].set_title("C.  Simple threshold bolta to hai --- par bahut der se",
                fontsize=12, fontweight="bold", loc="left", pad=8)
ax[2].set_ylabel("max(PWM) − min(PWM)")
ax[2].set_xlabel("time (seconds)")
ax[2].legend(fontsize=9.5, loc="upper left", framealpha=0.95)
ax[2].set_ylim(0, max(h_p99, float(b_sp.max())) * 1.30)

for a in ax:
    a.grid(alpha=0.18, lw=0.6)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

fig.suptitle(f"Motor fault: {FAULT_T:.0f}s pe shuru, {CRASH_T:.0f}s pe crash "
             f"--- detector ke paas {CRASH_T - FAULT_T:.0f} second the",
             fontsize=14.5, fontweight="bold", y=0.975)
fig.savefig(ROOT / "outputs" / "01_fault_signature.png", dpi=140, bbox_inches="tight",
            facecolor="white")
print("saved -> outputs/01_fault_signature.png")
print(f"  healthy p99 spread : {h_p99:.3f}")
print(f"  faulty  median      : {b_med:.3f}")
print(f"  ratio                : threshold {h_p99 / b_med:.1f}x bada hona padega signature se")
if cross_t and CRASH_T > 0:
    print(f"  naive feature ne threshold {cross_t:.1f}s pe paar kiya, crash {CRASH_T:.1f}s pe")
    print(f"  >> sirf {CRASH_T - cross_t:.1f} second ki warning --- itne me koi kuch nahi kar sakta")
