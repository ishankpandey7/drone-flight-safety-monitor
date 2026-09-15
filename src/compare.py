"""
Sab detectors ka IMAANDAR comparison --- ab LEAD TIME ke saath.
===============================================================

Pehle main sirf recall dekh raha tha: "kitne % faults pakde". Ab drone
sach me girta hai, to asli sawaal badal gaya hai:

    "Pilot ko kitne second ki warning mili?"

2 second ki warning bekaar hai --- utni der me koi kuch nahi kar sakta.
15 second me drone utar sakta hai. Isliye ab hum poori DISTRIBUTION dekhte
hain, ek average number nahi.

Do cheezein hamesha dhyan me:

  1. Crash ke baad ki windows ginti me nahi aati. Palte hue drone ko
     "detect" karna koi kamaal nahi hai.

  2. Har detector ka threshold alag operating point pe girta hai, isliye
     recall ka seedha comparison bemani hai. ROC poori range dikhata hai.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import flight_alarm, K_CONSEC

ROOT = Path(__file__).resolve().parent.parent
meta = json.load(open(ROOT / "data" / "processed" / "meta.json"))
fmeta = [m for m in meta if m["split"] == "test_faulty"]
KINDS = ["motor_degradation", "prop_damage", "battery_sag", "imu_drift"]
kinds = [m["fault_kind"] for m in fmeta]

DETECTORS = [
    ("Mahalanobis baseline", "baseline",   "#57B894"),
    ("LSTM autoencoder",     "lstm",       "#C9A227"),
    ("LSTM forecaster",      "forecaster", "#5B8DEF"),
    ("Hybrid A (fc->Mahal)", "hybrid_a",   "#A855F7"),
    ("Hybrid B (stats+fc)",  "hybrid_b",   "#EC4899"),
]
DETECTORS = [d for d in DETECTORS
             if (ROOT / "outputs" / f"{d[1]}_scores.npz").exists()]


# ------------------------------------------------------------------ helpers

def peak3(S, wt, mask_crash=True):
    """Har flight ka wo score jispe '3 lagatar windows' wala alarm bajega.
    Crash ke baad ki windows hata kar."""
    out = []
    for i, s in enumerate(S):
        if mask_crash and i < len(fmeta):
            crash = fmeta[i].get("crash_s", -1.0)
            s = s[wt <= crash] if crash > 0 else s
        best = -np.inf
        for j in range(len(s) - K_CONSEC + 1):
            best = max(best, min(s[j:j + K_CONSEC]))
        out.append(best if np.isfinite(best) else -np.inf)
    return np.array(out)


def lead_times(S, wt, thr):
    """Har CRASHED flight ka lead time. Alarm na baja to NaN (koi warning nahi)."""
    out = []
    for s, m in zip(S, fmeta):
        crash = m.get("crash_s", -1.0)
        if crash <= 0:
            continue                      # gira hi nahi, lead time ka matlab nahi
        keep = wt <= crash
        if keep.sum() < K_CONSEC:
            out.append(np.nan)
            continue
        alarm, widx = flight_alarm(s[keep], thr)
        out.append(crash - wt[keep][widx] if alarm else np.nan)
    return np.array(out)


def roc(ph, pf):
    ts = np.unique(np.concatenate([ph, pf]))
    ts = np.concatenate([[-np.inf], ts, [np.inf]])
    fa = np.array([np.mean(ph > t) for t in ts])
    tp = np.array([np.mean(pf > t) for t in ts])
    o = np.argsort(fa)
    return fa[o], tp[o]


# ------------------------------------------------------------------ compute

data = {}
for name, tag, col in DETECTORS:
    z = np.load(ROOT / "outputs" / f"{tag}_scores.npz")
    res = json.load(open(ROOT / "outputs" / f"{tag}_result.json"))
    wt = z["window_times"]
    ph = peak3(z["test_healthy"], wt, mask_crash=False)
    pf = peak3(z["test_faulty"], wt)
    data[name] = {
        "tag": tag, "col": col, "res": res, "wt": wt,
        "ph": ph, "pf": pf, "roc": roc(ph, pf),
        "leads": lead_times(z["test_faulty"], wt, float(z["threshold"])),
    }

n_crash = int(sum(1 for m in fmeta if m.get("crash_s", -1) > 0))

print()
print("=" * 96)
print(f"  LEAD TIME  ---  {n_crash}/{len(fmeta)} faulty flights crash hui")
print("=" * 96)
print()
print(f"  {'detector':<24}{'warning mili':>14}{'median':>9}{'p25':>8}"
      f"{'>=3s':>8}{'>=5s':>8}{'>=10s':>8}")
print("  " + "-" * 92)
for name, _, _ in DETECTORS:
    L = data[name]["leads"]
    ok = L[~np.isnan(L)]
    warned = len(ok) / len(L) if len(L) else 0
    if len(ok):
        print(f"  {name:<24}{100 * warned:>13.0f}%{np.median(ok):>8.1f}s"
              f"{np.percentile(ok, 25):>7.1f}s"
              f"{100 * np.mean(np.nan_to_num(L, nan=-1) >= 3):>7.0f}%"
              f"{100 * np.mean(np.nan_to_num(L, nan=-1) >= 5):>7.0f}%"
              f"{100 * np.mean(np.nan_to_num(L, nan=-1) >= 10):>7.0f}%")
    else:
        print(f"  {name:<24}{'0%':>14}{'-':>9}{'-':>8}{'-':>8}{'-':>8}{'-':>8}")
print("  " + "-" * 92)
print("  'warning mili'  = kitne % crash hone wali flights me alarm crash SE PEHLE baja")
print("  '>=5s'          = kitne % me kam se kam 5 second ka waqt mila")

best = max(DETECTORS, key=lambda d: np.mean(
    np.nan_to_num(data[d[0]]["leads"], nan=-1) >= 5))
print()
print(f"  >> 5+ second ki warning dene me sabse aage: {best[0]}")

# ------------------------------------------------------------------ ROC table
print()
print("=" * 96)
print("  Recall, ek hi FALSE ALARM level pe (threshold test pe ghumaya = CEILING)")
print("=" * 96)
print()
levels = [0.0, 0.04, 0.08, 0.12, 0.20]
print(f"  {'detector':<24}" + "".join(f"{f'FA<={int(100*f)}%':>12}" for f in levels))
print("  " + "-" * 88)
for name, _, _ in DETECTORS:
    fa, tp = data[name]["roc"]
    row = [float(tp[fa <= L + 1e-9].max()) if (fa <= L + 1e-9).any() else 0.0
           for L in levels]
    data[name]["ceil_row"] = row
    print(f"  {name:<24}" + "".join(f"{100 * r:>11.0f}%" for r in row))
print("  " + "-" * 88)

wins = {n: 0 for n, _, _ in DETECTORS}
for li in range(len(levels)):
    wins[max(wins, key=lambda n: data[n]["ceil_row"][li])] += 1
champ = max(wins, key=wins.get)
print()
print(f"  >> {champ} {wins[champ]}/{len(levels)} operating points pe aage hai.")
if wins[champ] < len(levels):
    print("     Ranking operating point ke saath badalti hai --- koi ek 'best' nahi.")

# ------------------------------------------------------------------ plot
fig, ax = plt.subplots(1, 3, figsize=(18.5, 5.6),
                       gridspec_kw={"width_ratios": [1, 1.15, 1.1]})

# --- A: ROC ---
for name, _, col in DETECTORS:
    fa, tp = data[name]["roc"]
    auc = np.trapezoid(tp, fa) if hasattr(np, "trapezoid") else np.trapz(tp, fa)
    ax[0].plot(100 * fa, 100 * tp, color=col, lw=2.2, label=f"{name}  ({auc:.2f})")
ax[0].plot([0, 100], [0, 100], ls=":", color="#999", lw=1.2, label="random")
ax[0].set_xlabel("False alarm rate  (% healthy flights)")
ax[0].set_ylabel("Recall  (% crashes crash se pehle pakde)")
ax[0].set_title("A.  Operating curve  (AUC)", fontsize=12.5, fontweight="bold", loc="left")
ax[0].legend(fontsize=8.5, loc="lower right")
ax[0].grid(alpha=0.2); ax[0].set_xlim(-2, 62); ax[0].set_ylim(-2, 102)

# --- B: LEAD TIME CDF --- yahi ab headline chart hai
grid = np.linspace(0, 40, 200)
for name, _, col in DETECTORS:
    L = np.nan_to_num(data[name]["leads"], nan=-1.0)
    y = [100 * np.mean(L >= g) for g in grid]
    ax[1].plot(grid, y, color=col, lw=2.4, label=name)
ax[1].axvline(5, color="#666", ls=":", lw=1.4)
ax[1].text(5.5, 96, "5s = kuch karne\nka waqt", fontsize=9, color="#444",
           va="top", fontweight="bold")
ax[1].set_xlabel("Warning ka waqt (seconds before crash)")
ax[1].set_ylabel("% crashes jinme itni warning mili")
ax[1].set_title("B.  Kitne second ki warning mili?", fontsize=12.5,
                fontweight="bold", loc="left")
ax[1].legend(fontsize=8.5, loc="upper right")
ax[1].grid(alpha=0.2); ax[1].set_xlim(0, 40); ax[1].set_ylim(0, 103)

# --- C: per fault type recall ---
n_det = len(DETECTORS)
w = 0.82 / n_det
xs = np.arange(len(KINDS))
for k, (name, _, col) in enumerate(DETECTORS):
    vals = [100 * data[name]["res"]["per_kind"][kd] for kd in KINDS]
    off = (k - (n_det - 1) / 2) * w
    ax[2].bar(xs + off, vals, w * 0.92, color=col, label=name)
    for x, v in zip(xs + off, vals):
        if v > 3:
            ax[2].text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=7, color="#333")
ax[2].set_xticks(xs)
ax[2].set_xticklabels([k.replace("_", "\n") for k in KINDS], fontsize=9.5)
ax[2].set_ylabel("Recall  (%)")
ax[2].set_title("C.  Fault type ke hisaab se", fontsize=12.5, fontweight="bold", loc="left")
ax[2].legend(fontsize=7.5, loc="lower left", framealpha=0.92)
ax[2].grid(alpha=0.2, axis="y"); ax[2].set_ylim(0, 118)

for a in ax:
    a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)

fig.suptitle("Drone fault detection --- asli sawaal: crash se kitne second pehle pata chala?",
             fontsize=15, fontweight="bold", y=1.0)
fig.tight_layout()
fig.savefig(ROOT / "outputs" / "02_comparison.png", dpi=140,
            bbox_inches="tight", facecolor="white")
print()
print("  saved -> outputs/02_comparison.png")
print()
