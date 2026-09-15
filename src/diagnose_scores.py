"""
Do baar andaza laga kar galat nikla. Ab poori tasveer dekhte hain.

Sawaal: threshold 24.4 hai. Ye itna UNCHA kyun hai? Aur battery_sag flights
        ka score asal me hai kitna?
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
meta = json.load(open(ROOT / "data" / "processed" / "meta.json"))
fmeta = [m for m in meta if m["split"] == "test_faulty"]

for tag in ["lstm", "baseline"]:
    z = np.load(ROOT / "outputs" / f"{tag}_scores.npz")
    Sv, Sh, Sf, thr = z["val"], z["test_healthy"], z["test_faulty"], float(z["threshold"])

    print()
    print("=" * 96)
    print(f"  {tag.upper()}   (threshold = {thr:.2f})")
    print("=" * 96)

    # har flight ka "3 lagatar windows" wala peak score --- yahi alarm decide karta hai
    def peak3(S):
        out = []
        for s in S:
            best = -np.inf
            for i in range(len(s) - 2):
                best = max(best, min(s[i], s[i + 1], s[i + 2]))
            out.append(best)
        return np.array(out)

    pv, ph, pf = peak3(Sv), peak3(Sh), peak3(Sf)

    print()
    print(f"  {'group':<24}{'median':>10}{'p90':>10}{'max':>10}   thr se upar")
    print("  " + "-" * 70)
    for name, p in [("val healthy (25)", pv), ("test healthy (25)", ph)]:
        print(f"  {name:<24}{np.median(p):>10.2f}{np.percentile(p, 90):>10.2f}"
              f"{p.max():>10.2f}{100 * np.mean(p > thr):>11.0f}%")

    kinds = [m["fault_kind"] for m in fmeta]
    for kind in ["motor_degradation", "prop_damage", "battery_sag", "imu_drift"]:
        sel = np.array([k == kind for k in kinds])
        p = pf[sel]
        print(f"  {kind:<24}{np.median(p):>10.2f}{np.percentile(p, 90):>10.2f}"
              f"{p.max():>10.2f}{100 * np.mean(p > thr):>11.0f}%")

    # ---- ye sabse zaroori number hai ----
    print()
    print("  Agar threshold sirf itna hota ki HEALTHY ka max just paar ho jaaye:")
    thr_ideal = max(pv.max(), ph.max())
    print(f"    threshold = {thr_ideal:.2f}  (abhi {thr:.2f} hai)")
    for kind in ["motor_degradation", "prop_damage", "battery_sag", "imu_drift"]:
        sel = np.array([k == kind for k in kinds])
        r = np.mean(pf[sel] > thr_ideal)
        print(f"    {kind:<24}{100 * r:5.0f}%")
    print(f"    {'TOTAL':<24}{100 * np.mean(pf > thr_ideal):5.0f}%")

    # ---- aur agar battery_sag ka signal HAI hi nahi? ----
    sel = np.array([k == "battery_sag" for k in kinds])
    print()
    print(f"  battery_sag flights ka peak score : {pf[sel].min():.2f} se {pf[sel].max():.2f}")
    print(f"  healthy flights ka peak score     : {min(pv.min(), ph.min()):.2f} se {max(pv.max(), ph.max()):.2f}")
    ov = np.mean(pf[sel] < np.percentile(np.concatenate([pv, ph]), 50))
    print(f"  >> {100 * ov:.0f}% battery_sag flights healthy ke MEDIAN se bhi neeche hain")
print()
