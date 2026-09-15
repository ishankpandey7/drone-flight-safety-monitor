"""
Baseline aur autoencoder ke result JSON purane (galat) denominator ke saath
likhe gaye the. Unhe saved scores se dobara nikal kar theek karo --- model
dobara train karne ki koi zaroorat nahi.
"""

import json
from pathlib import Path

import numpy as np

from common import flight_alarm, K_CONSEC

ROOT = Path(__file__).resolve().parent.parent
meta = json.load(open(ROOT / "data" / "processed" / "meta.json"))
fmeta = [m for m in meta if m["split"] == "test_faulty"]
n_crashed = sum(1 for m in fmeta if m.get("crash_s", -1.0) > 0)

print()
print(f"  {n_crashed} flights sach me crash hui --- yahi sahi denominator hai")
print()
print(f"  {'detector':<24}{'>=3s':>8}{'>=5s':>8}{'>=10s':>8}{'median':>9}{'p25':>8}")
print("  " + "-" * 66)

for tag in ["baseline", "lstm", "forecaster", "hybrid_a", "hybrid_b"]:
    sp = ROOT / "outputs" / f"{tag}_scores.npz"
    rp = ROOT / "outputs" / f"{tag}_result.json"
    if not (sp.exists() and rp.exists()):
        continue
    z = np.load(sp)
    S, wt, thr = z["test_faulty"], z["window_times"], float(z["threshold"])

    leads = []
    for s, m in zip(S, fmeta):
        crash = m.get("crash_s", -1.0)
        if crash <= 0:
            continue
        keep = wt <= crash
        if keep.sum() < K_CONSEC:
            continue
        alarm, widx = flight_alarm(s[keep], thr)
        if alarm:
            leads.append(crash - wt[keep][widx])
    leads = np.array(leads)

    r = json.load(open(rp))
    r["n_crashed"] = n_crashed
    for lim in (3, 5, 10):
        r[f"lead_ge{lim}"] = float(np.sum(leads >= lim) / n_crashed) if n_crashed else 0.0
    r["lead_median"] = float(np.median(leads)) if len(leads) else None
    r["lead_p25"] = float(np.percentile(leads, 25)) if len(leads) else None
    r["lead_min"] = float(leads.min()) if len(leads) else None
    json.dump(r, open(rp, "w"), indent=1)

    med = f"{r['lead_median']:.1f}s" if r["lead_median"] else "-"
    p25 = f"{r['lead_p25']:.1f}s" if r["lead_p25"] else "-"
    print(f"  {r['name'][:24]:<24}{100*r['lead_ge3']:>7.0f}%{100*r['lead_ge5']:>7.0f}%"
          f"{100*r['lead_ge10']:>7.0f}%{med:>9}{p25:>8}")
print("  " + "-" * 66)
print("  (ab saare detectors ka denominator ek hi hai --- seedhe compare ho sakte hain)")
print()
