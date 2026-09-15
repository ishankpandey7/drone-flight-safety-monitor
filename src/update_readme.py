"""
README.md ka results table apne aap bharo.

Numbers haath se likhne se wo hamesha purane ho jaate hain --- aur purane
numbers jhooth hi hote hain. Isliye README me ek marker hai, aur ye script
wahan asli results daal deti hai.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
KINDS = ["motor_degradation", "prop_damage", "battery_sag", "imu_drift"]
ROWS = [
    ("baseline",   "Mahalanobis baseline"),
    ("lstm",       "LSTM autoencoder"),
    ("forecaster", "LSTM forecaster"),
    ("hybrid_a",   "Hybrid A"),
    ("hybrid_b",   "Hybrid B"),
]
MARK = "<!-- RESULTS_TABLE -->"


def main():
    meta = json.load(open(ROOT / "data" / "processed" / "meta.json"))
    fmeta = [m for m in meta if m["split"] == "test_faulty"]
    crashed = [m for m in fmeta if m.get("crash_s", -1) > 0]
    surv = np.array([m["survived_s"] for m in crashed]) if crashed else np.array([])

    res = {}
    for tag, _ in ROWS:
        p = ROOT / "outputs" / f"{tag}_result.json"
        if p.exists():
            res[tag] = json.load(open(p))
    if not res:
        print("  koi result file nahi mili --- pehle pipeline chalao")
        return

    # "best" ab recall se nahi, 5+ second warning dene se tay hota hai ---
    # kyunki pilot ke liye wahi kaam ka number hai
    best_tag = max(res, key=lambda t: (res[t].get("lead_ge5", 0), res[t]["recall"]))

    L = []
    L.append(f"Test set me **{len(crashed)} of {len(fmeta)}** faulty flights sach me "
             f"crash hui. Fault shuru hone se crash tak ka waqt: "
             f"**{surv.min():.0f}–{surv.max():.0f}s** (median {np.median(surv):.0f}s) — "
             f"yani detector ke paas itna hi mauka tha.")
    L.append("")
    L.append("Threshold hamesha **validation set** se chuna gaya (100 healthy flights), "
             "test se nahi. Aur **crash ke baad ki windows ginti me nahi aati** — "
             "palte hue drone ko \"detect\" karna koi kamaal nahi hai.")
    L.append("")
    L.append("### Asli metric: kitne second ki warning mili")
    L.append("")
    L.append("| Detector | Warning mili | Median lead | p25 (kharab 25%) | ≥3s | ≥5s | ≥10s | FA |")
    L.append("|---|---|---|---|---|---|---|---|")
    for tag, name in ROWS:
        if tag not in res:
            continue
        r = res[tag]
        nm = f"**{name}**" if tag == best_tag else name
        # Ek decimal zaroori hai: 5.5s aur 4.7s dono "5s" ban jaate hain,
        # aur wahi farq is poore table ka point hai.
        med = f"{r['lead_median']:.1f}s" if r.get("lead_median") is not None else "—"
        p25 = f"{r['lead_p25']:.1f}s" if r.get("lead_p25") is not None else "—"
        L.append(f"| {nm} | {100 * r['recall']:.0f}% | {med} | {p25} | "
                 f"{100 * r.get('lead_ge3', 0):.0f}% | {100 * r.get('lead_ge5', 0):.0f}% | "
                 f"{100 * r.get('lead_ge10', 0):.0f}% | {100 * r['false_alarm']:.0f}% |")
    L.append("")
    L.append("*Warning mili = kitne % crashes me alarm crash se pehle baja. "
             "p25 = kharab 25% cases me kitna waqt mila (ye median se zyada zaroori hai — "
             "safety me worst case maayne rakhta hai).*")
    L.append("")
    L.append("![comparison](outputs/02_comparison.png)")
    L.append("")
    L.append("### Fault type ke hisaab se (recall / median lead time)")
    L.append("")
    L.append("| Detector | " + " | ".join(k.replace("_", " ") for k in KINDS) + " |")
    L.append("|---|" + "---|" * len(KINDS))
    for tag, name in ROWS:
        if tag not in res:
            continue
        r = res[tag]
        cells = []
        for k in KINDS:
            rec = 100 * r["per_kind"][k]
            lead = (r.get("per_kind_lead") or {}).get(k)
            cells.append(f"{rec:.0f}%" + (f" / {lead:.1f}s" if lead else ""))
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("Tulna ke liye: naive PWM-spread threshold 62% faults pakad leta hai --- "
             "par wo crash se sirf **1.7 second** pehle bolta hai. Uski problem recall "
             "nahi, **timing** hai. Yahi wajah hai ki is project ka metric lead time hai.")

    readme = ROOT / "README.md"
    txt = readme.read_text(encoding="utf-8")
    body = "\n".join(L)

    if MARK in txt:
        i = txt.index(MARK)
        j = txt.index("\n---", i)
        txt = txt[:i] + MARK + "\n\n" + body + "\n" + txt[j:]
    else:
        txt += "\n\n" + body
    readme.write_text(txt, encoding="utf-8")

    print(f"  README.md updated  ({len(res)} detectors, best = {best_tag})")


if __name__ == "__main__":
    main()
