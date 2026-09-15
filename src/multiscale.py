"""
STEP 8 --- Do detectors, do alag timescale pe.
==============================================

KYUN:

Ab tak ka poora system 8-second window pe chalta hai. Alarm ke liye 3 lagatar
windows chahiye. Iska matlab:

    window fill hone me      : 8 second
    3 lagatar windows        : +4 second
    ---------------------------------------
    minimum reaction time    : ~10-12 second

**Jo fault 10 second se kam me drone ko maar deta hai, wo is design se pakda
ja hi nahi sakta** --- chahe model kitna bhi accha ho. Ye limit WINDOW SIZE ne
banayi hai, model ne nahi. Isliye bade LSTM se kuch nahi hoga.

Dashboard me ye saaf dikhta hai: ek flight jisme motor 6 second me marr gaya,
uspe status aata hai "CRASHED (bina warning)".

ILAAJ:

Do detectors saath chalao, alag alag timescale pe --- aur dono ke alarm ko OR
kar do:

    SLOW : 8s window, 2s stride   -> zyada context, kam shor, ~10s reaction
    FAST : 2s window, 0.5s stride -> kam context, zyada shor, ~3s reaction

SABSE ZAROORI BAAT (warna ye poora experiment bekaar hai):

    Do detectors = do mauke jhootha alarm bajane ke.

    Agar main dono ko ALAG ALAG 5% false alarm pe set karun, to milakar system
    ~10% pe pahunch jayega. Phir "zyada faults jaldi pakde" wali jeet asal me
    false alarms KHAREED kar aayi hogi, behtar detection se nahi.

    Isliye dono ke threshold ek saath aise chune jaate hain ki MILAKAR wahi
    5% budget rahe --- bilkul wahi jo 8s-only system ko mila tha. Tab hi
    seedhi tulna ban sakti hai.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common
from common import FS, alarm_times, pick_threshold, report_from_alarms
from baseline import MahalanobisDetector

ROOT = Path(__file__).resolve().parent.parent
TARGET_FA = 0.05

# (naam, window seconds, stride seconds, k_consec)
SCALES = [
    ("SLOW  (8s window)", 8.0, 2.0, 3),
    ("FAST  (2s window)", 2.0, 0.5, 3),
]


def windowize(X, win_s, stride_s):
    """(N, T, C) -> features (N, W, 2C) aur har window ka END time."""
    win, stride = int(win_s * FS), int(stride_s * FS)
    N, T, C = X.shape
    starts = list(range(0, T - win + 1, stride))
    out = np.empty((N, len(starts), 2 * C), dtype=np.float64)
    for j, s in enumerate(starts):
        w = X[:, s:s + win, :]
        out[:, j, :C] = w.mean(axis=1)
        out[:, j, C:] = w.std(axis=1)
    wt = np.array([(s + win) / FS for s in starts])
    return out, wt


def union(alarm_lists):
    """Kai detectors ke alarm times se: sabse PEHLA alarm (ya None)."""
    n = len(alarm_lists[0])
    out = []
    for i in range(n):
        ts = [a[i] for a in alarm_lists if a[i] is not None]
        out.append(min(ts) if ts else None)
    return out


def main():
    d = np.load(ROOT / "data" / "processed" / "flights.npz", allow_pickle=True)
    meta = json.load(open(ROOT / "data" / "processed" / "meta.json"))
    fmeta = [m for m in meta if m["split"] == "test_faulty"]

    print()
    print("Windows bana rahe hain (do timescale)...")
    det, S = {}, {}
    for name, win_s, stride_s, k in SCALES:
        Ftr, wt = windowize(d["train"], win_s, stride_s)
        mh = MahalanobisDetector().fit(Ftr.reshape(-1, Ftr.shape[-1]))

        def sc(X, _mh=mh, _w=win_s, _s=stride_s):
            F, _ = windowize(X, _w, _s)
            n, w, f = F.shape
            return _mh.score(F.reshape(-1, f)).reshape(n, w)

        S[name] = {
            "val": sc(d["val"]), "healthy": sc(d["test_healthy"]),
            "faulty": sc(d["test_faulty"]), "wt": wt, "k": k,
        }
        det[name] = mh
        print(f"  {name:<20} {len(wt):4d} windows/flight   "
              f"(pehla faisla {wt[0]:.1f}s pe)")

    # ------------------------------------------------------------------
    # JOINT THRESHOLD CALIBRATION
    # ------------------------------------------------------------------
    # Ek hi "strictness" knob (alpha) dono pe lagao, aur sabse DHEELA alpha
    # dhoondho jispe MILAKAR false alarm rate abhi bhi budget ke andar ho.
    #
    # Alpha ghatane se dono ke threshold uthte hain, yani union FA girta hai.
    # Isliye bade alpha se chhote ki taraf chalo aur pehla valid utha lo.
    print()
    print(f"Joint threshold calibration (target: MILAKAR <= {100*TARGET_FA:.0f}% false alarm)")
    print()
    alphas = [0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01, 0.0075,
              0.005, 0.003, 0.002, 0.001]
    chosen = None
    for a in alphas:
        thrs = {n: pick_threshold(S[n]["val"], a, S[n]["k"]) for n, *_ in SCALES}
        av = [alarm_times(S[n]["val"], S[n]["wt"], thrs[n], None, S[n]["k"])
              for n, *_ in SCALES]
        ufa = float(np.mean([x is not None for x in union(av)]))
        mark = ""
        if ufa <= TARGET_FA and chosen is None:
            chosen = (a, thrs)
            mark = "  <-- chuna gaya"
        print(f"  alpha={a:<7.4f}  union val FA = {100*ufa:5.1f}%{mark}")
        if chosen:
            break

    if chosen is None:
        raise SystemExit("  Koi alpha kaam nahi kiya --- FAST detector bahut shor bhara hai")
    alpha, thrs = chosen
    print()
    for n, *_ in SCALES:
        print(f"  {n:<20} threshold = {thrs[n]:.3f}   (apna alpha {alpha})")

    # ------------------------------------------------------------------
    # EVALUATE
    # ------------------------------------------------------------------
    A_h, A_f, A_fu = {}, {}, {}
    for n, *_ in SCALES:
        s, k, wt = S[n], S[n]["k"], S[n]["wt"]
        A_h[n] = alarm_times(s["healthy"], wt, thrs[n], None, k)
        A_f[n] = alarm_times(s["faulty"], wt, thrs[n], fmeta, k)
        A_fu[n] = alarm_times(s["faulty"], wt, thrs[n], None, k)

    results = {}
    for n, *_ in SCALES:
        results[n] = report_from_alarms(f"{n}  akela", A_h[n], A_f[n], fmeta, A_fu[n])

    names = [n for n, *_ in SCALES]
    results["UNION"] = report_from_alarms(
        "SLOW + FAST  (dono ka OR, milakar wahi 5% FA budget)",
        union([A_h[n] for n in names]),
        union([A_f[n] for n in names]),
        fmeta,
        union([A_fu[n] for n in names]))

    # ------------------------------------------------------------------
    # Asli sawaal: kya FAST ne wo flights bachayi jo SLOW se chhoot rahi thi?
    # ------------------------------------------------------------------
    slow, fast = names
    print("=" * 78)
    print("  Kya FAST detector ne kuch NAYA joda?")
    print("=" * 78)
    print()
    only_fast, both, only_slow = [], [], []
    for i, m in enumerate(fmeta):
        s_ok, f_ok = A_f[slow][i] is not None, A_f[fast][i] is not None
        if f_ok and not s_ok:
            only_fast.append(i)
        elif s_ok and not f_ok:
            only_slow.append(i)
        elif s_ok:
            both.append(i)

    print(f"  sirf SLOW ne pakdi   : {len(only_slow):2d} flights")
    print(f"  sirf FAST ne pakdi   : {len(only_fast):2d} flights")
    print(f"  dono ne pakdi        : {len(both):2d} flights")
    print(f"  kisi ne nahi pakdi   : {len(fmeta) - len(only_slow) - len(only_fast) - len(both):2d} flights")

    if only_fast:
        print()
        print("  Jo flights SLOW se chhoot rahi thi, FAST ne bachayi:")
        for i in only_fast:
            m = fmeta[i]
            surv = m.get("survived_s", -1)
            lead = m["crash_s"] - A_f[fast][i] if m["crash_s"] > 0 else None
            ls = f"{lead:.1f}s ki warning" if lead is not None else "gira nahi"
            print(f"    {m['fault_kind']:<20} fault se crash tak {surv:5.1f}s "
                  f"-> {ls}")

    # kitni jaldi maar dene wale faults FAST ne bachaye?
    print()
    fastest = sorted([m for m in fmeta if m.get("crash_s", -1) > 0],
                     key=lambda m: m["survived_s"])[:15]
    idxs = [fmeta.index(m) for m in fastest]
    print(f"  Sabse tez marne wale 15 faults (fault se crash tak "
          f"{fastest[0]['survived_s']:.0f}-{fastest[-1]['survived_s']:.0f}s):")
    for lbl, A in [("SLOW akela", A_f[slow]), ("FAST akela", A_f[fast]),
                   ("dono (OR) ", union([A_f[n] for n in names]))]:
        got = sum(1 for i in idxs if A[i] is not None)
        print(f"    {lbl}  {got:2d}/15 pakdi   {'#' * got}")

    # ------------------------------------------------------------------
    # SENSITIVITY CHECK
    # ------------------------------------------------------------------
    # Union ka realized TEST false alarm 4% hai, SLOW-alone ka 0%. Wo 25 me
    # se sirf 1 flight hai --- par sawaal wajib hai: kya union ki jeet us
    # ek jhoothe alarm se KHAREEDI gayi hai?
    #
    # Isliye alpha ko aur sakht karke dekhte hain: jab union ka test FA bhi
    # 0% ho jaaye, tab bhi kya wo SLOW-alone se aage hai?
    #
    # (Threshold TEST dekh kar tune karna deploy ke liye cheating hai --- ye
    # sirf ek sawaal ka jawab hai, shipped setting nahi.)
    print()
    print("=" * 78)
    print("  Sensitivity: kya jeet us 1 false alarm se khareedi gayi?")
    print("=" * 78)
    print()
    print(f"  {'alpha':>8}{'test FA':>10}{'recall':>9}{'median':>9}{'p25':>8}"
          f"{'>=5s':>8}{'>=10s':>8}")
    print("  " + "-" * 62)
    n_crashed = sum(1 for m in fmeta if m.get("crash_s", -1) > 0)
    rows = []
    for a in [0.05, 0.04, alpha, 0.025, 0.02, 0.015, 0.01, 0.0075, 0.005, 0.003, 0.002, 0.001]:
        th = {n: pick_threshold(S[n]["val"], a, S[n]["k"]) for n, *_ in SCALES}
        ah = union([alarm_times(S[n]["healthy"], S[n]["wt"], th[n], None, S[n]["k"])
                    for n, *_ in SCALES])
        af = union([alarm_times(S[n]["faulty"], S[n]["wt"], th[n], fmeta, S[n]["k"])
                    for n, *_ in SCALES])
        fa_t = np.mean([x is not None for x in ah])
        rec = np.mean([x is not None for x in af])
        L = np.array([m["crash_s"] - x for x, m in zip(af, fmeta)
                      if x is not None and m.get("crash_s", -1) > 0])
        med = np.median(L) if len(L) else float("nan")
        p25 = np.percentile(L, 25) if len(L) else float("nan")
        g5 = np.sum(L >= 5) / n_crashed if len(L) else 0
        g10 = np.sum(L >= 10) / n_crashed if len(L) else 0
        rows.append({"alpha": a, "fa": float(fa_t), "recall": float(rec),
                     "ge5": float(g5), "ge10": float(g10)})
        tag = "  <- chuna gaya" if a == alpha else ""
        print(f"  {a:>8.4f}{100*fa_t:>9.0f}%{100*rec:>8.0f}%{med:>8.1f}s"
              f"{p25:>7.1f}s{100*g5:>7.0f}%{100*g10:>7.0f}%{tag}")
    print("  " + "-" * 62)
    sl = results[names[0]]
    print(f"  {'SLOW akela':>8}{100*sl['false_alarm']:>9.0f}%{100*sl['recall']:>8.0f}%"
          f"{sl['lead_median']:>8.1f}s{sl['lead_p25']:>7.1f}s"
          f"{100*sl['lead_ge5']:>7.0f}%{100*sl['lead_ge10']:>7.0f}%")

    # Nateeja DATA se nikalo, apni umeed se nahi.
    # Sawaal: kya koi aisa operating point hai jahan union ka test FA
    # SLOW-alone se zyada NA ho, aur phir bhi wo har metric pe aage ho?
    fair = [r for r in rows if r["fa"] <= sl["false_alarm"] + 1e-9
            and r["recall"] >= sl["recall"]
            and r["ge5"] >= sl["lead_ge5"]
            and r["ge10"] >= sl["lead_ge10"]]
    print()
    if fair:
        b = max(fair, key=lambda r: (r["ge5"], r["ge10"]))
        print(f"  >> JEET ASLI HAI. alpha={b['alpha']:.4f} pe union ka false alarm"
              f" {100*b['fa']:.0f}% hai")
        print(f"     (SLOW-alone jitna ya kam), aur phir bhi wo har metric pe aage hai:")
        print(f"       recall {100*b['recall']:.0f}% vs {100*sl['recall']:.0f}%   |   "
              f">=5s {100*b['ge5']:.0f}% vs {100*sl['lead_ge5']:.0f}%   |   "
              f">=10s {100*b['ge10']:.0f}% vs {100*sl['lead_ge10']:.0f}%")
        print(f"     Yani gain us extra false alarm se KHAREEDA nahi gaya.")
    else:
        print("  >> SAAVDHAN: jis bhi alpha pe union ka false alarm SLOW jitna hota hai,")
        print("     wahan wo SLOW se aage nahi rehta. Yani jeet false alarms se")
        print("     khareedi gayi thi --- ye sudhaar asli nahi hai.")
    print()

    # ------------------------------------------------------------------ plot
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4))
    COL = {names[0]: "#57B894", names[1]: "#E8A33D", "UNION": "#5B8DEF"}
    n_crashed = sum(1 for m in fmeta if m.get("crash_s", -1) > 0)

    grid = np.linspace(0, 40, 200)
    for key, A in [(names[0], A_f[names[0]]), (names[1], A_f[names[1]]),
                   ("UNION", union([A_f[n] for n in names]))]:
        leads = np.array([(m["crash_s"] - a) if (a is not None and m["crash_s"] > 0) else -1.0
                          for a, m in zip(A, fmeta) if m.get("crash_s", -1) > 0])
        y = [100 * np.mean(leads >= g) for g in grid]
        lw = 3.0 if key == "UNION" else 1.9
        ax[0].plot(grid, y, color=COL[key], lw=lw,
                   label=f"{key.strip()}  ({results[key]['recall']*100:.0f}% recall)")
    ax[0].axvline(5, color="#666", ls=":", lw=1.4)
    ax[0].set_xlabel("Warning ka waqt (seconds before crash)")
    ax[0].set_ylabel(f"% of {n_crashed} crashes")
    ax[0].set_title("A.  Kitne second ki warning mili?",
                    fontsize=12.5, fontweight="bold", loc="left")
    ax[0].legend(fontsize=9.5); ax[0].grid(alpha=0.2)
    ax[0].set_xlim(0, 40); ax[0].set_ylim(0, 103)

    # B: recall vs "fault kitni jaldi maarta hai"
    surv = np.array([m["survived_s"] for m in fmeta if m.get("crash_s", -1) > 0])
    order = np.argsort(surv)
    crashed_idx = [i for i, m in enumerate(fmeta) if m.get("crash_s", -1) > 0]
    bins = [(0, 10), (10, 20), (20, 30), (30, 60)]
    xs = np.arange(len(bins))
    w = 0.26
    for k, (key, A) in enumerate([(names[0], A_f[names[0]]), (names[1], A_f[names[1]]),
                                  ("UNION", union([A_f[n] for n in names]))]):
        vals = []
        for lo, hi in bins:
            sel = [i for i in crashed_idx if lo <= fmeta[i]["survived_s"] < hi]
            vals.append(100 * np.mean([A[i] is not None for i in sel]) if sel else 0)
        ax[1].bar(xs + (k - 1) * w, vals, w * 0.92, color=COL[key], label=key.strip())
        for x, v in zip(xs + (k - 1) * w, vals):
            ax[1].text(x, v + 2, f"{v:.0f}", ha="center", fontsize=8.5, color="#333")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([f"{lo}-{hi}s\n({sum(1 for i in crashed_idx if lo <= fmeta[i]['survived_s'] < hi)} flights)"
                           for lo, hi in bins], fontsize=9)
    ax[1].set_xlabel("Fault shuru hone se crash tak kitna waqt tha")
    ax[1].set_ylabel("Recall (%)")
    ax[1].set_title("B.  Tez marne wale faults pe kaun kaam karta hai?",
                    fontsize=12.5, fontweight="bold", loc="left", pad=22)
    # legend bars ke UPAR wali khaali patti me --- warna wo pehle group ko dhak deta hai
    ax[1].legend(fontsize=9, ncol=3, loc="upper center", framealpha=0.95,
                 bbox_to_anchor=(0.5, 1.0))
    ax[1].grid(alpha=0.2, axis="y")
    ax[1].set_ylim(0, 130)

    for a in ax:
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)

    fig.suptitle("Do timescale: kya chhota window tez marne wale faults bachata hai?",
                 fontsize=14.5, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "04_multiscale.png", dpi=140,
                bbox_inches="tight", facecolor="white")
    print()
    print("  saved -> outputs/04_multiscale.png")

    with open(ROOT / "outputs" / "multiscale_result.json", "w") as f:
        json.dump({"alpha": alpha, "thresholds": thrs,
                   "results": {k: v for k, v in results.items()}}, f, indent=1)
    print()


if __name__ == "__main__":
    main()
