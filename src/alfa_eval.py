"""
ALFA pe imtihaan --- aakhir me, ASLI hardware + ASLI fault + LABELLED time.
==========================================================================

Ab tak jo naapa NAHI ja saka tha:

    simulator  : lead time aur recall dono naape, par sab synthetic
    PX4 logs   : asli hardware, par ground truth nahi -> sirf "chup rehta hai?"
    ALFA       : asli hardware AUR fault ka waqt labelled -> DETECTION naapi ja sakti hai

DESIGN --- wahi jo PX4 pe kaam kiya tha (per-flight self-calibration):

  Har flight apne HI shuruaati hisse se seekhta hai (fault se pehle ka data,
  aur onset se 5 second pehle tak hi --- taaki reference me fault ke shuruaati
  lakshan na ghus jaayein). Phir wahi detector baaki flight pe chalta hai.

  Threshold LEAVE-ONE-FLIGHT-OUT aata hai: baaki flights ke healthy hisson se,
  us flight ka apna data kabhi nahi.

DO TIMESCALE --- aur yahan ye majboori hai, shauk nahi:

  Fault ke baad median sirf 16 second hain (kam se kam 8). 8-second window
  + 3 lagatar windows = ~12 second reaction. Kai sequences me itna waqt hai
  hi nahi. Isliye FAST (2s window) detector yahan zaroori hai --- theek wahi
  cheez jo maine simulator me design ki thi.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import alfa_data as A
from baseline import MahalanobisDetector
from common import flight_alarm

ROOT = Path(__file__).resolve().parent.parent
FS = A.FS                      # 10 Hz
GUARD_S = 5.0                  # onset se itna pehle tak hi reference lete hain
K = 3

# SLOW ka stride 1s rakha hai, 2s nahi: 30 features ke liye kaafi reference
# windows chahiye, aur ALFA me pre-fault median sirf 101 second hai.
SCALES = [("SLOW (8s)", 8.0, 1.0), ("FAST (2s)", 2.0, 0.5)]
VAL_FRAC = 0.35     # pre-fault ka itna hissa THRESHOLD ke liye alag rakho


def windows(X, win_s, stride_s):
    """(T,C) -> features (W,2C) aur har window ka END time (seconds)."""
    win, stride = int(win_s * FS), int(stride_s * FS)
    T, C = X.shape
    st = list(range(0, T - win + 1, stride))
    if not st:
        return np.zeros((0, 2 * C)), np.zeros(0)
    F = np.empty((len(st), 2 * C))
    for j, s in enumerate(st):
        w = X[s:s + win]
        F[j, :C] = w.mean(0)
        F[j, C:] = w.std(0)
    return F, np.array([(s + win) / FS for s in st])


def self_calibrated_scores(X, info, win_s, stride_s):
    """Flight ko apne hi healthy shuruaati hisse se jaancho.

    Pre-fault hissa DO me bantta hai:
      FIT : model isi pe fit hota hai
      VAL : held-out HEALTHY data --- threshold isi se aata hai

    Ye bantwara zaroori hai. Pehle maine poora pre-fault fit me de diya tha,
    aur phir threshold ke liye "pre-fault ka jo hissa fit me nahi tha" maanga
    --- jo definition se KHAALI set hai. Nateeja: threshold sirf 9 healthy
    flights se aata tha (SLOW me to zero se), aur wo bemani tha.

    Returns (scores, wt, i_fit_end, i_val_end) ya None.
    """
    F, wt = windows(X, win_s, stride_s)
    if len(F) < 10:
        return None
    onset = info["onset_s"]
    healthy_end = (onset - GUARD_S) if onset else info["dur_s"]
    i_val_end = int(np.sum(wt <= healthy_end))
    i_fit_end = int(i_val_end * (1 - VAL_FRAC))
    # covariance ke liye features se kaafi zyada windows chahiye
    if i_fit_end < F.shape[1] * 1.5 or (i_val_end - i_fit_end) < K:
        return None
    try:
        det = MahalanobisDetector().fit(F[:i_fit_end])
    except np.linalg.LinAlgError:
        return None
    return det.score(F), wt, i_fit_end, i_val_end


def main():
    data, keep, dropped, rejected = A.load_all()
    print()
    print(f"  {len(data)} sequences, {len(keep)} channels @ {FS:.0f} Hz")
    print(f"  channels: {keep}")

    faults = [(X, i) for X, _, i in data if i["onset_s"] is not None]
    healthy = [(X, i) for X, _, i in data if i["is_healthy"]]
    print(f"  {len(faults)} labelled-fault flights, {len(healthy)} healthy flights")

    results = {}
    for name, win_s, stride_s in SCALES:
        S = {}
        for X, i in faults + healthy:
            r = self_calibrated_scores(X, i, win_s, stride_s)
            if r is not None:
                S[i["seq"]] = (r[0], r[1], r[2], r[3], i)
        results[name] = S
        print(f"  {name:<12} {len(S)}/{len(data)} flights me reference kaafi tha")

    # ------------------------------------------------------------------
    # Threshold LEAVE-ONE-FLIGHT-OUT.
    # Calibration data = doosri flights ka HEALTHY hissa hi
    # (healthy flights ka post-reference, aur fault flights ka pre-onset).
    # ------------------------------------------------------------------
    def healthy_part(sc, wt, i_fit_end, i_val_end, info):
        """Held-out HEALTHY scores: fit region ke baad, par fault se pehle."""
        return sc[i_fit_end:i_val_end]

    def pick_thr(parts, target=0.05):
        parts = [p for p in parts if len(p) >= K]
        if not parts:
            return np.inf
        cand = np.unique(np.round(np.concatenate(parts), 2))
        thr = float(cand.max()) * 2
        for c in np.sort(cand)[::-1]:
            fa = np.mean([flight_alarm(p, c, K)[0] for p in parts])
            if fa <= target:
                thr = float(c)
            else:
                break
        return thr

    out = {}
    for name, S in results.items():
        det_rows, fa_rows = [], []
        for seq, (sc, wt, i_fit_end, i_val_end, info) in S.items():
            others = [healthy_part(*S[s2]) for s2 in S if s2 != seq]
            thr = pick_thr(others)

            # FALSE ALARM har flight pe naapo, sirf no_failure flights pe nahi.
            # Har fault flight ka pre-fault hissa bhi LABELLED HEALTHY hai, aur
            # uska held-out VAL region is flight ke apne fit me gaya hi nahi.
            # Sirf 9 no_failure flights pe naapne se denominator 2 reh jaata
            # tha ("50% FA" = 1 out of 2), jo bemani hai.
            hp = healthy_part(sc, wt, i_fit_end, i_val_end, info)
            if len(hp) >= K:
                fa_rows.append({"seq": seq, "alarm": flight_alarm(hp, thr, K)[0],
                                "is_no_failure": info["is_healthy"]})

            if info["onset_s"] is not None:
                onset = info["onset_s"]
                # sirf wo windows jo fault ke BAAD khatam hoti hain
                m = wt > onset
                alarm, idx = (False, -1)
                if m.sum() >= K:
                    alarm, idx = flight_alarm(sc[m], thr, K)
                delay = float(wt[m][idx] - onset) if alarm else None
                det_rows.append({"seq": seq, "type": info["fault_type"],
                                 "post_s": info["post_s"], "detected": alarm,
                                 "delay_s": delay, "thr": thr})
        out[name] = (det_rows, fa_rows)

    # ------------------------------------------------------------------ report
    KINDS = ["engines", "aileron", "rudder", "elevator"]
    summary = {}
    for name in results:
        det, fa = out[name]
        rec = float(np.mean([d["detected"] for d in det]))
        dl = np.array([d["delay_s"] for d in det if d["detected"]])
        far = float(np.mean([f["alarm"] for f in fa])) if fa else 0.0
        summary[name] = {"recall": rec, "fa": far,
                         "delay_median": float(np.median(dl)) if len(dl) else None,
                         "delay_p90": float(np.percentile(dl, 90)) if len(dl) else None,
                         "n": len(det)}
        print()
        print("=" * 80)
        print(f"  {name}   --- per-flight self-calibration, threshold leave-one-out")
        print("=" * 80)
        print(f"  Recall (fault pakda)        : {100*rec:5.1f}%   "
              f"({sum(d['detected'] for d in det)}/{len(det)} flights)")
        print(f"  False alarm (healthy hisse) : {100*far:5.1f}%   "
              f"({sum(f['alarm'] for f in fa)}/{len(fa)} flights)")
        if len(dl):
            print(f"  Detection delay  median : {np.median(dl):5.1f} s")
            print(f"                   p90    : {np.percentile(dl,90):5.1f} s")
            print(f"                   max    : {dl.max():5.1f} s")
        print()
        print("  Fault type ke hisaab se:")
        for k in KINDS:
            r = [d for d in det if d["type"] == k]
            if not r:
                continue
            rr = np.mean([d["detected"] for d in r])
            dd = [d["delay_s"] for d in r if d["detected"]]
            ds = f"{np.median(dd):5.1f}s" if dd else "    -"
            print(f"    {k:<10} {100*rr:5.0f}%  ({sum(d['detected'] for d in r)}/{len(r)})"
                  f"  delay {ds}")
        summary[name]["per_kind"] = {
            k: float(np.mean([d["detected"] for d in det if d["type"] == k]))
            for k in KINDS if any(d["type"] == k for d in det)}

    # ---------------- UNION ----------------
    # UNION intersection pe nahi, UNION pe. Agar SLOW kisi flight pe calibrate
    # hi nahi ho paaya (reference chhota), to deployment me tum bas FAST chala
    # loge --- wo flight ko test se hataana nahi hai.
    dets = {n: {d["seq"]: d for d in out[n][0]} for n in results}
    fas = {n: {f["seq"]: f for f in out[n][1]} for n in results}
    seqs = sorted(set().union(*[set(d) for d in dets.values()]))
    u_det, u_delay = [], []
    for s in seqs:
        ds = [dets[n][s] for n in results if s in dets[n] and dets[n][s]["detected"]]
        u_det.append(bool(ds))
        if ds:
            u_delay.append(min(d["delay_s"] for d in ds))
    fseqs = sorted(set().union(*[set(f) for f in fas.values()]))
    u_fa = [any(fas[n][s]["alarm"] for n in results if s in fas[n]) for s in fseqs]
    dl = np.array(u_delay)
    print()
    print("=" * 80)
    print("  SLOW + FAST  (dono ka OR)")
    print("=" * 80)
    print(f"  (dono scales ne {len(seqs)} flights pe score diya --- union isi pe hai)")
    print(f"  Recall                      : {100*np.mean(u_det) if u_det else 0:5.1f}%   "
          f"({sum(u_det)}/{len(u_det)} flights)")
    print(f"  False alarm                 : {100*np.mean(u_fa) if u_fa else 0:5.1f}%   "
          f"({sum(u_fa)}/{len(u_fa)} flights)")
    if len(dl):
        print(f"  Detection delay  median     : {np.median(dl):5.1f} s")
        print(f"                   p90        : {np.percentile(dl,90):5.1f} s")
    else:
        print("  Detection delay             : koi detection nahi")
    summary["UNION"] = {
        "recall": float(np.mean(u_det)) if u_det else 0.0,
        "fa": float(np.mean(u_fa)) if u_fa else 0.0,
        "delay_median": float(np.median(dl)) if len(dl) else None,
        "delay_p90": float(np.percentile(dl, 90)) if len(dl) else None,
        "n": len(u_det)}

    # ------------------------------------------------------------------ plot
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    names = list(results) + ["UNION"]
    COL = {"SLOW (8s)": "#57B894", "FAST (2s)": "#E8A33D", "UNION": "#5B8DEF"}

    for n in names:
        if n == "UNION":
            d = dl
        else:
            d = np.array([x["delay_s"] for x in out[n][0] if x["detected"]])
        if summary[n].get("recall") is None:
            continue
        if not len(d):
            continue
        g = np.linspace(0, 40, 200)
        y = [100 * np.mean(d <= t) * summary[n]["recall"] for t in g]
        ax[0].plot(g, y, color=COL[n], lw=3.0 if n == "UNION" else 2.0,
                   label=f"{n}  ({100*summary[n]['recall']:.0f}% recall)")
    ax[0].set_xlabel("Detection delay (seconds after fault onset)")
    ax[0].set_ylabel("% faults pakde")
    ax[0].set_title("A.  Kitni jaldi pakda?", fontsize=12.5, fontweight="bold", loc="left")
    ax[0].legend(fontsize=9.5); ax[0].grid(alpha=0.2); ax[0].set_ylim(0, 103)

    # seq -> fault type map (kisi bhi scale se, jo mile)
    ftype = {}
    for n in results:
        for sq, d in dets[n].items():
            ftype.setdefault(sq, d["type"])

    w = 0.26
    xs = np.arange(len(KINDS))
    for k, n in enumerate(names):
        vals = []
        for kd in KINDS:
            sel = [sq for sq in seqs if ftype.get(sq) == kd]
            if not sel:
                vals.append(0.0); continue
            if n == "UNION":
                v = np.mean([any(sq in dets[m] and dets[m][sq]["detected"]
                                 for m in results) for sq in sel])
            else:
                sel_n = [sq for sq in sel if sq in dets[n]]
                v = np.mean([dets[n][sq]["detected"] for sq in sel_n]) if sel_n else 0.0
            vals.append(100 * v)
        ax[1].bar(xs + (k - 1) * w, vals, w * 0.9, color=COL[n], label=n)
        for x, v in zip(xs + (k - 1) * w, vals):
            if v > 3:
                ax[1].text(x, v + 2, f"{v:.0f}", ha="center", fontsize=8)
    counts = {kd: sum(1 for sq in seqs if ftype.get(sq) == kd) for kd in KINDS}
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([f"{k}\n({counts[k]})" for k in KINDS], fontsize=9.5)
    ax[1].set_ylabel("Recall (%)")
    ax[1].set_title("B.  Fault type ke hisaab se", fontsize=12.5,
                    fontweight="bold", loc="left")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.2, axis="y"); ax[1].set_ylim(0, 118)

    for a in ax:
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("CMU ALFA --- asli UAV, asli faults, labelled onset",
                 fontsize=14.5, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "06_alfa.png", dpi=140,
                bbox_inches="tight", facecolor="white")
    print()
    print("  saved -> outputs/06_alfa.png")
    json.dump(summary, open(ROOT / "outputs" / "alfa_result.json", "w"), indent=1)
    print()


if __name__ == "__main__":
    main()
