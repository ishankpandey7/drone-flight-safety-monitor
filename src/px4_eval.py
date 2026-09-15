"""
Asli PX4 logs pe imtihaan --- teen alag sawaal.
===============================================

Pehle ek baat jo is poore experiment ki had tay karti hai:

    ASLI LOGS ME GROUND TRUTH NAHI HOTA.

Kisi PX4 log me ye likha nahi hota ki "motor 3 ka fault 47.2s pe shuru hua".
Isliye lead time aur recall yahan naapa hi NAHI ja sakta. Jo naapa ja sakta
hai wo hai: **detector kitni baar bolta hai**.

Aur ek aur baat jo bhoolni nahi chahiye:

    logs.px4.io pe log log KYUN karte hain? Aksar isliye ki KUCH GADBAD HUI.

Flight Review ek debugging tool hai. Toh in flights ko "healthy" maan lena
galat hai. Agar detector bolta hai, wo zaroori nahi ki jhootha alarm ho ---
ho sakta hai us flight me sach me kuch tha. Isliye main ise "false alarm
rate" nahi, **"alarm rate"** kahunga.

TEEN TESTS:

  1. TRANSFER   : mere simulator pe fit kiya model, asli logs pe chalao.
                  Ummeed: bura fail hoga. Sawaal ye nahi ki fail hoga ya
                  nahi --- sawaal ye hai ki KITNA aur KYUN.

  2. CROSS-VEHICLE : asli logs pe hi fit karo (kuch drones), aur ANJAAN
                  drones pe chalao. Ye poochta hai: kya "normal drone" ka
                  koi ek saanchha hota hai?

  3. SELF-CALIBRATED : har flight apne hi shuruaati hisse se seekhe, phir
                  baaki flight pe khud ko jaanche. Asli deployment aise hi
                  hota hai --- detector TUMHARE drone pe lagta hai.
"""

import json
from pathlib import Path

import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baseline import MahalanobisDetector
from common import flight_alarm, K_CONSEC
import px4_data as P

ROOT = Path(__file__).resolve().parent.parent
FS, WIN_S, STRIDE_S = 50.0, 8.0, 2.0


def windowize(X, win_s=WIN_S, stride_s=STRIDE_S):
    """(T, C) ek flight -> (W, 2C) features."""
    win, stride = int(win_s * FS), int(stride_s * FS)
    T, C = X.shape
    starts = list(range(0, T - win + 1, stride))
    if not starts:
        return np.zeros((0, 2 * C))
    out = np.empty((len(starts), 2 * C))
    for j, s in enumerate(starts):
        w = X[s:s + win]
        out[j, :C] = w.mean(0)
        out[j, C:] = w.std(0)
    return out


def alarm_frac(score_list, thr, k=K_CONSEC):
    """Kitne % flights me alarm baja."""
    hits = [flight_alarm(s, thr, k)[0] for s in score_list if len(s) >= k]
    return float(np.mean(hits)) if hits else 0.0, hits


def main():
    real = np.load(ROOT / "data" / "px4" / "real_flights.npz", allow_pickle=True)
    rmeta = json.load(open(ROOT / "data" / "px4" / "real_meta.json"))
    keys = [k for k in real.files if k.startswith("f")]
    R_all = [real[k] for k in keys]
    CH = P.CHANNELS

    # ==================================================================
    # DEDUP --- yahi sabse zaroori step hai.
    # ------------------------------------------------------------------
    # logs.px4.io pe log log KAI BAAR upload hote hain, har baar naye UUID
    # ke saath. Bina dedup ke wahi flight train AUR test dono me chali
    # jaati, aur "anjaan drone pe generalize kiya" wala result asal me
    # APNE HI training data pe test karna hota.
    # ==================================================================
    seen, keep = {}, []
    for i, x in enumerate(R_all):
        k = (len(x),) + tuple(np.round(np.median(x, axis=0), 4))
        if k in seen:
            continue
        seen[k] = i
        keep.append(i)
    R = [R_all[i] for i in keep]
    rmeta = [rmeta[i] for i in keep]
    print()
    print(f"  {len(R_all)} flights download hui, {len(R_all)-len(R)} DUPLICATE nikli")
    print(f"  {len(R)} unique flights, kul {sum(len(x) for x in R)/FS/60:.0f} minute")

    # vehicle proxy: hardware + battery cells + current scale.
    # airframe field bekaar hai (52/55 me sirf "Generic Quadcopter").
    vi, ci = CH.index("voltage"), CH.index("current")
    vgroup = []
    for x, m in zip(R, rmeta):
        v, c = float(np.median(x[:, vi])), float(np.median(x[:, ci]))
        vgroup.append((m["hw"], int(round(v / 3.85)), int(round(c / 10))))
    n_veh = len(set(vgroup))
    print(f"  ~{n_veh} alag vehicles (hardware + battery + current se andaza)")

    sim = np.load(ROOT / "data" / "processed" / "flights.npz", allow_pickle=True)

    # ==================================================================
    # Pehle DIAGNOSTIC: asli data mere simulator se kitna door hai?
    # ==================================================================
    print()
    print("=" * 92)
    print("  Asli data vs mera simulator --- har channel ka median")
    print("=" * 92)
    print()
    S = sim["train"].reshape(-1, len(CH))
    Rall = np.concatenate(R)
    print(f"  {'channel':<10}{'SIM median':>14}{'REAL median':>14}{'SIM p5..p95':>22}"
          f"{'REAL p5..p95':>24}")
    print("  " + "-" * 88)
    far = []
    for i, c in enumerate(CH):
        sm, rm = np.median(S[:, i]), np.median(Rall[:, i])
        s5, s95 = np.percentile(S[:, i], [5, 95])
        r5, r95 = np.percentile(Rall[:, i], [5, 95])
        # real ka median sim ke band ke andar hai ya nahi?
        out = not (s5 <= rm <= s95)
        if out:
            far.append(c)
        print(f"  {c:<10}{sm:>14.3f}{rm:>14.3f}{f'{s5:.2f}..{s95:.2f}':>22}"
              f"{f'{r5:.2f}..{r95:.2f}':>24}{'   <-- BAHAR' if out else ''}")
    print()
    print(f"  19 me se {len(far)} channels ka asli median mere simulator ke "
          f"5-95% band se BAHAR hai:")
    print(f"    {', '.join(far)}")

    # ==================================================================
    # TEST 1 --- TRANSFER
    # ==================================================================
    print()
    print("=" * 92)
    print("  TEST 1 --- simulator pe fit kiya model, asli logs pe")
    print("=" * 92)

    Ftr = np.concatenate([windowize(x) for x in sim["train"]])
    det_sim = MahalanobisDetector().fit(Ftr)
    Fva = [windowize(x) for x in sim["val"]]
    Sva = [det_sim.score(f) for f in Fva if len(f)]
    thr_sim = np.percentile(np.concatenate(Sva), 99.5)

    Sreal = [det_sim.score(windowize(x)) for x in R]
    Sreal = [s for s in Sreal if len(s)]
    fa_sim, _ = alarm_frac(Sva, thr_sim)
    fa_real, _ = alarm_frac(Sreal, thr_sim)

    med_sim = np.median(np.concatenate(Sva))
    med_real = np.median(np.concatenate(Sreal))
    print()
    print(f"  threshold (sim val ka 99.5 pct) : {thr_sim:8.1f}")
    print(f"  sim healthy flights ka median score  : {med_sim:10.1f}")
    print(f"  ASLI flights ka median score         : {med_real:10.1f}"
          f"   ({med_real/med_sim:.0f}x zyada)")
    print()
    print(f"  alarm rate, sim healthy flights : {100*fa_sim:5.1f}%")
    print(f"  alarm rate, ASLI flights        : {100*fa_real:5.1f}%")

    # ==================================================================
    # TEST 2 & 3 --- LEAVE-ONE-VEHICLE-OUT
    # ------------------------------------------------------------------
    # Pehle maine ek hi random split liya tha: 7 test flights. Us par
    # "28.6% alarm rate" kehna bemani hai --- 2 flights ka farq 28 points
    # hila deta hai.
    #
    # Isliye ab LOVO: baari baari har vehicle ko hata kar baaki pe train
    # karo, aur hataye hue vehicle pe test karo. Isse SAARI 51 flights
    # test me aati hain, aur koi bhi vehicle apne hi data pe nahi jaancha
    # jaata.
    #
    # Threshold dono tests me EK HI TARAH se aata hai (bache hue vehicles
    # ke scores ka wo point jahan 5% flights alarm karein), taaki farq
    # sirf NORMALIZATION ka rahe.
    # ==================================================================
    def pick(scores, target=0.05):
        if not scores:
            return np.inf
        cand = np.unique(np.round(np.concatenate(scores), 2))
        thr = float(cand.max()) * 2
        for c in np.sort(cand)[::-1]:
            if alarm_frac(scores, c)[0] <= target:
                thr = float(c)
            else:
                break
        return thr

    vgs = sorted(set(vgroup), key=str)
    Wr = [windowize(x) for x in R]

    print()
    print("=" * 92)
    print("  TEST 2 --- asli logs pe fit, ANJAAN drone pe test (leave-one-vehicle-out)")
    print("=" * 92)
    print()
    print(f"  {len(vgs)} vehicles, {len(R)} flights --- {len(vgs)} folds")

    hits2, per_veh2 = [], []
    for v in vgs:
        te_i = [i for i, g in enumerate(vgroup) if g == v]
        tr_i = [i for i, g in enumerate(vgroup) if g != v]
        if len(tr_i) < 5:
            continue
        # tr_i me se kuch val ke liye alag rakho (threshold ke liye)
        # NOTE: yahan pehle hash(str(v)) tha. Python me STRING ka hash har
        # process me alag hota hai (PYTHONHASHSEED), isliye har run pe
        # alag validation split banta tha aur alarm rate 5%-19% ke beech
        # uchhalta tha. crc32 deterministic hai --- ab har run same.
        rng2 = np.random.default_rng(zlib.crc32(str(v).encode()))
        tr_v = sorted(set(vgroup[i] for i in tr_i), key=str)
        # NOTE: numpy fancy-indexing yahan istemaal mat karo --- tr_v tuples
        # ki list hai, aur np.array(..., dtype=object) unhe 2D array bana
        # deta hai, jisse indexing se tuple ki jagah array milta hai.
        sel = rng2.permutation(len(tr_v))[:max(2, len(tr_v) // 4)]
        va_g = {tr_v[j] for j in sel}
        fit_i = [i for i in tr_i if vgroup[i] not in va_g]
        va_i = [i for i in tr_i if vgroup[i] in va_g]
        if not fit_i or not va_i:
            continue
        det = MahalanobisDetector().fit(np.concatenate([Wr[i] for i in fit_i]))
        thr = pick([det.score(Wr[i]) for i in va_i if len(Wr[i])])
        h = [flight_alarm(det.score(Wr[i]), thr)[0] for i in te_i if len(Wr[i]) >= K_CONSEC]
        hits2 += h
        per_veh2.append((v, float(np.mean(h)) if h else 0.0, len(h)))
    fa2 = float(np.mean(hits2))
    print(f"  alarm rate, anjaan drones : {100*fa2:5.1f}%   "
          f"({sum(hits2)}/{len(hits2)} flights)")

    # ==================================================================
    print()
    print("=" * 92)
    print("  TEST 3 --- har flight apne hi pehle 60s se seekhe (per-drone calibration)")
    print("=" * 92)
    print()
    ref_s = 60.0
    n_ref = int((ref_s - WIN_S) / STRIDE_S) + 1
    S3, ok_i = {}, []
    for i, F in enumerate(Wr):
        if len(F) < n_ref + 3 * K_CONSEC:
            continue
        try:
            dl = MahalanobisDetector().fit(F[:n_ref])
        except np.linalg.LinAlgError:
            continue
        S3[i] = dl.score(F[n_ref:])
        ok_i.append(i)
    print(f"  {len(ok_i)}/{len(R)} flights me itna data tha (60s reference + baaki)")

    hits3 = []
    for v in vgs:
        te_i = [i for i in ok_i if vgroup[i] == v]
        ot_i = [i for i in ok_i if vgroup[i] != v]
        if not te_i or len(ot_i) < 5:
            continue
        thr = pick([S3[i] for i in ot_i])
        hits3 += [flight_alarm(S3[i], thr)[0] for i in te_i]
    fa3 = float(np.mean(hits3)) if hits3 else 0.0
    print(f"  alarm rate, anjaan drones : {100*fa3:5.1f}%   "
          f"({sum(hits3)}/{len(hits3)} flights)")
    print()
    print(f"  {'':>34}{'TEST 2':>12}{'TEST 3':>12}")
    print(f"  {'(cross-vehicle model)':<34}{100*fa2:>11.1f}%{'':>12}")
    print(f"  {'(per-drone self-calibration)':<34}{'':>12}{100*fa3:>11.1f}%")

    # ------------------------------------------------------------------ plot
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    a = np.concatenate(Sva); b = np.concatenate(Sreal)
    bins = np.logspace(0, max(4, np.log10(b.max()) + 0.3), 70)
    ax[0].hist(a, bins=bins, alpha=0.75, color="#57B894", label="sim healthy")
    ax[0].hist(b, bins=bins, alpha=0.6, color="#E5484D", label="ASLI PX4 logs")
    ax[0].axvline(thr_sim, color="#111", ls="--", lw=1.6,
                  label=f"sim threshold ({thr_sim:.0f})")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("Mahalanobis score (log scale)")
    ax[0].set_ylabel("windows")
    ax[0].set_title("A.  Sim pe fit kiya model, asli data pe",
                    fontsize=12.5, fontweight="bold", loc="left")
    ax[0].legend(fontsize=9.5); ax[0].grid(alpha=0.2)

    labels = ["TEST 1\nsim-trained\n-> asli", "TEST 2\nasli-trained\n-> anjaan drone",
              "TEST 3\nkhud se seekha\n(per-drone)"]
    vals = [100 * fa_real, 100 * fa2, 100 * fa3]
    cols = ["#E5484D", "#E8A33D", "#57B894"]
    bars = ax[1].bar(labels, vals, color=cols, width=0.6)
    for bb, v in zip(bars, vals):
        ax[1].text(bb.get_x() + bb.get_width() / 2, v + 2, f"{v:.0f}%",
                   ha="center", fontsize=12, fontweight="bold")
    ax[1].axhline(5, color="#111", ls=":", lw=1.5)
    ax[1].text(2.45, 7, "5% budget", fontsize=9.5, ha="right", color="#111")
    ax[1].set_ylabel("Alarm rate, asli flights pe (%)")
    ax[1].set_ylim(0, 112)
    ax[1].set_title("B.  Kitni baar bola?", fontsize=12.5, fontweight="bold", loc="left")
    ax[1].grid(alpha=0.2, axis="y")

    for q in ax:
        q.set_axisbelow(True)
        for sp in ("top", "right"):
            q.spines[sp].set_visible(False)
    fig.suptitle("Asli PX4 logs pe imtihaan", fontsize=14.5, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "05_px4_real.png", dpi=140,
                bbox_inches="tight", facecolor="white")
    print()
    print("  saved -> outputs/05_px4_real.png")

    json.dump({"n_flights": len(R), "channels_out_of_band": far,
               "test1_alarm_rate": fa_real, "test1_median_ratio": float(med_real/med_sim),
               "test2_alarm_rate": fa2, "test3_alarm_rate": fa3},
              open(ROOT / "outputs" / "px4_result.json", "w"), indent=1)
    print()


if __name__ == "__main__":
    main()
