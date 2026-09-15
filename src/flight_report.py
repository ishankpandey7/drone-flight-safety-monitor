"""
STEP 7 --- Ek flight ka "black box" report, crash ke saath.
===========================================================

Yahi wo demo hai jiska maine shuru me zikr kiya tha: ek faulty flight ko
replay karo, aur dekho ki alarm crash se KITNE second pehle baja.

Teen cheezein ek tasveer me:
    A) motors ne kya kiya
    B) us fault ka jo khaas channel hai, usne kya kiya
    C) detector ka score, threshold, alarm, aur crash

Sabse zaroori baat: crash ke baad ki windows score me GINTI NAHI hoti.
Wahan drone palat raha hai --- use pakadna koi kamaal nahi hai.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import flight_alarm, K_CONSEC

ROOT = Path(__file__).resolve().parent.parent

TELL = {
    "motor_degradation": ("vibe_x", "vibration"),
    "prop_damage":       ("vibe_x", "vibration"),
    "battery_sag":       ("voltage", "battery voltage"),
    "imu_drift":         ("gyro_x", "gyro X (roll rate)"),
}
PWM = ["pwm_1", "pwm_2", "pwm_3", "pwm_4"]
COL = {"pwm_1": "#5B8DEF", "pwm_2": "#57B894", "pwm_3": "#E5484D", "pwm_4": "#C9A227"}


def evaluate(S, wt, thr, m):
    """Ek flight ka faisla: alarm kab baja, lead time kitna."""
    crash = m.get("crash_s", -1.0)
    keep = (wt <= crash) if crash > 0 else np.ones(len(wt), bool)
    alarm, widx = (False, -1)
    if keep.sum() >= K_CONSEC:
        alarm, widx = flight_alarm(S[keep], thr)
    t_alarm = float(wt[keep][widx]) if alarm else None
    lead = (crash - t_alarm) if (alarm and crash > 0) else None
    return crash, t_alarm, lead


def make_report(flight_idx, detector="baseline", outfile=None):
    d = np.load(ROOT / "data" / "processed" / "flights.npz", allow_pickle=True)
    meta = json.load(open(ROOT / "data" / "processed" / "meta.json"))
    fmeta = [mm for mm in meta if mm["split"] == "test_faulty"]
    CH = [str(c) for c in d["channels"]]
    idx = {c: i for i, c in enumerate(CH)}

    z = np.load(ROOT / "outputs" / f"{detector}_scores.npz")
    S, wt, thr = z["test_faulty"][flight_idx], z["window_times"], float(z["threshold"])

    x = d["test_faulty"][flight_idx]
    m = fmeta[flight_idx]
    t = np.arange(len(x)) / 50.0
    t0 = m["fault_start_s"]
    crash, t_alarm, lead = evaluate(S, wt, thr, m)

    # plot sirf crash tak (uske baad drone zameen pe hai, dikhane ka fayda nahi)
    t_end = min(crash + 3, t[-1]) if crash > 0 else t[-1]
    vis = t <= t_end

    tell_ch, tell_name = TELL[m["fault_kind"]]
    bad_motor = f"pwm_{m['fault_motor'] + 1}"

    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1, 1.3], "hspace": 0.25})

    # ---------------------------------------------------------- A: motors
    for c in PWM:
        bad = (c == bad_motor and m["fault_kind"] in ("motor_degradation", "prop_damage"))
        ax[0].plot(t[vis], x[vis, idx[c]], lw=1.6 if bad else 0.7,
                   alpha=1.0 if bad else 0.4, color=COL[c], zorder=3 if bad else 2,
                   label=c.replace("pwm_", "Motor ") + (" (kharab)" if bad else ""))
    ax[0].set_ylabel("PWM")
    ax[0].legend(ncol=4, fontsize=8.5, loc="upper left", framealpha=0.9)
    ax[0].set_title("A.  Chaaron motors", fontsize=11.5, fontweight="bold", loc="left")
    ax[0].set_ylim(-0.05, 1.05)

    # ---------------------------------------------------------- B: tell channel
    ax[1].plot(t[vis], x[vis, idx[tell_ch]], lw=1.0, color="#7C3AED")
    ax[1].set_ylabel(tell_ch)
    ax[1].set_title(f"B.  {tell_name}  --- is fault ka asli nishaan",
                    fontsize=11.5, fontweight="bold", loc="left")

    # ---------------------------------------------------------- C: score
    sv = wt <= t_end
    ax[2].plot(wt[sv], S[sv], lw=2.0, color="#0F172A", label="anomaly score")
    ax[2].axhline(thr, color="#E5484D", ls="--", lw=1.6, label=f"threshold ({thr:.1f})")
    ax[2].fill_between(wt[sv], thr, S[sv], where=(S[sv] > thr),
                       color="#E5484D", alpha=0.16)
    ax[2].set_ylabel("anomaly score")
    ax[2].set_xlabel("time (seconds)")
    ax[2].set_title("C.  Detector ka score", fontsize=11.5, fontweight="bold", loc="left")
    ax[2].legend(fontsize=9, loc="upper left", framealpha=0.9)

    # ---------------------------------------------------------- markers
    for a in ax:
        a.axvline(t0, color="#E8A33D", ls="--", lw=1.6, alpha=0.85)
        if t_alarm:
            a.axvline(t_alarm, color="#059669", lw=2.0, alpha=0.9)
        if crash > 0:
            a.axvline(crash, color="#B91C1C", lw=2.4, alpha=0.95)
            a.axvspan(crash, t_end, color="#B91C1C", alpha=0.07)
        a.grid(alpha=0.18)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)

    # lead time ko ek teer se dikhao --- yahi poori tasveer ka point hai
    if t_alarm and crash > 0:
        y = ax[2].get_ylim()[1] * 0.55
        ax[2].annotate("", xy=(crash, y), xytext=(t_alarm, y),
                       arrowprops=dict(arrowstyle="<->", lw=2.2, color="#059669"))
        ax[2].text((t_alarm + crash) / 2, y * 1.08, f"{lead:.0f}s ki warning",
                   ha="center", fontsize=12, fontweight="bold", color="#059669")

    yl = ax[2].get_ylim()
    ax[2].text(t0, yl[1] * 0.97, " fault shuru", color="#E8A33D",
               fontsize=9.5, fontweight="bold", va="top")
    if t_alarm:
        ax[2].text(t_alarm, yl[1] * 0.88, " ALARM", color="#059669",
                   fontsize=9.5, fontweight="bold", va="top")
    if crash > 0:
        ax[2].text(crash, yl[1] * 0.97, " CRASH", color="#B91C1C",
                   fontsize=9.5, fontweight="bold", va="top")

    if lead is not None:
        verdict, vcol = f"{lead:.0f} SECOND KI WARNING MILI", "#059669"
    elif crash > 0:
        verdict, vcol = "MISS --- crash se pehle alarm nahi baja", "#B91C1C"
    else:
        verdict, vcol = ("PAKDA GAYA (drone gira nahi)" if t_alarm else "MISS"), "#E8A33D"

    crash_txt = f"crash @ {crash:.0f}s" if crash > 0 else "gira nahi"
    fig.suptitle(
        f"{m['flight_id']}   |   {m['fault_kind']}  (motor {m['fault_motor'] + 1}, "
        f"ramp {m['fault_ramp_s']:.0f}s, {crash_txt})   |   {verdict}",
        fontsize=11.5, fontweight="bold", y=0.985, color=vcol)

    out = outfile or ROOT / "outputs" / f"03_report_{m['fault_kind']}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out), verdict, m, lead


if __name__ == "__main__":
    meta = json.load(open(ROOT / "data" / "processed" / "meta.json"))
    fmeta = [m for m in meta if m["split"] == "test_faulty"]
    z = np.load(ROOT / "outputs" / "baseline_scores.npz")
    S, wt, thr = z["test_faulty"], z["window_times"], float(z["threshold"])

    print()
    for kind in ["motor_degradation", "prop_damage", "battery_sag", "imu_drift"]:
        cands = [(i, m) for i, m in enumerate(fmeta) if m["fault_kind"] == kind]
        # Demo ke liye wo flight chuno jisme lead time sabse accha ho --- par
        # agar koi bhi nahi pakda gaya to imaandari se ek MISS dikhao.
        scored = []
        for i, m in cands:
            _, _, lead = evaluate(S[i], wt, thr, m)
            scored.append((lead if lead is not None else -1e9, i))
        pick = max(scored)[1]
        path, verdict, m, lead = make_report(pick)
        print(f"  {kind:<20} flight #{pick:<3} ramp={m['fault_ramp_s']:.0f}s  ->  {verdict}")
    print()
