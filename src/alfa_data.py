"""
CMU ALFA dataset --- asli UAV flights, asli faults, aur LABELLED fault time.
===========================================================================

Ye dataset is project ka aakhri khula sawaal band kar sakta hai:

    "Kya ye tareeka ASLI hardware pe ASLI fault pakadta hai?"

PX4 logs pe sirf ye naapa ja saka tha ki detector healthy flights pe chup
rehta hai (ground truth nahi tha). ALFA me fault ka WAQT labelled hai.

TEEN BAATEIN JO YAHAN ALAG HAIN --- aur jinhe chhupana nahi chahiye:

1. **Ye fixed-wing hai, quadcopter nahi.** Carbon-Z T-28: ek engine, aur
   aileron/elevator/rudder. `pwm_1..4` jaisa kuch hai hi nahi. Toh ye
   "mera quad detector asli quads pe" ka test NAHI hai. Ye "kya ye TAREEKA
   asli labelled faults pakadta hai" ka test hai.

2. **Sample rates bahut dheeme hain.** rc-out 2.9 Hz, attitude 2.5 Hz,
   raw IMU 10 Hz. Mera simulator 50 Hz pe tha. 50 Hz pe resample karna
   bemani hoga --- wo interpolation hai, information nahi. Isliye 10 Hz.

3. **Fault ke baad bahut kam waqt hai** --- median 16 second, kam se kam 8.
   8-second window + 3 lagatar windows = ~12s reaction time. Kai sequences
   me itna waqt hai hi nahi. Isliye yahan chhoti windows zaroori hain.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "data" / "alfa" / "x" / "processed"
FS = 10.0          # Hz --- raw IMU ki rate; isse tez jaana jhooth hoga

# (channel naam, file suffix, CSV column)
SPEC = [
    ("gyro_x",    "mavros-imu-data_raw", "field.angular_velocity.x"),
    ("gyro_y",    "mavros-imu-data_raw", "field.angular_velocity.y"),
    ("gyro_z",    "mavros-imu-data_raw", "field.angular_velocity.z"),
    ("accel_x",   "mavros-imu-data_raw", "field.linear_acceleration.x"),
    ("accel_y",   "mavros-imu-data_raw", "field.linear_acceleration.y"),
    ("accel_z",   "mavros-imu-data_raw", "field.linear_acceleration.z"),
    ("alt",       "mavros-global_position-rel_alt", "field.data"),
    ("vz",        "mavros-local_position-velocity", "field.twist.linear.z"),
    # Airspeed fixed-wing ka sabse keemti channel hai: engine fail hone pe
    # measured, commanded se neeche gir jaata hai. Dono rakhte hain ---
    # Mahalanobis unka RISHTA khud pakad lega (covariance se).
    ("airspeed",     "mavros-nav_info-airspeed", "field.measured"),
    ("airspeed_cmd", "mavros-nav_info-airspeed", "field.commanded"),
    # channels0/6/7 constant hain, aur channels5 bilkul channels4 ki copy hai
    ("rc_1",      "mavros-rc-out", "field.channels1"),
    ("rc_2",      "mavros-rc-out", "field.channels2"),
    ("rc_3",      "mavros-rc-out", "field.channels3"),
    ("rc_4",      "mavros-rc-out", "field.channels4"),
    # battery in flights me instrumented hi nahi hai (voltage poori flight
    # 0.0, current 0.01) --- degenerate filter inhe apne aap hata dega
    ("voltage",   "mavros-battery", "field.voltage"),
    ("current",   "mavros-battery", "field.current"),
]
QUAT = "mavros-imu-data"   # orientation -> roll, pitch


def _hold(t_src, v_src, t_dst):
    """Nearest-hold resample. Interpolate NAHI karte --- 2.9 Hz ke signal ko
    smooth line bana dena aise transitions bana deta hai jo hue hi nahi."""
    i = np.clip(np.searchsorted(t_src, t_dst, side="right") - 1, 0, len(v_src) - 1)
    return v_src[i]


def _read(d, seq, suffix):
    p = d / f"{seq}-{suffix}.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    return df if "%time" in df.columns and len(df) > 3 else None


def fault_onset(d, seq):
    """Ground truth: fault kis waqt shuru hua (flight start se seconds).
    None agar koi fault nahi (no_failure sequence)."""
    ref = _read(d, seq, "mavros-imu-data_raw")
    if ref is None:
        return None, None
    t0 = ref["%time"].values[0] / 1e9
    best = None
    for g in sorted(d.glob(f"{seq}-failure_status-*.csv")):
        df = pd.read_csv(g)
        if "field.data" not in df:
            continue
        v = df["field.data"].values
        tt = df["%time"].values / 1e9
        idx = np.where(v != 0)[0]
        if len(idx):
            cand = (float(tt[idx[0]] - t0), g.stem.split("failure_status-")[-1])
            if best is None or cand[0] < best[0]:
                best = cand
    return (best if best else (None, None))


def load_sequence(seq_dir):
    """Ek sequence -> (X (T,C) float32, channels, info) ya (None, None, reason)."""
    d = Path(seq_dir)
    seq = d.name
    ref = _read(d, seq, "mavros-imu-data_raw")
    if ref is None:
        return None, None, "no imu_data_raw"

    t = ref["%time"].values / 1e9
    t0, t1 = t[0], t[-1]
    n = int((t1 - t0) * FS)
    if n < 300:                               # 30 second se chhoti
        return None, None, f"bahut chhoti ({(t1-t0):.0f}s)"
    tg = t0 + np.arange(n) / FS

    ch, names, missing = [], [], []
    for name, suffix, col in SPEC:
        df = _read(d, seq, suffix)
        if df is None or col not in df.columns:
            missing.append(name)
            continue
        v = pd.to_numeric(df[col], errors="coerce").values.astype(float)
        ts = df["%time"].values / 1e9
        ok = np.isfinite(v)
        if ok.sum() < 3:
            missing.append(name)
            continue
        ch.append(_hold(ts[ok], v[ok], tg))
        names.append(name)

    # attitude quaternion -> roll, pitch
    q = _read(d, seq, QUAT)
    if q is not None and "field.orientation.w" in q.columns:
        ts = q["%time"].values / 1e9
        qx, qy, qz, qw = (q[f"field.orientation.{a}"].values.astype(float)
                          for a in ("x", "y", "z", "w"))
        roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
        ch.append(_hold(ts, roll, tg));  names.append("roll")
        ch.append(_hold(ts, pitch, tg)); names.append("pitch")
    else:
        missing.append("roll/pitch")

    X = np.stack(ch, axis=-1)

    # Vibration ka koi direct channel nahi hai. Raw accel ka 1-second
    # rolling std hi sabse kareeb hai --- wahi cheez PX4 ka vibe metric
    # bhi naapta hai.
    if all(f"accel_{a}" in names for a in "xyz"):
        ai = [names.index(f"accel_{a}") for a in "xyz"]
        w = int(FS)
        mag = np.sqrt((X[:, ai] ** 2).sum(1))
        s = pd.Series(mag).rolling(w, min_periods=1).std().fillna(0).values
        X = np.concatenate([X, s[:, None]], axis=-1)
        names.append("vibe")

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Yahan channels DROP NAHI karte. Agar har flight apne hisaab se drop
    # kare to har flight ka feature-set alag ho jaayega, aur Mahalanobis
    # me flights aapas me compare hi nahi ho payengi. Degenerate channels
    # GLOBALLY load_all() me hatte hain.
    dropped = [names[i] for i in range(X.shape[1]) if float(np.std(X[:, i])) <= 1e-9]

    onset, ftype = fault_onset(d, seq)
    info = {
        "seq": seq, "dur_s": round(n / FS, 1), "n": n,
        "onset_s": onset, "fault_type": ftype,
        "post_s": round((n / FS) - onset, 1) if onset else None,
        "missing": missing, "dropped": dropped,
        "is_healthy": ("no_failure" in seq),
    }
    return X, names, info


def load_all(min_present=0.95):
    """Saari sequences load karo, EK HI fixed channel-set ke saath.

    Do pass:
      1. sab load karo, aur har channel ke liye gino ki wo kitni flights me
         sach me badalta hai
      2. sirf wo channels rakho jo >= min_present flights me kaam ke hain,
         aur har flight ko usi set pe kaat do

    Bina iske har flight ka feature-set alag hota hai (ALFA me kuch flights
    me airspeed sensor tha hi nahi --- 47 me se 34 me wo poori tarah ZERO
    hai; battery kabhi instrumented hi nahi tha), aur Mahalanobis me flights
    aapas me compare hi nahi ho sakti.
    """
    raw, rejected = [], []
    for d in sorted(BASE.iterdir()):
        if not d.is_dir():
            continue
        X, names, info = load_sequence(d)
        if X is None:
            rejected.append({"seq": d.name, "reject": names or info})
        else:
            raw.append((X, names, info))

    # pass 1 --- har channel kitni flights me sach me badalta hai
    from collections import Counter
    good = Counter()
    for X, names, _ in raw:
        for i, nm in enumerate(names):
            if float(np.std(X[:, i])) > 1e-9:
                good[nm] += 1
    n = len(raw)
    order = [nm for nm, _, _ in SPEC] + ["roll", "pitch", "vibe"]
    keep = [nm for nm in order if good.get(nm, 0) >= min_present * n]
    dropped_global = [(nm, good.get(nm, 0)) for nm in order
                      if nm in {x for _, nms, _ in raw for x in nms}
                      and nm not in keep]

    # pass 2 --- sabko usi set pe kaato
    out = []
    for X, names, info in raw:
        idx = [names.index(nm) for nm in keep]
        info["channels"] = keep
        out.append((X[:, idx], list(keep), info))
    return out, keep, dropped_global, rejected
