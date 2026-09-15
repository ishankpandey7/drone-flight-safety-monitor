"""
Asli PX4 flight logs laao aur unhe hamare 19-channel schema me badlo.
=====================================================================

TEEN CHEEZEIN JO YAHAN GALAT HO SAKTI THI:

1. **SITL logs.** logs.px4.io pe bahut se logs `PX4_SITL` hardware ke hain ---
   wo software-in-the-loop SIMULATION hain, asli drone nahi. Un pe "asli data
   pe test" karna matlab kisi AUR ke simulator pe test karna. Reject.

2. **Jo flights hui hi nahi.** Bench tests, failed arms, parameter setups.
   Pehla log jo maine download kiya usme PWM 900 pe atka tha aur current
   0.3 A --- drone kabhi arm hi nahi hua. Reject.

3. **Sample rates alag alag hain.** sensor_combined 250 Hz, attitude 32 Hz,
   actuator_outputs 19 Hz, battery 5 Hz. Sabko 50 Hz pe laana padega ---
   aur ye ek ASLI fidelity gap hai: mere simulator me PWM 50 Hz pe hai,
   asli log me 19 Hz. Chhoti window wala detector isi detail pe jeeta tha.
"""

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "px4"
RAW.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (px4-anomaly-study; educational)"}

# simulator ka schema --- bilkul wahi order
CHANNELS = [
    "roll", "pitch",
    "gyro_x", "gyro_y", "gyro_z",
    "accel_x", "accel_y", "accel_z",
    "alt", "vz",
    "pwm_1", "pwm_2", "pwm_3", "pwm_4",
    "voltage", "current",
    "vibe_x", "vibe_y", "vibe_z",
]
FS = 50.0


# ---------------------------------------------------------------- index

def fetch_index(pages=8, per_page=150):
    """logs.px4.io ka public index (DataTables server-side API)."""
    out = []
    for pg in range(pages):
        p = {"draw": "1", "start": str(pg * per_page), "length": str(per_page),
             "search[value]": "", "search[regex]": "false",
             "order[0][column]": "1", "order[0][dir]": "desc"}
        for i in range(10):
            p[f"columns[{i}][data]"] = str(i)
            p[f"columns[{i}][name]"] = ""
            p[f"columns[{i}][searchable]"] = "true"
            p[f"columns[{i}][orderable]"] = "true"
            p[f"columns[{i}][search][value]"] = ""
            p[f"columns[{i}][search][regex]"] = "false"
        url = "https://logs.px4.io/browse_data_retrieval?" + urllib.parse.urlencode(p)
        req = urllib.request.Request(url, headers={**UA, "X-Requested-With": "XMLHttpRequest"})
        d = json.loads(urllib.request.urlopen(req, timeout=120).read())
        for r in d["data"]:
            m = re.search(r"log=([0-9a-f-]{36})", r[1] or "")
            if not m:
                continue
            dm = re.match(r"(?:(\d+)m)?(\d+)s", r[7] or "")
            secs = (int(dm.group(1) or 0) * 60 + int(dm.group(2))) if dm else 0
            out.append({"uid": m.group(1), "vehicle": r[3], "airframe": r[4],
                        "hw": r[5], "sw": r[6], "secs": secs, "modes": r[9]})
        if len(d["data"]) < per_page:
            break
        time.sleep(0.5)          # server pe daya karo
    return out


def candidates(idx, min_s=180, max_s=1200):
    """Sirf asli hardware ke quadrotor logs, theek lambai ke."""
    out = []
    for e in idx:
        if e["vehicle"] != "Quadrotor":
            continue
        if "SITL" in (e["hw"] or "").upper():     # <-- simulation, reject
            continue
        if not (min_s <= e["secs"] <= max_s):
            continue
        out.append(e)
    return out


def download(uid, timeout=240):
    path = RAW / f"{uid}.ulg"
    if path.exists() and path.stat().st_size > 10_000:
        return path
    url = f"https://logs.px4.io/download?log={uid}"
    req = urllib.request.Request(url, headers=UA)
    data = urllib.request.urlopen(req, timeout=timeout).read()
    if len(data) < 10_000:
        return None
    path.write_bytes(data)
    return path


# ---------------------------------------------------------------- parsing

def _resample(t_src, v_src, t_dst):
    """Nearest-hold resample (interpolation nahi --- sensor readings hold hoti hain,
    aur interpolate karne se aise transitions ban jaate hain jo hue hi nahi)."""
    idx = np.searchsorted(t_src, t_dst, side="right") - 1
    idx = np.clip(idx, 0, len(v_src) - 1)
    return v_src[idx]


def _quat_to_rp(q0, q1, q2, q3):
    """Quaternion -> roll, pitch (radians)."""
    roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
    s = np.clip(2 * (q0 * q2 - q3 * q1), -1.0, 1.0)
    pitch = np.arcsin(s)
    return roll, pitch


def _pick_battery(D):
    """Jis battery instance me current sach me badalta ho use lo."""
    best, best_var = None, -1
    for (name, mid), d in D.items():
        if name != "battery_status":
            continue
        cur = d.get("current_a", d.get("current_filtered_a"))
        if cur is None or len(cur) < 10:
            continue
        v = float(np.std(cur))
        if v > best_var:
            best, best_var = (mid, d), v
    return best


def _pick_actuator(D):
    """Wo actuator_outputs instance jisme 4 motors sach me chal rahe hon."""
    best, best_var = None, -1
    for (name, mid), d in D.items():
        if name != "actuator_outputs":
            continue
        try:
            outs = np.stack([d[f"output[{i}]"] for i in range(4)])
        except KeyError:
            continue
        v = float(np.mean(np.std(outs, axis=1)))
        if v > best_var:
            best, best_var = (mid, d, outs), v
    return best


def parse_ulog(path, min_flight_s=90):
    """ULog -> (X (T,19) float32, info) ya (None, reason) agar flight valid nahi."""
    from pyulog import ULog
    try:
        u = ULog(str(path))
    except Exception as e:
        return None, f"parse error: {type(e).__name__}"

    D = {(d.name, d.multi_id): d.data for d in u.data_list}

    need = [("sensor_combined", 0), ("vehicle_attitude", 0), ("vehicle_local_position", 0)]
    for k in need:
        if k not in D:
            return None, f"missing {k[0]}"

    act = _pick_actuator(D)
    bat = _pick_battery(D)
    if act is None:
        return None, "no actuator_outputs"
    if bat is None:
        return None, "no battery_status"
    _, adat, outs = act
    _, bdat = bat

    # ---------------- ARMED window dhoondho ----------------
    # Motors chal rahe hon = PWM apne minimum se upar. Sirf usi hisse ko
    # flight maano. (Pehla log jo download kiya wo poora 900 pe atka tha.)
    lo = np.percentile(outs, 1)
    active = np.all(outs > lo + 40, axis=0)
    if active.sum() < 10:
        return None, "kabhi arm nahi hua (PWM flat)"
    at = adat["timestamp"][active]
    t0, t1 = at[0], at[-1]
    if (t1 - t0) / 1e6 < min_flight_s:
        return None, f"flight bahut chhoti ({(t1-t0)/1e6:.0f}s)"

    # ---------------- 50 Hz grid ----------------
    n = int((t1 - t0) / 1e6 * FS)
    tg = t0 + (np.arange(n) / FS * 1e6)

    sc = D[("sensor_combined", 0)]
    at_ = D[("vehicle_attitude", 0)]
    lp = D[("vehicle_local_position", 0)]

    roll, pitch = _quat_to_rp(*[at_[f"q[{i}]"] for i in range(4)])

    ch = {}
    ch["roll"] = _resample(at_["timestamp"], roll, tg)
    ch["pitch"] = _resample(at_["timestamp"], pitch, tg)
    for i, a in enumerate("xyz"):
        ch[f"gyro_{a}"] = _resample(sc["timestamp"], sc[f"gyro_rad[{i}]"], tg)
        ch[f"accel_{a}"] = _resample(sc["timestamp"], sc[f"accelerometer_m_s2[{i}]"], tg)
    # NED -> up
    ch["alt"] = -_resample(lp["timestamp"], lp["z"], tg)
    ch["vz"] = -_resample(lp["timestamp"], lp["vz"], tg)

    # PWM ko 0..1 me laao. Log ki apni range use karte hain (ESC calibration
    # har vehicle pe alag hoti hai; asli deployment me bhi yahi pata hoga).
    pmin, pmax = float(np.percentile(outs, 0.5)), float(np.percentile(outs, 99.5))
    if pmax - pmin < 50:
        return None, "PWM range bahut chhoti (arm nahi hua?)"
    for i in range(4):
        v = _resample(adat["timestamp"], outs[i], tg)
        ch[f"pwm_{i+1}"] = np.clip((v - pmin) / (pmax - pmin), 0, 1)

    volt = bdat.get("voltage_filtered_v", bdat.get("voltage_v"))
    cur = bdat.get("current_filtered_a", bdat.get("current_a"))
    ch["voltage"] = _resample(bdat["timestamp"], volt, tg)
    ch["current"] = _resample(bdat["timestamp"], cur, tg)

    es = D.get(("estimator_status", 0))
    if es is not None and "vibe[0]" in es:
        for i, a in enumerate("xyz"):
            ch[f"vibe_{a}"] = _resample(es["timestamp"], es[f"vibe[{i}]"], tg)
    else:
        vs = D.get(("vehicle_imu_status", 0))
        if vs is not None and "accel_vibration_metric" in vs:
            v = _resample(vs["timestamp"], vs["accel_vibration_metric"], tg)
            for a in "xyz":
                ch[f"vibe_{a}"] = v
        else:
            return None, "koi vibration channel nahi"

    X = np.stack([ch[c] for c in CHANNELS], axis=-1).astype(np.float32)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ---------------- PHYSICAL PLAUSIBILITY ----------------
    # Asli logs me kabhi kabhi bilkul bakwaas values hoti hain --- ek log me
    # accel_x = 2.2e20 tha. `isfinite` check use paas kar deta hai kyunki wo
    # value finite HAI, bas bemaani hai. Aur ek aisa log poore dataset ko
    # zeher kar deta hai (covariance, normalization, sab).
    #
    # Isliye har channel ki physical had lagao. Kabhi kabhaar ka spike clip
    # kar do; agar bahut saara data had ke bahar hai to poora log reject.
    bounds = {
        "roll": (-3.2, 3.2), "pitch": (-3.2, 3.2),
        "gyro_x": (-35, 35), "gyro_y": (-35, 35), "gyro_z": (-35, 35),
        "accel_x": (-100, 100), "accel_y": (-100, 100), "accel_z": (-100, 100),
        "alt": (-100, 1000), "vz": (-50, 50),
        "voltage": (5, 120), "current": (-5, 400),
        "vibe_x": (0, 500), "vibe_y": (0, 500), "vibe_z": (0, 500),
    }
    n_bad = 0
    for c, (lo_b, hi_b) in bounds.items():
        j = CHANNELS.index(c)
        bad = (X[:, j] < lo_b) | (X[:, j] > hi_b)
        n_bad += int(bad.sum())
        X[:, j] = np.clip(X[:, j], lo_b, hi_b)
    frac_bad = n_bad / (X.shape[0] * len(bounds))
    if frac_bad > 0.005:
        return None, f"sensor data bakwaas ({100*frac_bad:.1f}% values had ke bahar)"

    # ---------------- ye SACH ME udi thi? ----------------
    alt = X[:, CHANNELS.index("alt")]
    curr = X[:, CHANNELS.index("current")]
    pw = X[:, [CHANNELS.index(f"pwm_{i}") for i in range(1, 5)]]
    if alt.max() - alt.min() < 3.0:
        return None, f"altitude mushkil se badli ({alt.max()-alt.min():.1f} m)"
    if float(np.median(curr)) < 1.5:
        return None, f"current bahut kam ({np.median(curr):.2f} A) --- udi nahi"
    if float(np.mean(np.std(pw, axis=0))) < 0.01:
        return None, "PWM hila hi nahi"

    info = {
        "uid": path.stem, "n": int(n), "dur_s": round(n / FS, 1),
        "alt_range": round(float(alt.max() - alt.min()), 1),
        "current_med": round(float(np.median(curr)), 2),
        "voltage_med": round(float(np.median(X[:, CHANNELS.index("voltage")])), 2),
        "pwm_med": round(float(np.median(pw)), 3),
        "hw": u.initial_parameters.get("SYS_AUTOSTART", -1),
    }
    return X, info
