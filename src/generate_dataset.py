"""
Poora dataset banao.
====================

Sabse zaroori design decision yahan hai:

    Model ko hum SIRF healthy flights dikhayenge. Ek bhi fault nahi.

Ye ulta lagta hai, par soch ke dekho:

  * Agar hum "fault classifier" banate (healthy vs faulty), toh model sirf
    WOHI 4 faults pakad paata jo humne sikhaye. Paanchwa naya fault --- jo
    asli duniya me zaroor aayega --- wo chup chaap miss ho jaata.

  * Anomaly detection me hum ulta karte hain: model sirf "normal" ka matlab
    seekhta hai. Phir JO BHI normal se hatt ke hai, wo pakda jaata hai ---
    chahe humne wo fault kabhi dekha ho ya nahi.

Isi wajah se faulty flights sirf TEST me jaati hain, train me nahi.

Split:
    train   : 100 healthy   <- model isi pe seekhega
    val     :  25 healthy   <- threshold tay karne ke liye
    test    :  25 healthy   +  60 faulty (4 fault type x 15)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# Paths ko SCRIPT ki jagah se nikalo, na ki "tum kahan se chala rahe ho" se.
# Warna `cd src && python generate_dataset.py` aur `python src/generate_dataset.py`
# do alag jagah file dhoondhte hain --- aur ek chup chaap fail ho jaata hai.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

from simulator import simulate_flight, Fault

# ---------------------------------------------------------------- config

DURATION   = 200.0
FAULT_KINDS = ["motor_degradation", "prop_damage", "battery_sag", "imu_drift"]

N_TRAIN         = 100
N_VAL           = 100
N_TEST_HEALTHY  = 25
N_TEST_PER_FAULT = 15

# Model ko ye channels dikhenge. Kuch jaan boojh kar CHHODE gaye hain:
#   t              -> sirf ek counter hai, isse model "flight kab khatam hoti
#                     hai" ratt lega, physics nahi seekhega
#   fault_active   -> ye to LABEL hai! Isko input me daalna cheating hai
#   yaw            -> absolute heading bekaar hai (drone kis disha me hai isse
#                     sehat ka koi lena dena nahi). gyro_z se rate mil hi raha hai
#   pos_x, pos_y   -> bina limit ke bhatakte hain, koi physics nahi batate
FEATURE_CHANNELS = [
    "roll", "pitch",
    "gyro_x", "gyro_y", "gyro_z",
    "accel_x", "accel_y", "accel_z",
    "alt", "vz",
    "pwm_1", "pwm_2", "pwm_3", "pwm_4",
    "voltage", "current",
    "vibe_x", "vibe_y", "vibe_z",
]


def make_fault(rng):
    """Ek random fault banao.

    ramp_s ab sabse zaroori knob hai --- wahi tay karta hai ki drone ko
    marne me kitna waqt lagega. probe_crash.py se naapa gaya:

        ramp 10s  ->  drone ~5-29s me gir jaata hai   (achanak wali failure)
        ramp 60s  ->  drone ~26-56s me girta hai      (dheeme marta motor)

    Dono chahiye. Tez wali batati hai ki detector kitna jaldi bol sakta hai;
    dheemi wali asli sawaal poochti hai --- kya hum ise BAHUT pehle pakad
    sakte hain?
    """
    return Fault(
        kind=str(rng.choice(FAULT_KINDS)),
        # 25-75s ke beech shuru ho: pehle kuch normal flight dikhe, aur
        # crash ke liye bhi kaafi waqt bache
        start_s=float(rng.uniform(25, 75)),
        motor=int(rng.integers(0, 4)),
        ramp_s=float(rng.uniform(10, 60)),
    )


def build(split_name, n, faulty, seed0, rng):
    """n flights banao aur (array, metadata) return karo."""
    X, M = [], []
    t0 = time.time()
    for i in range(n):
        fault = make_fault(rng) if faulty else None
        log, meta = simulate_flight(duration_s=DURATION, fault=fault, seed=seed0 + i)

        X.append(np.stack([log[c] for c in FEATURE_CHANNELS], axis=-1))
        meta["split"] = split_name
        meta["flight_id"] = f"{split_name}_{i:04d}"
        # fault kab active tha, ye TEST ke liye ground truth hai
        meta["fault_active_frac"] = float(log["fault_active"].mean())
        M.append(meta)

        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"    {split_name:<14} {i + 1:3d}/{n}   ({time.time() - t0:.1f}s)")

    return np.stack(X).astype(np.float32), M


def main():
    rng = np.random.default_rng(42)
    all_X, all_M = {}, []

    print()
    print("Healthy flights (model inhi pe train hoga):")
    all_X["train"], m = build("train", N_TRAIN, False, 1000, rng); all_M += m
    all_X["val"],   m = build("val",   N_VAL,   False, 2000, rng); all_M += m

    print()
    print("Test set --- healthy:")
    Xth, m = build("test_healthy", N_TEST_HEALTHY, False, 3000, rng); all_M += m

    print()
    print("Test set --- faulty (ye model ne kabhi nahi dekhe):")
    Xtf_parts = []
    for k, kind in enumerate(FAULT_KINDS):
        Xp, M = [], []
        for i in range(N_TEST_PER_FAULT):
            f = make_fault(rng)
            f.kind = kind                      # is batch me fault type fix rakho
            log, meta = simulate_flight(duration_s=DURATION, fault=f, seed=4000 + k * 100 + i)
            Xp.append(np.stack([log[c] for c in FEATURE_CHANNELS], axis=-1))
            meta["split"] = "test_faulty"
            meta["flight_id"] = f"test_faulty_{kind}_{i:03d}"
            meta["fault_active_frac"] = float(log["fault_active"].mean())
            M.append(meta)
        Xtf_parts.append(np.stack(Xp).astype(np.float32))
        all_M += M
        print(f"    {kind:<20} {N_TEST_PER_FAULT} flights")

    all_X["test_healthy"] = Xth
    all_X["test_faulty"]  = np.concatenate(Xtf_parts)

    # ------------------------------------------------------------ save
    npz_path  = OUT / "flights.npz"
    meta_path = OUT / "meta.json"
    np.savez_compressed(
        npz_path,
        train=all_X["train"],
        val=all_X["val"],
        test_healthy=all_X["test_healthy"],
        test_faulty=all_X["test_faulty"],
        channels=np.array(FEATURE_CHANNELS),
    )
    with open(meta_path, "w") as f:
        json.dump(all_M, f, indent=1)

    print()
    print("=" * 70)
    for k, v in all_X.items():
        print(f"  {k:<16} shape = {v.shape}    "
              f"({v.shape[0]} flights x {v.shape[1]} steps x {v.shape[2]} channels)")
    print("=" * 70)
    print(f"  saved -> {npz_path}")
    print(f"  saved -> {meta_path}  ({len(all_M)} entries)")

    # ---- ek chhota sa sanity check: kya healthy aur faulty alag hain? ----
    pwm_idx = [FEATURE_CHANNELS.index(f"pwm_{i}") for i in range(1, 5)]

    def spread(X):
        p = X[:, :, pwm_idx]
        return np.median(p.max(-1) - p.min(-1), axis=1)

    # ---- crash summary: lead time naapne ke liye ye sabse zaroori hai ----
    fm = [m for m in all_M if m["split"] == "test_faulty"]
    hm = [m for m in all_M if m["split"] != "test_faulty"]
    hcrash = sum(1 for m in hm if m["crash_s"] > 0)
    print()
    print("  Crash summary:")
    print(f"    healthy flights jo giri : {hcrash}/{len(hm)}"
          f"  {'(theek hai)' if hcrash == 0 else '<-- BUG!'}")
    for kind in FAULT_KINDS:
        k = [m for m in fm if m["fault_kind"] == kind]
        crashed = [m for m in k if m["crash_s"] > 0]
        if crashed:
            surv = np.array([m["survived_s"] for m in crashed])
            print(f"    {kind:<20} {len(crashed)}/{len(k)} giri   "
                  f"fault se crash tak: {surv.min():.0f}-{surv.max():.0f}s "
                  f"(median {np.median(surv):.0f}s)")
        else:
            print(f"    {kind:<20} 0/{len(k)} giri")
    allc = [m for m in fm if m["crash_s"] > 0]
    print()
    print(f"    kul {len(allc)}/{len(fm)} faulty flights crash hui.")
    print(f"    >> Detector ke paas average {np.mean([m['survived_s'] for m in allc]):.0f} "
          f"second hai alarm bajane ka.")

    sh, sf = spread(all_X["test_healthy"]), spread(all_X["test_faulty"])
    print()
    print("  Sanity check --- motors ka median farq, har flight ka:")
    print(f"    healthy flights : {sh.mean():.4f}  (+- {sh.std():.4f})")
    print(f"    faulty  flights : {sf.mean():.4f}  (+- {sf.std():.4f})")
    overlap = np.mean(sf < np.percentile(sh, 95))
    print(f"    {100 * overlap:.0f}% faulty flights healthy ke 95th percentile ke ANDAR hain")
    print("    (jitna zyada overlap, utna hi mushkil problem --- aur utna hi asli)")


if __name__ == "__main__":
    main()
