"""
Candidates download karo, parse karo, aur SIRF asli flights rakho.

Har log teen imtihaan paas karna chahiye:
  1. asli hardware ho (SITL nahi)       <- index me hi filter ho gaya
  2. drone sach me arm hua ho           <- PWM apne minimum se upar
  3. drone sach me uda ho               <- altitude badli, current bahut,
                                           PWM hila

Teesra sabse zaroori hai. Bina uske dataset bench tests se bhar jaata hai,
detector un pe chup rehta hai, aur mujhe jhoothi tasalli mil jaati hai ki
"false alarm rate kam hai".
"""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

import px4_data as P

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "px4"
TARGET = 55          # itni valid flights chahiye
MAX_TRY = 130


def main():
    cands = json.load(open(OUT / "candidates.json"))
    print(f"  {len(cands)} candidates; {TARGET} valid flights chahiye")
    print()

    flights, metas, reasons = {}, [], Counter()
    tried = bytes_dl = 0

    for e in cands:
        if len(flights) >= TARGET or tried >= MAX_TRY:
            break
        tried += 1
        uid = e["uid"]
        try:
            path = P.download(uid)
        except Exception as ex:
            reasons[f"download: {type(ex).__name__}"] += 1
            continue
        if path is None:
            reasons["download: empty"] += 1
            continue
        bytes_dl += path.stat().st_size

        X, info = P.parse_ulog(path)
        if X is None:
            reasons[info] += 1
            path.unlink(missing_ok=True)      # bekaar log rakhne ka fayda nahi
            print(f"  [{tried:3d}] {uid[:8]}  REJECT  {info}")
            continue

        info.update({k: e[k] for k in ("hw", "sw", "airframe")})
        flights[uid] = X
        metas.append(info)
        print(f"  [{tried:3d}] {uid[:8]}  OK  {info['dur_s']:6.1f}s  "
              f"alt_range={info['alt_range']:5.1f}m  I={info['current_med']:5.1f}A  "
              f"V={info['voltage_med']:5.1f}  ({len(flights)}/{TARGET})")
        time.sleep(0.4)

    print()
    print("=" * 74)
    print(f"  {tried} logs try kiye, {len(flights)} valid flights mili "
          f"({bytes_dl/1e6:.0f} MB download)")
    print("=" * 74)
    print()
    print("  Reject hone ki wajah:")
    for r, c in reasons.most_common():
        print(f"    {c:3d}  {r}")

    if not flights:
        raise SystemExit("\n  Ek bhi valid flight nahi mili.")

    np.savez_compressed(OUT / "real_flights.npz",
                        **{f"f{i}": X for i, X in enumerate(flights.values())},
                        channels=np.array(P.CHANNELS))
    json.dump(metas, open(OUT / "real_meta.json", "w"), indent=1)

    durs = np.array([m["dur_s"] for m in metas])
    print()
    print(f"  saved -> data/px4/real_flights.npz  ({len(flights)} flights)")
    print(f"  flight lambai : {durs.min():.0f}-{durs.max():.0f}s "
          f"(median {np.median(durs):.0f}s), kul {durs.sum()/60:.0f} minute")
    print(f"  {len(set(m['hw'] for m in metas))} alag flight controllers, "
          f"{len(set(m['airframe'] for m in metas))} alag airframes")


if __name__ == "__main__":
    main()
