"""
Parameters size karne se pehle naapo: har fault type kitni der me maarta hai?

Bina is naap ke main andaza laga kar ramp range chunta, aur ya to saare
faults 5 second me maar dete (detector ke paas mauka hi nahi), ya koi
crash hi nahi hota (lead time naapne ko kuch bacha hi nahi).
"""

import numpy as np

from simulator import simulate_flight, Fault

KINDS = ["motor_degradation", "prop_damage", "battery_sag", "imu_drift"]
RAMPS = [10, 20, 35, 60]
SEEDS = [11, 22, 33]
DUR = 260.0
FAULT_AT = 40.0

print()
print("=" * 88)
print(f"  Fault {FAULT_AT:.0f}s pe shuru. Crash tak kitne second mile? (3 seeds ka average)")
print("=" * 88)
print()
print(f"  {'fault type':<22}" + "".join(f"{f'ramp {r}s':>14}" for r in RAMPS))
print("  " + "-" * 84)

for kind in KINDS:
    row = []
    for ramp in RAMPS:
        surv, reasons = [], []
        for sd in SEEDS:
            _, m = simulate_flight(
                duration_s=DUR, seed=sd,
                fault=Fault(kind=kind, start_s=FAULT_AT, motor=sd % 4, ramp_s=float(ramp)))
            if m["crash_s"] > 0:
                surv.append(m["survived_s"])
                reasons.append(m["crash_reason"])
        if surv:
            row.append(f"{np.mean(surv):.0f}s")
        else:
            row.append("bacha")
    print(f"  {kind:<22}" + "".join(f"{v:>14}" for v in row))

print()
print("  'bacha' = is ramp pe drone gira hi nahi --- fault itna dheema tha ki")
print("  flight khatam ho gayi aur drone abhi bhi ud raha tha.")

# ---- healthy flights kabhi nahi girni chahiye ----
print()
crashes = 0
for sd in range(40):
    _, m = simulate_flight(duration_s=DUR, seed=5000 + sd, fault=None)
    if m["crash_s"] > 0:
        crashes += 1
        print(f"  !! healthy flight seed {5000 + sd} {m['crash_s']:.0f}s pe giri "
              f"({m['crash_reason']}) --- ye BUG hai")
print(f"  40 healthy flights me se {crashes} giri "
      f"{'(theek hai)' if crashes == 0 else '<-- THEEK KARNA PADEGA'}")
print()
