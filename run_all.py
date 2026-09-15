"""
Poora project ek command se chalao:   python run_all.py

Har step apne se pehle wale step ka output istemaal karta hai, isliye order
maayne rakhta hai. Agar koi step fail ho to yahin ruk jao --- aage badhne se
sirf confusing errors milenge.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("simulator.py",       "Step 1  --  do demo flights (healthy + faulty)"),
    ("probe_crash.py",     "Step 1a --  har fault kitni der me maarta hai"),
    ("check_signature.py", "Step 1b --  fault ka nishaan data me hai ya nahi"),
    ("plot_signature.py",  "Step 1c --  wo nishaan ek tasveer me"),
    ("generate_dataset.py","Step 2  --  poora dataset (~12 min)"),
    ("check_dataset.py",   "Step 2b --  signal maujood hai? naive detector kitna kharab?"),
    ("baseline.py",        "Step 3  --  Mahalanobis baseline (bina ML)"),
    ("lstm_model.py",      "Step 4  --  LSTM autoencoder (~13 min)"),
    ("rescore.py",         "Step 4b --  aggregator ka farq"),
    ("forecaster.py",      "Step 5  --  LSTM forecaster (~7 min)"),
    ("hybrid.py",          "Step 5b --  forecaster errors -> Mahalanobis"),
    ("compare.py",         "Step 6  --  sabka imaandar comparison (ROC)"),
    ("flight_report.py",   "Step 7  --  har fault type ka flight report"),
    ("multiscale.py",      "Step 8  --  do timescale (8s + 2s), joint calibration"),
    ("make_dashboard.py",  "Step 9  --  interactive HTML replay"),
    ("update_readme.py",   "Step 10 --  README ka results table refresh"),
]


def main():
    only = sys.argv[1:] or None
    t_all = time.time()

    for script, desc in STEPS:
        if only and not any(o in script for o in only):
            continue
        print()
        print("=" * 78)
        print(f"  {desc}")
        print(f"  $ python src/{script}")
        print("=" * 78)
        t0 = time.time()
        r = subprocess.run([sys.executable, str(ROOT / "src" / script)], cwd=ROOT)
        if r.returncode != 0:
            print(f"\n  !! {script} fail ho gaya (exit {r.returncode}). Yahin ruk rahe hain.")
            return r.returncode
        print(f"  [{time.time() - t0:.0f}s]")

    print()
    print("=" * 78)
    print(f"  Sab ho gaya  ({time.time() - t_all:.0f}s)")
    print("  outputs/ folder me tasveerein aur results hain.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
