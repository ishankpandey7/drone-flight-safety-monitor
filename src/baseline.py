"""
STEP 3 --- Imaandar baseline (bina kisi neural network ke).
===========================================================

Ye step log chhod dete hain, aur yahi sabse badi galti hai.

Agar main apne LSTM ko sirf us bewakoof "PWM spread" detector se compare
karunga (jo 15% recall deta hai), to LSTM zaroor jeetega --- par wo jeet
bekaar hogi. Main sirf itna prove karunga ki ek accha model ek ganday model
se behtar hai. Wo to pehle se pata tha.

Asli sawaal ye hai:

    Kya is problem ko solve karne ke liye DEEP LEARNING chahiye hi,
    ya purani statistics kaafi hai?

Isliye yahan hum ek serious baseline banate hain:

    1. Har flight ko 8-second ke windows me kaato
    2. Har window me, har channel ka mean aur std nikalo  (19 x 2 = 38 numbers)
    3. Sirf HEALTHY training flights se seekho ki ye 38 numbers normally
       kis "cloud" me rehte hain --- mean vector + covariance matrix
    4. Naye window ka score = us cloud se Mahalanobis distance

Mahalanobis distance normal Euclidean distance se behtar kyun hai:
    Euclidean poochta hai "center se kitni door?"
    Mahalanobis poochta hai "center se kitni door, us DISHA me kitna
    variation normal hai uske hisaab se?"

    Yani agar current hamesha 5 se 15 A ke beech rehta hai, to 16 A koi
    badi baat nahi. Par agar voltage hamesha 15.8 se 16.1 ke beech rehta
    hai, to 15.2 alarm ki baat hai --- chahe absolute farq zyada ho.

    Aur sabse zaroori: covariance channels ke RISHTE bhi pakadta hai.
    "current bada hai" normal ho sakta hai; "current bada hai LEKIN
    altitude badh nahi rahi" --- ye anomaly hai.

Yahi cheez hai jo naive detector nahi kar paya. Dekhte hain kitna sudhaar
aata hai.
"""

import json
from pathlib import Path

import numpy as np

import common
from common import FS, WIN_S, STRIDE_S, K_CONSEC

ROOT = Path(__file__).resolve().parent.parent
WIN = int(WIN_S * FS)
STRIDE = int(STRIDE_S * FS)


# ------------------------------------------------------------------ features

def windowize(X):
    """(N, T, C)  ->  (N, W, 2C)   har window ka mean aur std."""
    N, T, C = X.shape
    starts = range(0, T - WIN + 1, STRIDE)
    out = np.empty((N, len(starts), 2 * C), dtype=np.float64)
    for j, s in enumerate(starts):
        w = X[:, s:s + WIN, :]
        out[:, j, :C] = w.mean(axis=1)
        out[:, j, C:] = w.std(axis=1)
    return out




# ------------------------------------------------------------------ model

class MahalanobisDetector:
    def fit(self, F):
        """F: (n_windows, n_features) --- sirf HEALTHY data."""
        self.mu = F.mean(axis=0)
        self.sd = F.std(axis=0) + 1e-8
        Z = (F - self.mu) / self.sd
        cov = np.cov(Z, rowvar=False)
        # Ridge: covariance matrix ko invert karne layak banata hai.
        # Bina iske, agar do features lagbhag ek jaise hain to matrix
        # singular ho jaata hai aur inverse me infinity aa jaati hai.
        cov += np.eye(cov.shape[0]) * 1e-3
        self.inv = np.linalg.inv(cov)
        return self

    def score(self, F):
        Z = (F - self.mu) / self.sd
        # har row ka  z^T * inv * z   --- loop se bachne ke liye einsum
        return np.sqrt(np.einsum("ij,jk,ik->i", Z, self.inv, Z))


# ------------------------------------------------------------------ eval



def main():
    d = np.load(ROOT / "data" / "processed" / "flights.npz", allow_pickle=True)
    meta = json.load(open(ROOT / "data" / "processed" / "meta.json"))
    fmeta = [m for m in meta if m["split"] == "test_faulty"]

    Xtr, Xva = d["train"], d["val"]
    Xth, Xtf = d["test_healthy"], d["test_faulty"]
    T = Xtr.shape[1]
    wt = common.window_times(T)

    print()
    print("Windows bana rahe hain...")
    Ftr, Fva = windowize(Xtr), windowize(Xva)
    Fth, Ftf = windowize(Xth), windowize(Xtf)
    print(f"  train : {Ftr.shape[0]} flights x {Ftr.shape[1]} windows x {Ftr.shape[2]} features")

    # ---- sirf healthy train data pe fit ----
    det = MahalanobisDetector().fit(Ftr.reshape(-1, Ftr.shape[-1]))

    def score_all(F):
        n, w, f = F.shape
        return det.score(F.reshape(-1, f)).reshape(n, w)

    Sva, Sth, Stf = score_all(Fva), score_all(Fth), score_all(Ftf)

    # ---- evaluate (common.py ka wahi code jo LSTM ke liye bhi chalega) ----
    res = common.report("BASELINE  --  Mahalanobis distance, 38 windowed features",
                        Sva, Sth, Stf, fmeta, wt)

    print("  Naive detector (sirf PWM spread) ki recall thi : 15%")
    print(f"  Ye baseline                                    : {100 * res['recall']:.0f}%")
    print()
    print("  >> Ab LSTM ko ISSE jeetna hoga, us 15% wale se nahi.")
    print()

    np.savez(ROOT / "outputs" / "baseline_scores.npz",
             val=Sva, test_healthy=Sth, test_faulty=Stf,
             window_times=wt, threshold=res["threshold"])
    with open(ROOT / "outputs" / "baseline_result.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
