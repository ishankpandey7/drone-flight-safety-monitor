"""
RflyMAD (asli quadcopter, labelled fault onset) ko hamare 19-channel schema me laao.
====================================================================================

YE DATASET KYUN CHAHIYE THA
---------------------------
Ab tak project ka sabse bada khula sawaal ye tha: **chaar motors ka aapsi
rishta** --- is project ka quad-specific hissa --- sirf MERE simulator me test
hua hai. Asli data pe do imtihaan hue, par dono adhoore the:

  * logs.px4.io  : asli quadcopter, par ground truth NAHI. Wahan sirf ye naapa
                   ki healthy flight pe detector chup rehta hai ya nahi.
  * CMU ALFA     : asli fault, labelled onset --- par FIXED-WING. Chaar motors
                   ka rishta wahan hota hi nahi.

RflyMAD dono kami ek saath bharta hai: **asli quadcopter + labelled fault onset**.
497 real flight cases, teen asli drones (Droneyee X200 / X450 / X680), aur usme
`motor` aur `propeller` faults alag-alag types ke roop me hain.

FAULT KA LABEL KAHAN HAI
------------------------
RflyMAD PX4 firmware me ek custom ULog topic jodta hai: `rfly_ctrl_lxl`.
Uske `id` field me fault ka code hota hai:

    id == 1500                      ->  koi fault nahi (asli flight ka default)
    id == 0                         ->  koi fault nahi (simulation ka default)
    123450..123459  ya  123540..123549  ->  FAULT ACTIVE

(Ye ranges unke apne toolkit `fileprocess.py::RFLY_fault_id_check` se aayi hain,
meri guess nahi hain.)

Toh **fault onset = pehla sample jahan `id` fault range me ghusta hai.** Yahi wo
cheez hai jo logs.px4.io pe nahi thi.

DHYAN DENE WALI BAAT --- METRIC BADAL JAATA HAI
-----------------------------------------------
Mere simulator me faults DHEERE badhte hain aur crash tak jaate hain, isliye
wahan "lead time" (crash se kitne second pehle alarm) naapa. RflyMAD me fault
INJECT hota hai (achanak), aur safety pilot drone ko recover kar leta hai ---
flight crash tak jaati hi nahi. Isliye yahan sirf **detection delay** naap sakte
hain: fault lagne ke kitne second baad alarm baja. Wahi ALFA me kiya tha (5.3s).

Ye limitation data ki hai, method ki nahi --- par isko chhupana beimaani hogi.

CHALANA
-------
    python src/rfly_data.py --selftest   # bina dataset ke logic check karo
    python src/rfly_data.py              # downloaded dataset ka summary
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

import px4_data as P
from px4_data import CHANNELS, FS

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "rfly"

# ---------------------------------------------------------------- constants

# Fault id ki ranges --- RflyMAD ke apne toolkit se (fileprocess.py).
FAULT_ID_RANGES = ((123450, 123459), (123540, 123549))
NO_FAULT_IDS = (0, 1500)          # 1500 = real flight, 0 = simulation

# Folder ke naam --- unke README ki structure se.
FAULT_TYPES = (
    "motor", "propeller", "low_voltage", "wind_affect", "load_lose",
    "accelerometer", "gyroscope", "magnetometer", "barometer", "GPS", "no_fault",
)
FLIGHT_STATUS = ("hover", "waypoint", "velocity", "circling", "acce", "dece")

# Hamare liye kaam ke faults: yahi chaar-motor wale rishte ko hilate hain.
QUAD_FAULTS = ("motor", "propeller")


def is_fault_id(v):
    """Kya ye fault id 'fault active' batata hai? (scalar ya array, dono chalte hain)"""
    a = np.asarray(v)
    out = np.zeros(a.shape, dtype=bool)
    for lo, hi in FAULT_ID_RANGES:
        out |= (a >= lo) & (a <= hi)
    return out if out.ndim else bool(out)


# ---------------------------------------------------------------- discovery

def _tag_from_parts(parts, known):
    """Path ke hisso me se pehla jaana-pehchana naam dhoondo (case-insensitive)."""
    low = {k.lower(): k for k in known}
    for p in parts:
        if p.lower() in low:
            return low[p.lower()]
    return None


def find_cases(root=RAW):
    """Har flight case dhoondo jisme ek .ulg ho.

    Directory structure unke README me aisi hai:

        Real/<flight_status>/<fault_type>/<case>/<px4_dir>/xxx.ulg
                                                <ros_dir>/xxx.bag
                                                TestInfo.csv

    ...par unke apne sample me ek level kam tha (`Real/hover/12_1/...`). Isliye
    yahan depth pe bharosa NAHI karte --- har .ulg dhoondte hain aur path ke
    hisso me se jaane-pehchane naam utha lete hain. Ye thoda dheela hai, par
    dataset ka layout badla to bhi chalta rahega.
    """
    root = Path(root)
    if not root.exists():
        return []

    cases = []
    for ulg in sorted(root.rglob("*.ulg")):
        px4_dir = ulg.parent
        # TestInfo.csv case folder me hota hai --- ya to .ulg ke parent me,
        # ya uske ek upar (jab .ulg apne alag PX4_path folder me ho).
        case_dir = px4_dir
        for cand in (px4_dir, px4_dir.parent):
            if (cand / "TestInfo.csv").exists():
                case_dir = cand
                break
        else:
            case_dir = px4_dir.parent if px4_dir.parent != root else px4_dir

        try:
            parts = ulg.relative_to(root).parts
        except ValueError:
            parts = ulg.parts

        cases.append({
            "ulg": ulg,
            "case_dir": case_dir,
            "case_id": case_dir.name,
            "fault_type": _tag_from_parts(parts, FAULT_TYPES),
            "flight_status": _tag_from_parts(parts, FLIGHT_STATUS),
            "is_real": any(p.lower() == "real" for p in parts),
        })
    return cases


def read_testinfo(case_dir):
    """TestInfo.csv ko dict me padho.

    NOTE: is file ka exact format kahin documented nahi hai (unke toolkit me
    bhi ise sirf copy kiya jaata hai, padha nahi). Isliye yahan do aam shakal
    handle ki hain --- header+row, aur do-column key/value --- aur kuch bhi
    galat ho to khaali dict lauta dete hain. Ye metadata sirf extra hai;
    fault ka asli label ULog se aata hai, is file se nahi.

    Do-column file DONO ho sakti hai, isliye faisla row COUNT se hota hai:
    theek do rows = header + ek flight ki values (docs kehte hain ye "a concise
    summary information of a SINGLE flight case" hai); do se zyada rows jinme
    har row me theek 2 cell hon = key/value list.
    """
    f = Path(case_dir) / "TestInfo.csv"
    if not f.exists():
        return {}
    try:
        rows = [r for r in csv.reader(f.open(newline="", encoding="utf-8-sig")) if r]
    except Exception:
        return {}
    if len(rows) < 2:
        return {}

    header_row = len(rows[0]) == len(rows[1]) and len(rows[0]) >= 2
    if len(rows) == 2 and header_row:
        return {k.strip(): v.strip() for k, v in zip(rows[0], rows[1])}
    if all(len(r) == 2 for r in rows):
        return {r[0].strip(): r[1].strip() for r in rows}
    if header_row:
        return {k.strip(): v.strip() for k, v in zip(rows[0], rows[1])}
    return {}


# ---------------------------------------------------------------- fault label

def fault_from_ulog(u):
    """ULog object -> fault ki jaankari, ya None agar koi fault mila hi nahi.

    Lautata hai: {onset_us, id, n_samples, n_fault, controls}
    """
    d = None
    for ds in u.data_list:
        if ds.name == "rfly_ctrl_lxl":
            d = ds.data
            break
    if d is None or "id" not in d or "timestamp" not in d:
        return None

    ids = np.asarray(d["id"])
    ts = np.asarray(d["timestamp"])
    if len(ids) == 0:
        return None

    mask = is_fault_id(ids)
    if not mask.any():
        return {"onset_us": None, "id": None, "n_samples": int(len(ids)),
                "n_fault": 0, "controls": None}

    i = int(np.argmax(mask))          # pehla True
    ctrl = []
    for k in range(16):
        key = f"controls[{k}]"
        if key not in d:
            break
        ctrl.append(float(np.asarray(d[key])[i]))

    return {
        "onset_us": int(ts[i]),
        "id": int(ids[i]),
        "n_samples": int(len(ids)),
        "n_fault": int(mask.sum()),
        "controls": ctrl or None,
    }


# ---------------------------------------------------------------- load

def load_case(case, min_flight_s=20.0, min_alt_range_m=0.5, min_current_a=1.0):
    """Ek flight case -> (X (T,19) float32, meta) ya (None, wajah).

    Channel nikalne ka kaam `px4_data.parse_ulog` hi karta hai --- wahi code
    logs.px4.io ke 37 asli flights pe chal chuka hai, isliye dobara likhne ka
    koi matlab nahi. Bas thresholds dheele kiye hain: RflyMAD ki flights chhoti
    indoor hover tests hain, unme 90 second / 3 meter altitude wali shart har
    achhi flight ko reject kar degi.

    Yahan ka ASLI naya kaam ek hi hai: fault ka absolute ULog timestamp ko X ke
    apne time grid me badalna. parse_ulog apni marzi ka armed-window chunta hai
    (t0_us), toh fault ka index = (onset_us - t0_us) * FS.
    """
    from pyulog import ULog

    ulg = Path(case["ulg"])
    X, info = P.parse_ulog(ulg, min_flight_s=min_flight_s,
                           min_alt_range_m=min_alt_range_m,
                           min_current_a=min_current_a)
    if X is None:
        return None, info                      # info = reject karne ki wajah

    try:
        u = ULog(str(ulg))
    except Exception as e:
        return None, f"rfly_ctrl padhne me dikkat: {type(e).__name__}"

    fault = fault_from_ulog(u)
    if fault is None:
        return None, "rfly_ctrl_lxl topic hi nahi (RflyMAD log nahi?)"

    meta = {
        "case_id": case.get("case_id"),
        "fault_type": case.get("fault_type"),
        "flight_status": case.get("flight_status"),
        "n": info["n"],
        "dur_s": info["dur_s"],
        "fault_id": fault["id"],
        "fault_controls": fault["controls"],
        "rfly_samples": fault["n_samples"],
        "rfly_fault_samples": fault["n_fault"],
        "testinfo": read_testinfo(case["case_dir"]),
    }

    # ---- fault onset ko X ke grid pe laao ----
    if fault["onset_us"] is None:
        # no_fault case --- ye galat nahi hai, bas label nahi hai.
        meta.update(onset_s=None, onset_idx=None, onset_where="no_fault")
        return X, meta

    onset_s = (fault["onset_us"] - info["t0_us"]) / 1e6
    idx = int(round(onset_s * FS))

    if idx < 0:
        # Fault tab laga jab drone abhi arm bhi nahi hua tha. Aise case me
        # "detection delay" ka koi matlab nahi --- poori flight already faulty hai.
        meta.update(onset_s=round(onset_s, 2), onset_idx=None, onset_where="before_arm")
    elif idx >= info["n"]:
        meta.update(onset_s=round(onset_s, 2), onset_idx=None, onset_where="after_end")
    else:
        meta.update(onset_s=round(onset_s, 2), onset_idx=idx, onset_where="ok")

    return X, meta


# ---------------------------------------------------------------- selftest

def _selftest():
    """Bina dataset ke jitna check ho sakta hai, utna karo.

    Dataset 5.57 GB ka hai aur abhi download nahi hua. Par do sabse nazuk hisse
    --- fault id ka matlab, aur folder tree padhna --- dono pure logic hain,
    unhe abhi jaancha ja sakta hai. ULog se channel nikalne wala hissa pehle se
    37 asli PX4 flights pe chal chuka hai.
    """
    import shutil
    import tempfile

    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        ok = ok and bool(cond)

    print("\n  --- 1. fault id ka matlab ---")
    check("1500 = koi fault nahi",            not is_fault_id(1500))
    check("0 = koi fault nahi",               not is_fault_id(0))
    check("123450 = fault (range ka pehla)",  is_fault_id(123450))
    check("123459 = fault (range ka aakhri)", is_fault_id(123459))
    check("123540 = fault (doosri range)",    is_fault_id(123540))
    check("123549 = fault (doosri range)",    is_fault_id(123549))
    check("123460 = fault NAHI (beech ka gap)", not is_fault_id(123460))
    check("123539 = fault NAHI (beech ka gap)", not is_fault_id(123539))

    arr = np.array([1500, 1500, 1500, 123451, 123451, 123451])
    m = is_fault_id(arr)
    check("array pe bhi chalta hai", m.tolist() == [0, 0, 0, 1, 1, 1])
    check("onset sahi index pe", int(np.argmax(m)) == 3)

    print("\n  --- 2. folder tree padhna ---")
    tmp = Path(tempfile.mkdtemp(prefix="rflytest_"))
    try:
        # (a) poori structure: Real/<status>/<fault>/<case>/<px4dir>/x.ulg
        deep = tmp / "Real" / "hover" / "motor" / "12_1" / "log_6_2023-5-17"
        deep.mkdir(parents=True)
        (deep / "log_6_2023-5-17.ulg").write_bytes(b"x")
        (deep.parent / "TestInfo.csv").write_text("case,fault\n12_1,motor\n", encoding="utf-8")

        # (b) unke sample wali chhoti structure (fault_type level nahi)
        flat = tmp / "Real" / "waypoint" / "56_1" / "log_9_2023-5-18"
        flat.mkdir(parents=True)
        (flat / "log_9_2023-5-18.ulg").write_bytes(b"x")

        cases = find_cases(tmp)
        check("dono cases mile", len(cases) == 2)

        by_id = {c["case_id"]: c for c in cases}
        c1 = by_id.get("12_1")
        check("gehri structure: case_id sahi", c1 is not None)
        if c1:
            check("gehri structure: fault_type=motor", c1["fault_type"] == "motor")
            check("gehri structure: status=hover", c1["flight_status"] == "hover")
            check("gehri structure: real flag", c1["is_real"] is True)
            check("TestInfo.csv (header+row) padhi gayi",
                  read_testinfo(c1["case_dir"]).get("fault") == "motor")

        # TestInfo.csv doosri shakal me bhi aa sakti hai --- key/value list.
        kv = tmp / "Real" / "hover" / "motor" / "kv_case"
        kv.mkdir(parents=True)
        (kv / "TestInfo.csv").write_text(
            "fault,motor\nfault_param,0.3\nstatus,hover\n", encoding="utf-8")
        got = read_testinfo(kv)
        check("TestInfo.csv (key/value) padhi gayi",
              got.get("fault") == "motor" and got.get("fault_param") == "0.3")
        check("TestInfo.csv na ho to khaali dict", read_testinfo(tmp) == {})

        c2 = by_id.get("56_1")
        check("chhoti structure: case mila", c2 is not None)
        if c2:
            check("chhoti structure: status=waypoint", c2["flight_status"] == "waypoint")
            check("chhoti structure: fault_type None (folder hai hi nahi)",
                  c2["fault_type"] is None)

        check("khaali folder pe crash nahi", find_cases(tmp / "nahi_hai") == [])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("=" * 74)
    if ok:
        print("  Sab selftest pass. Ab dataset download karke asli imtihaan baaki hai.")
    else:
        print("  !! Kuch selftest fail hue --- upar dekho.")
    print("=" * 74)
    return 0 if ok else 1


# ---------------------------------------------------------------- main

def main():
    if "--selftest" in sys.argv:
        return _selftest()

    cases = find_cases()
    if not cases:
        print()
        print("=" * 74)
        print(f"  {RAW} me koi .ulg nahi mili.")
        print("=" * 74)
        print()
        print("  RflyMAD ka sirf REAL-FLIGHT hissa chahiye (poora 114 GB nahi):")
        print("    Real-Motor    5.57 GB   <- ye sabse zaroori hai")
        print("    Real-Sensors  4.01 GB")
        print("    Real-No Fault  973 MB")
        print()
        print("    https://rfly-openha.github.io/documents/4_resources/dataset.html")
        print()
        print(f"  Unzip karke {RAW} me daalo, phir ye dobara chalao.")
        print("  Tab tak logic check karne ke liye:  python src/rfly_data.py --selftest")
        return 1

    real = [c for c in cases if c["is_real"]]
    print()
    print("=" * 74)
    print(f"  {len(cases)} flight cases mile ({len(real)} real flight)")
    print("=" * 74)
    print()

    from collections import Counter
    print("  fault type ke hisaab se:")
    for k, v in Counter(c["fault_type"] or "?" for c in cases).most_common():
        star = "  <- quad-specific" if k in QUAD_FAULTS else ""
        print(f"    {v:4d}  {k}{star}")
    print()
    print("  flight status ke hisaab se:")
    for k, v in Counter(c["flight_status"] or "?" for c in cases).most_common():
        print(f"    {v:4d}  {k}")

    # Pehla motor-fault case load karke dikhao ki label sach me nikal raha hai
    quad = [c for c in cases if c["fault_type"] in QUAD_FAULTS]
    if quad:
        print()
        print("-" * 74)
        print(f"  Pehla quad-fault case load karke dekhte hain: {quad[0]['case_id']}")
        print("-" * 74)
        X, meta = load_case(quad[0])
        if X is None:
            print(f"  REJECT: {meta}")
        else:
            print(f"    shape        : {X.shape}  ({meta['dur_s']}s @ {FS:.0f} Hz)")
            print(f"    fault type   : {meta['fault_type']}")
            print(f"    fault id     : {meta['fault_id']}")
            print(f"    onset        : {meta['onset_s']}s  (index {meta['onset_idx']}, "
                  f"{meta['onset_where']})")
            print(f"    rfly samples : {meta['rfly_fault_samples']}/{meta['rfly_samples']} "
                  f"fault-active")
    return 0


if __name__ == "__main__":
    sys.exit(main())
