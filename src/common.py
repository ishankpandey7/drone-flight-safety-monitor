"""
Dono detectors (baseline aur LSTM) ke liye EK HI evaluation code.

Ye alag file isliye hai taaki dono ko bilkul ek hi tarah se jaancha jaaye.
Agar main dono ke liye alag alag threshold logic likhta, to farq model ka
hai ya evaluation ka --- ye pata hi nahi chalta.
"""

import numpy as np

FS = 50.0
WIN_S = 8.0        # kitne second ki window dekh kar faisla
STRIDE_S = 2.0     # har kitne second me naya faisla
K_CONSEC = 3       # itni lagatar windows fail hon tab alarm bajega


def flight_alarm(scores, thr, k=K_CONSEC):
    """Ek flight ke window-scores se batao: alarm baja ya nahi, aur kab.

    Ek hi window ka spike alarm nahi banata --- k lagatar windows chahiye.
    Ye jhoothe alarm bahut kam kar deta hai (ek random spike se kaam nahi
    chalega, problem TIKI honi chahiye), aur asli fault kabhi ek window ka
    nahi hota, wo rehta hai.
    """
    over = scores > thr
    run = 0
    for i, o in enumerate(over):
        run = run + 1 if o else 0
        if run >= k:
            return True, i
    return False, -1


def pick_threshold(S_val, target_fa=0.05, k=K_CONSEC):
    """VAL (healthy) scores se threshold chuno.

    IMPORTANT: threshold hamesha VAL se aana chahiye, TEST se nahi.
    Test dekh kar threshold chunna apne hi exam ki answer key dekhne
    jaisa hai --- number accha aayega, matlab kuch nahi hoga.

    Sabse SENSITIVE threshold chuno jispe false alarm rate ab bhi
    target ke andar ho.
    """
    cand = np.unique(np.round(S_val.ravel(), 3))
    thr = float(cand.max()) * 2
    for c in np.sort(cand)[::-1]:
        fa = np.mean([flight_alarm(s, c, k)[0] for s in S_val])
        if fa <= target_fa:
            thr = float(c)
        else:
            break
    return thr


def window_times(T_raw):
    """Har window ka END time (seconds). Detection isi waqt ho sakta hai ---
    window poori hone se pehle faisla nahi kar sakte."""
    win, stride = int(WIN_S * FS), int(STRIDE_S * FS)
    return np.array([(s + win) / FS for s in range(0, T_raw - win + 1, stride)])


def alarm_times(S, wt, thr, fmeta=None, k=K_CONSEC):
    """Har flight ka alarm time (seconds), ya None agar alarm nahi baja.

    fmeta diya ho to CRASH KE BAAD KI SAARI WINDOWS HATA DI JAATI HAIN.
    Crash ke baad drone palat raha hai / zameen pe pada hai --- wo telemetry
    itni pagal hoti hai ki koi bhi detector use pakad lega. Us par credit
    lena apne aap ko dhoka dena hai, aur recall jhoothe taur pe 90%+ dikha
    dega.
    """
    out = []
    for i, s in enumerate(S):
        crash = fmeta[i].get("crash_s", -1.0) if fmeta is not None else -1.0
        keep = (wt <= crash) if crash > 0 else np.ones(len(wt), bool)
        if keep.sum() < k:
            out.append(None)
            continue
        alarm, widx = flight_alarm(s[keep], thr, k)
        out.append(float(wt[keep][widx]) if alarm else None)
    return out


def report(name, S_val, S_healthy, S_faulty, fmeta, wt, target_fa=0.05):
    """Ek detector ke scores se report banao (threshold val se chuna jaata hai)."""
    thr = pick_threshold(S_val, target_fa)
    a_h = alarm_times(S_healthy, wt, thr)
    a_f = alarm_times(S_faulty, wt, thr, fmeta)
    # crash ke baad bhi baja ya nahi --- ye "bekaar alarm" ginne ke liye
    a_f_unmasked = alarm_times(S_faulty, wt, thr)
    res = report_from_alarms(name, a_h, a_f, fmeta, a_f_unmasked)
    res["threshold"] = thr
    return res


def report_from_alarms(name, alarms_healthy, alarms_faulty, fmeta,
                       alarms_faulty_unmasked=None):
    """ALARM TIMES se report banao --- scores se nahi.

    Ye alag isliye hai taaki single-scale aur multi-scale dono systems
    BILKUL ek hi metric code se guzrein. Agar dono ke liye alag alag
    hisaab likhta, to farq system ka hai ya measurement ka --- pata hi
    nahi chalta.

    ASLI METRIC: LEAD TIME --- alarm aur crash ke beech kitne second the.
    "Fault detect kar liya" ka koi matlab nahi agar drone pehle hi gir chuka ho.
    """
    fa = float(np.mean([a is not None for a in alarms_healthy]))

    hits, leads, lags, per_kind, per_kind_lead = [], [], [], {}, {}
    useless = 0

    for i, (a, m) in enumerate(zip(alarms_faulty, fmeta)):
        crash = m.get("crash_s", -1.0)
        kind = m["fault_kind"]
        alarm = a is not None

        per_kind.setdefault(kind, []).append(alarm)
        hits.append(alarm)

        if alarm:
            lags.append(a - m["fault_start_s"])
            if crash > 0:
                leads.append(crash - a)
                per_kind_lead.setdefault(kind, []).append(crash - a)
        elif crash > 0 and alarms_faulty_unmasked is not None \
                and alarms_faulty_unmasked[i] is not None:
            useless += 1          # baja to sahi, par gir jaane ke baad

    recall = float(np.mean(hits))
    leads = np.array(leads)
    lags = np.array(lags)
    S_healthy = alarms_healthy
    n_h, n_f = len(alarms_healthy), len(alarms_faulty)

    print()
    print("=" * 78)
    print(f"  {name}")
    print("=" * 78)
    print(f"  False alarm rate (test healthy) : {100 * fa:5.1f}%   "
          f"({int(round(fa * n_h))}/{n_h} flights)")
    print(f"  Recall  (crash se PEHLE alarm)  : {100 * recall:5.1f}%   "
          f"({int(round(recall * n_f))}/{n_f} flights)")
    if useless:
        print(f"  ... aur {useless} flights me alarm crash ke BAAD baja (bekaar)")

    # DENOMINATOR DHYAN SE.
    # "X% flights me 5+ second mile" ka denominator SAARI crashed flights hai,
    # sirf pakdi gayi flights nahi. Warna jo detector sirf aasan (lambe lead
    # wale) cases pakadta hai wo is metric pe khud-ba-khud accha dikhega ---
    # jabki asal me wo zyada crashes miss kar raha hai.
    n_crashed = sum(1 for m in fmeta if m.get("crash_s", -1.0) > 0)

    def pct_at_least(lim):
        return float(np.sum(leads >= lim) / n_crashed) if n_crashed else 0.0

    print()
    print(f"  >>> LEAD TIME --- pilot ko kitne second ki warning mili")
    print(f"      ({n_crashed} flights sach me crash hui; miss = 0 second ki warning)")
    if len(leads):
        print(f"        median : {np.median(leads):5.1f} s   (sirf pakdi gayi flights me)")
        print(f"        p25    : {np.percentile(leads, 25):5.1f} s   (kharab cases)")
        print(f"        min    : {leads.min():5.1f} s   /   max : {leads.max():5.1f} s")
        print()
        for lim in (3, 5, 10):
            print(f"        {100 * pct_at_least(lim):5.0f}% of ALL {n_crashed} crashes me "
                  f"{lim}+ second ki warning mili")
    else:
        print("        koi lead time nahi --- ek bhi alarm crash se pehle nahi baja")

    print()
    print("  Fault type ke hisaab se (recall / median lead):")
    kind_recall, kind_lead = {}, {}
    for kind in ["motor_degradation", "prop_damage", "battery_sag", "imu_drift"]:
        v = per_kind.get(kind, [])
        r = float(np.mean(v)) if v else 0.0
        kind_recall[kind] = r
        L = per_kind_lead.get(kind, [])
        ml = float(np.median(L)) if L else None
        kind_lead[kind] = ml
        ls = f"{ml:5.1f}s" if ml is not None else "    -"
        print(f"    {kind:<20} {100 * r:5.0f}%  {ls}   {'#' * int(r * 20)}")

    if len(lags):
        print()
        print(f"  (detection lag, fault shuru hone se: median {np.median(lags):.1f}s)")
    print()

    return {
        "name": name, "false_alarm": fa,
        "recall": recall, "per_kind": kind_recall, "per_kind_lead": kind_lead,
        "lead_median": float(np.median(leads)) if len(leads) else None,
        "lead_p25": float(np.percentile(leads, 25)) if len(leads) else None,
        "lead_min": float(leads.min()) if len(leads) else None,
        "lead_ge3": pct_at_least(3),
        "lead_ge5": pct_at_least(5),      # denominator = SAARI crashed flights
        "lead_ge10": pct_at_least(10),
        "n_crashed": n_crashed,
        "useless_alarms": useless,
        "lag_median": float(np.median(lags)) if len(lags) else None,
    }
