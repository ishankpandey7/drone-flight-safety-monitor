"""
Quadcopter flight telemetry simulator, with fault injection.
============================================================

Yahan hum ek chhota sa drone "physics + autopilot" bana rahe hain.

Sabse important baat samajhne ki:

    Autopilot ko NAHI pata ki koi motor kharab hai.
    Wo sirf itna dekhta hai ki "drone jhuk raha hai", aur usko seedha
    karne ke liye us motor ko aur zor se chalata hai.

Isi wajah se fault ka nishaan PWM me apne aap ubhar kar aata hai --- humein
usko haath se "banana" nahi padta. Yahi cheez is data ko asli banati hai,
aur yahi cheez model ke liye seekhne layak hai.
"""

import numpy as np
from dataclasses import dataclass

# ---------------------------------------------------------------- constants

DT        = 0.02      # 50 Hz -> har 0.02 second me ek reading
G         = 9.81      # gravity
MASS      = 1.2       # kg
ARM       = 0.25      # motor se center tak ki doori (m)
T_MAX     = 10.0      # ek motor ka max thrust (Newton), full battery pe
IXX = IYY = 0.015     # roll/pitch inertia
IZZ       = 0.028     # yaw inertia
K_YAW     = 0.02      # thrust se yaw torque ka ratio
V_NOM     = 14.8      # 4S battery nominal voltage
V_FULL    = 16.8      # full charge

# Motor layout (quad-X), upar se dekhne pe:
#
#        M1 (CCW)   M2 (CW)
#             \     /
#              \   /
#               [X]        <- nose upar ki taraf
#              /   \
#             /     \
#        M4 (CW)    M3 (CCW)
#
# Roll  (+ = right side neeche)  -> left motors  (1,4) zyada thrust dete hain
# Pitch (+ = nose upar)          -> rear motors  (3,4) zyada thrust dete hain
# Yaw   (+ = CW ghoomna)         -> CCW motors   (1,3) zyada thrust dete hain

TELEMETRY_CHANNELS = [
    "t",
    "roll", "pitch", "yaw",
    "gyro_x", "gyro_y", "gyro_z",
    "accel_x", "accel_y", "accel_z",
    "alt", "vz", "pos_x", "pos_y",
    "pwm_1", "pwm_2", "pwm_3", "pwm_4",
    "voltage", "current",
    "vibe_x", "vibe_y", "vibe_z",
    "fault_active",
]


# ---------------------------------------------------------------- helpers

class PID:
    """Ek chhota PID controller.

    P = abhi kitni galti hai
    I = galti kitni der se chali aa rahi hai   <-- fault ka nishaan yahin banta hai
    D = galti kitni tezi se badal rahi hai

    Jab ek motor kamzor padta hai, drone us taraf jhukne lagta hai. Galti
    hataa nahi karti, isliye I term dheere dheere badhta jaata hai aur us
    motor ko zyada se zyada PWM deta hai. Wahi hamara signal hai.
    """

    def __init__(self, kp, ki, kd, i_limit=2.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_limit = i_limit
        self.integral = 0.0

    def step(self, error, rate, dt):
        self.integral += error * dt
        # integral ko cap karna zaroori hai warna wo bhaag jaata hai ("windup")
        self.integral = float(np.clip(self.integral, -self.i_limit, self.i_limit))
        return self.kp * error + self.ki * self.integral - self.kd * rate


@dataclass
class Fault:
    """Ek fault jo flight ke beech me shuru hota hai aur BADHTA rehta hai.

    Pehle version me fault ek had pe jaakar ruk jaata tha (motor ka health
    0.65 pe sthir). Wo galat tha --- asli duniya me marta hua motor sthir
    nahi hota, wo marta rehta hai jab tak drone gir na jaye.

    Ab degradation ki koi had nahi hai. `ramp_s` sirf ye batata hai ki wo
    kitni TEZI se bigadta hai.
    """
    kind: str             # motor_degradation | prop_damage | battery_sag | imu_drift
    start_s: float        # kis second pe shuru hua
    motor: int = 0        # kaunsa motor (0..3), jahan relevant ho
    ramp_s: float = 20.0  # itne second me poori tarah bigad jaata hai


# Crash kab maana jaaye
CRASH_TILT   = 1.05    # rad (~60 deg) --- itna jhuk gaya to wapas nahi aa sakta
CRASH_SINK   = -1.5    # m/s --- zameen se itni tezi se takraya
CRASH_MIN_T  = 6.0     # takeoff ke pehle 6 second ko crash mat samajho

# Autopilot ki apni limits (asli autopilot me bhi aisi hi hoti hain)
MAX_TILT_SP  = 0.35    # rad (~20 deg) --- isse zyada tilt pilot maang hi nahi sakta
TILT_RATE    = 0.6     # rad/s --- tilt setpoint itni tezi se hi badal sakta hai
YAW_RATE     = 1.2     # rad/s (~70 deg/s) --- heading bhi jhatke se nahi badalta


# ---------------------------------------------------------------- simulator

def simulate_flight(duration_s=120.0, fault=None, seed=0):
    """Ek poori flight simulate karo aur telemetry return karo.

    Returns: dict[str, np.ndarray] -- har telemetry channel ek array.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s / DT)

    # ================================================================
    # HAR FLIGHT KI APNI "PERSONALITY"
    # ----------------------------------------------------------------
    # Ye hissa sabse zaroori hai. Agar saari healthy flights bilkul ek
    # jaisi hongi, to model ka kaam bakwaas aasan ho jayega: "jo bhi
    # 0.49 nahi hai wo fault". Aisa model test me 99% dega aur asli
    # log pe fail ho jayega.
    #
    # Isliye hum healthy drone me bhi wahi cheezein daalte hain jo asli
    # duniya me hoti hain --- KHAAS kar ke wo do jo PERMANENT ASYMMETRY
    # banati hain, kyunki wahi fault se confuse ho sakti hain.
    # ================================================================

    # 1) Do motor kabhi bilkul ek jaise nahi bante (+- ~3%)
    motor_eff = np.clip(rng.normal(1.0, 0.025, 4), 0.90, 1.10)

    # 2) Payload kabhi center me nahi hota -> drone hamesha halka jhuka rehta hai
    cg_offset = rng.normal(0.0, 0.009, 2)          # meters, x aur y

    # 3) Har pilot / mission ka andaz alag hota hai
    style       = rng.choice(["calm", "normal", "aggressive"], p=[0.3, 0.45, 0.25])
    maneuver_k  = {"calm": 0.35, "normal": 1.0, "aggressive": 1.9}[style]
    wind_k      = {"calm": 0.5,  "normal": 1.0, "aggressive": 2.2}[style]

    # 4) Har drone ka sensor thoda alag shor karta hai
    gyro_noise  = rng.uniform(0.008, 0.020)
    att_noise   = rng.uniform(0.003, 0.007)

    # 5) Battery har baar poori charged nahi hoti
    charge = rng.uniform(0.80, 1.0)

    # ---- drone ki asli state (ye "sach" hai, sensor ka reading nahi) ----
    pos  = np.zeros(3)          # x, y, z
    vel  = np.zeros(3)
    att  = np.zeros(3)          # roll, pitch, yaw
    rate = np.zeros(3)          # angular velocity

    # ---- battery ----
    r_intern = rng.uniform(0.030, 0.042)    # ohm, internal resistance

    # ---- motor health: 1.0 = bilkul theek ----
    health     = np.ones(4)
    gyro_bias  = np.zeros(3)
    vibe_extra = np.zeros(3)

    # ---- controllers ----
    pid_z     = PID(kp=4.0,  ki=1.2, kd=3.5, i_limit=3.0)
    pid_roll  = PID(kp=40.0, ki=8.0, kd=8.0, i_limit=1.5)
    pid_pitch = PID(kp=40.0, ki=8.0, kd=8.0, i_limit=1.5)
    pid_yaw   = PID(kp=12.0, ki=2.0, kd=4.0, i_limit=1.0)

    # ---- mission ----
    # Asli drone kabhi ek jagah pathar ki tarah khada nahi rehta. Do cheezein
    # chalti rehti hain saath saath:
    #   (a) waypoints  -> bade, achanak wale badlav
    #   (b) lagatar halki wander -> pilot ka haath, GPS ka shor, turbulence
    z_target   = 0.0
    z_sp       = 0.0
    yaw_target = 0.0
    yaw_sp     = 0.0
    roll_base = pitch_base = 0.0
    roll_ou   = pitch_ou   = 0.0       # lagatar drift karne wala hissa
    roll_sp   = pitch_sp   = 0.0       # rate-limited, jhatke se nahi badalta
    n_wp = int(rng.integers(4, 10))
    waypoint_times = np.sort(rng.uniform(6, duration_s - 6, size=n_wp))
    wp_idx = 0

    wind = np.zeros(3)          # dheere badalne wala background hawa
    gust = np.zeros(3)          # achanak aane wale jhonke

    log = {k: np.zeros(n) for k in TELEMETRY_CHANNELS}
    lx = ARM / np.sqrt(2)
    crash_t, crash_reason = -1.0, ""      # -1 = flight bach gayi

    for i in range(n):
        t = i * DT

        # ---------------------------------------------------- fault injection
        f_active = 0.0
        if fault is not None and t >= fault.start_s:
            # progress: 1.0 pe "poori tarah bigda", par YAHIN RUKTA NAHI.
            # 3.0 pe cap sirf isliye ki numbers phat na jaayein.
            p = min(3.0, (t - fault.start_s) / max(fault.ramp_s, 1e-6))
            f_active = 1.0

            if fault.kind == "motor_degradation":
                # bearing ghis raha hai / winding garam ho rahi hai.
                # Thrust girta jaata hai. Ek waqt aata hai jab baaki teen
                # motor drone ko seedha nahi rakh paate -> palat jaata hai.
                health[fault.motor] = max(0.05, 1.0 - 0.95 * p)
                vibe_extra[:] = 0.5 * min(p, 1.5)

            elif fault.kind == "prop_damage":
                # Propeller ka tukda toota --- pehla jhatka achanak, phir
                # baaki blade bhi tootta jaata hai (unbalanced load se).
                health[fault.motor] = max(0.10, 0.88 - 0.80 * p)
                vibe_extra[:] = 2.5 + 2.0 * min(p, 1.5)

            elif fault.kind == "battery_sag":
                # Cell ka internal resistance badhta jaata hai. Aage jaakar
                # DEATH SPIRAL shuru hota hai: voltage girti hai -> thrust
                # kam -> controller PWM badhata hai -> current badhta hai ->
                # voltage aur girti hai. Ek point ke baad zyada PWM dene se
                # ULTA KAM thrust milta hai, aur drone bas gir jaata hai.
                r_intern = 0.035 + 0.40 * p

            elif fault.kind == "imu_drift":
                # Gyro jhooth bolne laga. Autopilot ek aisi tilt ko theek
                # karne ki koshish karta hai jo hai hi nahi --- aur asal me
                # drone ko ulti taraf jhuka deta hai. Bias jitna badhta hai,
                # asli tilt utni badhti hai, aur ek din wo palat jaata hai.
                gyro_bias[0] = 1.8 * p
                gyro_bias[2] = 0.7 * p

        # ---------------------------------------------------- mission logic
        if t < 4.0:
            z_target = 2.0                               # takeoff
        elif wp_idx < len(waypoint_times) and t >= waypoint_times[wp_idx]:
            z_target   = rng.uniform(1.5, 6.5)           # nayi altitude
            yaw_target = rng.uniform(-np.pi, np.pi)      # naya heading
            roll_base  = rng.uniform(-0.20, 0.20) * maneuver_k
            pitch_base = rng.uniform(-0.20, 0.20) * maneuver_k
            wp_idx += 1

        # altitude setpoint jhatke se nahi, dheere dheere badalta hai
        # (asli autopilot me bhi trajectory smooth hoti hai)
        z_sp += np.clip(z_target - z_sp, -1.5 * DT, 1.5 * DT)

        # heading bhi dheere ghoomta hai. Pehle ye ek hi step me poora pi
        # radian ghoom jaata tha --- yaw PID phir itna torque maangta tha ki
        # mixer poora saturate ho jaata aur drone palat jaata.
        dyaw = (yaw_target - yaw_sp + np.pi) % (2 * np.pi) - np.pi
        yaw_sp += float(np.clip(dyaw, -YAW_RATE * DT, YAW_RATE * DT))
        yaw_sp = (yaw_sp + np.pi) % (2 * np.pi) - np.pi

        # lagatar halki wander --- yahi cheez "normal" ko boring hone se bachati hai
        roll_ou  = 0.992 * roll_ou  + rng.normal(0, 0.004) * maneuver_k
        pitch_ou = 0.992 * pitch_ou + rng.normal(0, 0.004) * maneuver_k
        roll_want  = float(np.clip(roll_base + roll_ou,  -MAX_TILT_SP, MAX_TILT_SP))
        pitch_want = float(np.clip(pitch_base + pitch_ou, -MAX_TILT_SP, MAX_TILT_SP))

        # Tilt setpoint JHATKE se nahi badalta --- asli autopilot use rate-limit
        # karta hai. Bina iske ek waypoint step 40 degree ka swing maang leta
        # hai, PID overshoot karta hai, aur HEALTHY drone hi palat jaata hai.
        # (Pehle version me 40 me se 7 healthy flights isi wajah se gir rahi thi.)
        roll_sp  += float(np.clip(roll_want - roll_sp,  -TILT_RATE * DT, TILT_RATE * DT))
        pitch_sp += float(np.clip(pitch_want - pitch_sp, -TILT_RATE * DT, TILT_RATE * DT))

        # hawa: dheema background + kabhi kabhi achanak jhonka
        wind = 0.995 * wind + rng.normal(0, 0.06, 3) * wind_k
        gust *= 0.97
        if rng.random() < 0.004 * wind_k:                # ~har kuch second me
            gust = rng.normal(0, 0.9, 3) * wind_k

        # ---------------------------------------------------- sensors
        # NOTE: controller ko sirf ye "measured" values dikhte hain, asli sach nahi
        gyro_meas = rate + gyro_bias + rng.normal(0, gyro_noise, 3)
        att_meas  = att + gyro_bias * 0.6 + rng.normal(0, att_noise, 3)

        # ---------------------------------------------------- control
        thrust_cmd = MASS * (G + pid_z.step(z_sp - pos[2], vel[2], DT))
        thrust_cmd = float(np.clip(thrust_cmd, 0.0, 4 * T_MAX * 0.95))

        tau_x = IXX * pid_roll.step(roll_sp - att_meas[0], gyro_meas[0], DT)
        tau_y = IYY * pid_pitch.step(pitch_sp - att_meas[1], gyro_meas[1], DT)
        yaw_err = (yaw_sp - att_meas[2] + np.pi) % (2 * np.pi) - np.pi
        tau_z = IZZ * pid_yaw.step(yaw_err, gyro_meas[2], DT)

        # ---------------------------------------------------- mixer
        # YAW KI PRIORITY SABSE KAM HAI.
        #
        # Agar mixer saturate ho raha hai to sabse pehle yaw authority kaato.
        # Wajah seedhi hai: roll/pitch ka control khoya = CRASH. Yaw ka control
        # khoya = drone bas ghoom jayega, par udta rahega.
        #
        # Bina iske ek yaw command chaaron motors ko 0/1 pe pel deta hai aur
        # HEALTHY drone palat jaata hai. (Yahi bug tha --- 40 me se 7 healthy
        # flights isi wajah se gir rahi thi.) Asli autopilot, PX4 ho ya
        # ArduPilot, bilkul yahi desaturation karta hai.
        for _ in range(8):
            T_des = np.array([
                thrust_cmd / 4 + tau_x / (4 * lx) - tau_y / (4 * lx) + tau_z / (4 * K_YAW),
                thrust_cmd / 4 - tau_x / (4 * lx) - tau_y / (4 * lx) - tau_z / (4 * K_YAW),
                thrust_cmd / 4 - tau_x / (4 * lx) + tau_y / (4 * lx) + tau_z / (4 * K_YAW),
                thrust_cmd / 4 + tau_x / (4 * lx) + tau_y / (4 * lx) - tau_z / (4 * K_YAW),
            ])
            if T_des.max() <= T_MAX and T_des.min() >= 0.0:
                break
            tau_z *= 0.5
        T_des = np.clip(T_des, 0.0, T_MAX)

        # ---- YAHI woh do line hain jo sab kuch asli banati hain ----
        # Controller PWM nikalne ke liye maanta hai ki battery NOMINAL pe hai
        # aur saare motors health=1.0 hain. Usko nahi pata ki kuch kharab hai.
        pwm = np.sqrt(T_des / T_MAX)
        pwm = np.clip(pwm, 0.0, 1.0)

        # ---------------------------------------------------- battery
        voltage_ocv = V_FULL - (V_FULL - 13.2) * (1 - charge)
        current = 1.5 + 9.0 * float(np.sum(pwm ** 2.2))
        voltage = voltage_ocv - current * r_intern
        # Clamp zaroori hai: death spiral me current itna badh jaata hai ki
        # bina clamp ke voltage MINUS me chali jaati hai --- aur phir
        # v_factor = (v/V_NOM)**2 use wapas positive bana deta, yani kharab
        # battery se ZYADA thrust. Bilkul ulta physics.
        voltage = max(voltage, 4.0)
        charge -= (current * DT) / (3600 * 5.0)          # 5 Ah pack
        charge = max(charge, 0.05)

        # ---------------------------------------------------- plant (asli physics)
        # Asli thrust = command * factory variation * motor ki sehat * battery
        #
        # motor_eff yahan HEALTHY drone me bhi maujood hai (+- 3%). Isi wajah
        # se healthy drone me bhi halki permanent asymmetry rehti hai --- aur
        # isiliye model ko "asymmetry hai ya nahi" nahi, "asymmetry KITNI hai
        # aur kya wo flight ke context se match karti hai" seekhna padega.
        v_factor = (voltage / V_NOM) ** 2
        T_act = T_MAX * (pwm ** 2) * motor_eff * health * v_factor

        total_T = float(np.sum(T_act))
        torque = np.array([
            lx * (T_act[0] - T_act[1] - T_act[2] + T_act[3]),        # roll
            lx * (-T_act[0] - T_act[1] + T_act[2] + T_act[3]),       # pitch
            K_YAW * (T_act[0] - T_act[1] + T_act[2] - T_act[3]),     # yaw
        ])

        # Payload center me nahi hai -> uska weight ek permanent torque banata hai,
        # jise controller ko hamesha counter karna padta hai.
        torque[0] += -cg_offset[1] * total_T
        torque[1] += cg_offset[0] * total_T

        rate += (torque / np.array([IXX, IYY, IZZ])) * DT
        rate *= 0.998                                    # aerodynamic damping
        att += rate * DT
        att[2] = (att[2] + np.pi) % (2 * np.pi) - np.pi

        # thrust body ke Z axis me lagta hai; tilt se horizontal force banta hai
        sr, cr = np.sin(att[0]), np.cos(att[0])
        sp_, cp = np.sin(att[1]), np.cos(att[1])
        air = wind + gust
        acc = np.array([
            (total_T / MASS) * (sp_ * np.cos(att[2]) + sr * np.sin(att[2])) + air[0],
            (total_T / MASS) * (sp_ * np.sin(att[2]) - sr * np.cos(att[2])) + air[1],
            (total_T / MASS) * cr * cp - G + air[2] * 0.3,
        ])
        vel += acc * DT
        vel *= 0.999
        pos += vel * DT
        if pos[2] < 0:                                   # zameen se neeche mat jao
            pos[2] = 0.0
            vel[2] = max(vel[2], 0.0)

        # ---------------------------------------------------- vibration
        # base vibration motor ki speed se aata hai; fault use badha deta hai
        base_vibe = 0.25 + 1.8 * float(np.std(pwm)) + 0.5 * float(np.mean(pwm)) ** 2
        vibe = base_vibe + vibe_extra + np.abs(rng.normal(0, 0.08, 3))

        # ---------------------------------------------------- log
        log["t"][i] = t
        log["roll"][i], log["pitch"][i], log["yaw"][i] = att
        log["gyro_x"][i], log["gyro_y"][i], log["gyro_z"][i] = gyro_meas
        log["accel_x"][i], log["accel_y"][i], log["accel_z"][i] = acc + rng.normal(0, 0.05, 3)
        log["alt"][i], log["vz"][i] = pos[2], vel[2]
        log["pos_x"][i], log["pos_y"][i] = pos[0], pos[1]
        log["pwm_1"][i], log["pwm_2"][i], log["pwm_3"][i], log["pwm_4"][i] = pwm
        log["voltage"][i], log["current"][i] = voltage, current
        log["vibe_x"][i], log["vibe_y"][i], log["vibe_z"][i] = vibe
        log["fault_active"][i] = f_active

        # ---------------------------------------------------- crash?
        # Quadcopter do tareeke se marta hai:
        #   1. PALAT jaata hai --- itna jhuk gaya ki thrust neeche ki taraf
        #      nahi bachi, aur controller wapas seedha nahi kar sakta
        #   2. GIR jaata hai --- thrust weight se kam pad gaya, zameen se takraya
        if t > CRASH_MIN_T and crash_t < 0:
            tipped  = abs(att[0]) > CRASH_TILT or abs(att[1]) > CRASH_TILT
            slammed = pos[2] <= 0.05 and vel[2] < CRASH_SINK
            if tipped or slammed:
                crash_t = t
                crash_reason = "palat gaya" if tipped else "zameen se takraya"
                # Array ka size fix rehna chahiye, isliye baaki ko aakhri
                # value se bhar do. Crash ke baad ka data waise bhi bekaar
                # hai --- evaluation me use mask kar dete hain.
                for k in log:
                    if k != "t":
                        log[k][i + 1:] = log[k][i]
                log["t"][i + 1:] = np.arange(i + 1, n) * DT
                break

    meta = {
        "style": str(style),
        "motor_eff": motor_eff.round(4).tolist(),
        "cg_offset": np.round(cg_offset, 4).tolist(),
        "start_charge": round(float(charge), 3),
        "fault_kind": fault.kind if fault else "none",
        "fault_start_s": fault.start_s if fault else -1.0,
        "fault_motor": fault.motor if fault else -1,
        "fault_ramp_s": fault.ramp_s if fault else 0.0,
        "crash_s": float(crash_t),          # -1 agar flight bach gayi
        "crash_reason": crash_reason,
        "survived_s": float(crash_t - fault.start_s) if (fault and crash_t > 0) else -1.0,
        "seed": seed,
        "duration_s": duration_s,
    }
    return log, meta


if __name__ == "__main__":
    import pandas as pd

    DUR = 180.0
    FAULT_AT = 80.0

    print("1) Ek healthy flight...")
    healthy, h_meta = simulate_flight(duration_s=DUR, fault=None, seed=1)

    print("2) Wahi drone, wahi mission -- lekin motor 3 dheere dheere kharab ho raha hai...")
    broken, b_meta = simulate_flight(
        duration_s=DUR,
        fault=Fault(kind="motor_degradation", start_s=FAULT_AT, motor=2, ramp_s=20.0),
        seed=1,
    )

    import json as _json
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parent.parent
    (_root / "data" / "raw").mkdir(parents=True, exist_ok=True)

    for name, log in [("healthy", healthy), ("motor_fault", broken)]:
        df = pd.DataFrame(log)
        path = _root / "data" / "raw" / f"demo_{name}.csv"
        df.to_csv(path, index=False)
        print(f"   saved {path.name}  ({len(df)} rows, {len(df.columns)} channels)")

    # Meta bhi save karo --- crash kab hua ye CSV me nahi hai, aur uske bina
    # check_signature.py crash ke BAAD ka data padh lega aur jhoothe numbers dega.
    with open(_root / "data" / "raw" / "demo_meta.json", "w") as f:
        _json.dump({"healthy": h_meta, "motor_fault": b_meta}, f, indent=1)

    print()
    print(f"   healthy flight crash?     : "
          f"{'NAHI (theek hai)' if h_meta['crash_s'] < 0 else 'HAAN -- ye BUG hai!'}")
    if b_meta["crash_s"] > 0:
        print(f"   faulty flight crash       : {b_meta['crash_s']:.1f}s pe, {b_meta['crash_reason']}")
        print(f"   fault se crash tak         : {b_meta['survived_s']:.1f} seconds")
        print(f"   >> detector ke paas itna hi waqt hai alarm bajane ka")
    else:
        print(f"   faulty flight crash       : nahi hua ({DUR:.0f}s me) -- ramp dheemi hai")
    print()
    print(f"   flight style              : {h_meta['style']}")
    print(f"   motor factory variation   : {h_meta['motor_eff']}   <- healthy drone me bhi!")
    print(f"   payload CG offset (m)     : {h_meta['cg_offset']}")
    print()
    print(f"   average altitude          : {healthy['alt'].mean():6.2f} m")
    print(f"   max altitude              : {healthy['alt'].max():6.2f} m")
    pwms = np.array([healthy["pwm_%d" % i] for i in range(1, 5)])
    print(f"   average PWM               : {pwms.mean():6.3f}")
    print(f"   PWM ka std deviation      : {pwms.std():6.3f}   <- pehle ye lagbhag 0 tha")
