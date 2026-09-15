"""
STEP 7 --- Interactive flight replay dashboard.
===============================================

Ek self-contained HTML file banata hai jisme tum ek asli flight ko replay
kar sakte ho aur dekh sakte ho ki anomaly score kab uthta hai.

Ye wahi demo hai jo shuru me plan kiya tha: timeline chalti hai, sab normal,
phir score chadhta hai --- aur uske baad fault. Koi bounding box is se
compete nahi kar sakta.

Data seedha HTML me embed hota hai (10 Hz pe, taaki file chhoti rahe), toh
file kahin bhi khul jayegi --- server ki zaroorat nahi.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DETECTOR = "baseline"          # sabse accha detector --- compare.py ne yahi bataya

SHOW_CH = ["pwm_1", "pwm_2", "pwm_3", "pwm_4",
           "vibe_x", "voltage", "gyro_x", "alt"]
TELL = {"motor_degradation": "vibe_x", "prop_damage": "vibe_x",
        "battery_sag": "voltage", "imu_drift": "gyro_x", "none": "alt"}


def build():
    d = np.load(ROOT / "data" / "processed" / "flights.npz", allow_pickle=True)
    meta = json.load(open(ROOT / "data" / "processed" / "meta.json"))
    CH = [str(c) for c in d["channels"]]
    idx = {c: i for i, c in enumerate(CH)}

    z = np.load(ROOT / "outputs" / f"{DETECTOR}_scores.npz")
    thr = float(z["threshold"])
    wt = z["window_times"].tolist()

    fmeta = [m for m in meta if m["split"] == "test_faulty"]
    hmeta = [m for m in meta if m["split"] == "test_healthy"]

    flights = []

    def add(x, s, m, label):
        # 50 Hz -> 10 Hz, aur 3 decimal --- file chhoti rakhne ke liye
        xs = x[::5]
        flights.append({
            "label": label,
            "kind": m.get("fault_kind", "none"),
            "faultStart": m.get("fault_start_s", -1),
            "faultMotor": m.get("fault_motor", -1),
            "ramp": round(m.get("fault_ramp_s", 0), 0),
            "crash": round(m.get("crash_s", -1), 1),
            "crashWhy": m.get("crash_reason", ""),
            "style": m.get("style", "?"),
            "dt": 0.1,
            "ch": {c: [round(float(v), 3) for v in xs[:, idx[c]]] for c in SHOW_CH},
            "score": [round(float(v), 2) for v in s],
            "tell": TELL.get(m.get("fault_kind", "none"), "alt"),
        })

    # Har fault type ke 2 example: ek TEZ marne wala, ek DHEEME marne wala.
    # Dono dikhane se saaf pata chalta hai ki lead time kis pe depend karta hai.
    for kind in ["motor_degradation", "prop_damage", "battery_sag", "imu_drift"]:
        cand = sorted([(i, m) for i, m in enumerate(fmeta)
                       if m["fault_kind"] == kind and m["crash_s"] > 0],
                      key=lambda im: im[1]["survived_s"])
        picks = ([cand[0], cand[-1]] if len(cand) > 1 else cand)
        for n, (i, m) in enumerate(picks):
            tag = "tez" if n == 0 else "dheema"
            add(d["test_faulty"][i], z["test_faulty"][i], m,
                f"{kind}  ({tag}: {m['survived_s']:.0f}s me crash)")

    # do healthy flights --- taaki dikhe ki normal kaisa lagta hai
    for i in range(2):
        add(d["test_healthy"][i], z["test_healthy"][i], hmeta[i],
            f"healthy  #{i + 1}  ({hmeta[i]['style']})")

    results = {}
    for tag in ["baseline", "lstm", "forecaster", "hybrid_a", "hybrid_b"]:
        p = ROOT / "outputs" / f"{tag}_result.json"
        if p.exists():
            results[tag] = json.load(open(p))

    payload = {"flights": flights, "threshold": thr, "windowTimes": wt,
               "detector": DETECTOR, "results": results}

    html = TEMPLATE.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    out = ROOT / "outputs" / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"  saved -> {out}  ({kb:.0f} KB, {len(flights)} flights)")
    return out


TEMPLATE = r"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Drone Flight Safety Monitor</title>
<style>
  :root{
    --bg:#f7f8fa; --panel:#fff; --ink:#12161f; --muted:#5d6b80; --line:#e3e7ee;
    --ok:#0f9d58; --warn:#e8a33d; --bad:#e5484d; --accent:#4a7ddb;
    --m1:#5B8DEF; --m2:#57B894; --m3:#E5484D; --m4:#C9A227;
  }
  @media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
    --bg:#0d1017; --panel:#151a23; --ink:#e8ecf3; --muted:#8e9bb0; --line:#242b38;
  }}
  :root[data-theme="dark"]{
    --bg:#0d1017; --panel:#151a23; --ink:#e8ecf3; --muted:#8e9bb0; --line:#242b38;
  }
  *{box-sizing:border-box}
  body{background:var(--bg);color:var(--ink);
    font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
    margin:0;padding:18px}
  .wrap{max-width:1060px;margin:0 auto}
  h1{font-size:19px;margin:0 0 2px;letter-spacing:-.2px}
  .sub{color:var(--muted);font-size:13px;margin-bottom:16px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
    padding:14px 16px;margin-bottom:12px}
  .row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
  select,button{font:inherit;color:var(--ink);background:var(--panel);
    border:1px solid var(--line);border-radius:8px;padding:7px 11px;cursor:pointer}
  button:hover,select:hover{border-color:var(--accent)}
  button.primary{background:var(--accent);color:#fff;border-color:var(--accent);
    font-weight:600;min-width:92px}
  .status{display:flex;align-items:center;gap:14px;padding:13px 16px;border-radius:10px;
    font-weight:700;font-size:16px;letter-spacing:.3px;transition:background .25s}
  .dot{width:12px;height:12px;border-radius:50%;flex:none}
  .s-ok{background:color-mix(in srgb,var(--ok) 12%,transparent);color:var(--ok)}
  .s-warn{background:color-mix(in srgb,var(--warn) 15%,transparent);color:var(--warn)}
  .s-bad{background:color-mix(in srgb,var(--bad) 14%,transparent);color:var(--bad)}
  .metrics{margin-left:auto;display:flex;gap:20px;font-weight:500;font-size:13px;
    color:var(--muted);font-variant-numeric:tabular-nums}
  .metrics b{color:var(--ink);font-variant-numeric:tabular-nums}
  canvas{width:100%;display:block}
  .ctitle{font-size:12px;font-weight:700;color:var(--muted);
    text-transform:uppercase;letter-spacing:.6px;margin-bottom:6px}
  .legend{display:flex;gap:14px;font-size:12px;color:var(--muted);margin-top:4px}
  .sw{display:inline-block;width:10px;height:3px;border-radius:2px;
    vertical-align:middle;margin-right:5px}
  input[type=range]{flex:1;min-width:200px;accent-color:var(--accent)}
  .time{font-variant-numeric:tabular-nums;color:var(--muted);min-width:96px;
    text-align:right;font-size:13px}
  table{border-collapse:collapse;width:100%;font-size:13px;
    font-variant-numeric:tabular-nums}
  th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:right}
  th:first-child,td:first-child{text-align:left}
  th{color:var(--muted);font-weight:600;font-size:11.5px;text-transform:uppercase;
    letter-spacing:.5px}
  tr.best td{font-weight:700;color:var(--accent)}
  .note{color:var(--muted);font-size:12.5px;margin-top:9px;line-height:1.55}
</style>

<div class="wrap">
  <h1>Drone Flight Safety Monitor</h1>
  <div class="sub">Telemetry anomaly detection &mdash; flight replay karo aur dekho detector kab bolta hai</div>

  <div class="card">
    <div class="row">
      <select id="sel"></select>
      <button class="primary" id="play">▶ Play</button>
      <button id="rst">↺ Reset</button>
      <span id="info" style="color:var(--muted);font-size:12.5px"></span>
    </div>
  </div>

  <div class="card" style="padding:0;overflow:hidden">
    <div class="status s-ok" id="status">
      <span class="dot" id="dot" style="background:var(--ok)"></span>
      <span id="stext">NORMAL</span>
      <span class="metrics">
        <span>score <b id="mScore">—</b></span>
        <span>threshold <b id="mThr">—</b></span>
        <span>t <b id="mT">0.0s</b></span>
        <span id="mAlarmWrap" hidden>warning <b id="mAlarm">—</b></span>
      </span>
    </div>
  </div>

  <div class="card">
    <div class="ctitle">Motors (PWM)</div>
    <canvas id="c1" height="150"></canvas>
    <div class="legend">
      <span><i class="sw" style="background:var(--m1)"></i>Motor 1</span>
      <span><i class="sw" style="background:var(--m2)"></i>Motor 2</span>
      <span><i class="sw" style="background:var(--m3)"></i>Motor 3</span>
      <span><i class="sw" style="background:var(--m4)"></i>Motor 4</span>
    </div>
  </div>

  <div class="card">
    <div class="ctitle" id="t2">Tell channel</div>
    <canvas id="c2" height="120"></canvas>
  </div>

  <div class="card">
    <div class="ctitle">Anomaly score</div>
    <canvas id="c3" height="150"></canvas>
    <div class="legend">
      <span><i class="sw" style="background:var(--ink)"></i>score</span>
      <span><i class="sw" style="background:var(--bad)"></i>threshold</span>
      <span><i class="sw" style="background:var(--warn)"></i>fault shuru (ground truth)</span>
      <span><i class="sw" style="background:var(--ok)"></i>alarm baja</span>
      <span><i class="sw" style="background:#B91C1C"></i>CRASH</span>
    </div>
  </div>

  <div class="card">
    <div class="row" style="margin-bottom:10px">
      <input type="range" id="scrub" min="0" max="1000" value="0">
      <span class="time" id="tlabel">0.0 / 0.0 s</span>
    </div>
  </div>

  <div class="card">
    <div class="ctitle">Detectors ka muqabla</div>
    <table id="res"></table>
    <div class="note" id="resnote"></div>
  </div>
</div>

<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

let cur = 0, t = 0, playing = false, raf = null, last = 0;

// ---------------------------------------------------------------- setup
D.flights.forEach((f, i) => {
  const o = document.createElement('option');
  o.value = i; o.textContent = f.label; $('sel').appendChild(o);
});

function dur(f){
  // Replay crash ke thodi der baad tak hi chalao --- uske aage drone
  // zameen pe pada hai, dekhne ko kuch nahi.
  const full = (f.ch.pwm_1.length - 1) * f.dt;
  return f.crash > 0 ? Math.min(full, f.crash + 3) : full;
}

function alarmBy(f, time){
  // Wahi rule jo evaluation me hai: 3 lagatar windows threshold paar karein.
  // Aur ek baar baj gaya to LATCH ho jaata hai --- score wapas neeche aane se
  // fault "theek" nahi ho jaata. Asli safety system aise hi kaam karta hai.
  let run = 0;
  for (let i = 0; i < D.windowTimes.length; i++){
    if (D.windowTimes[i] > time) break;
    // Crash ke baad ki windows ginti nahi. Wahan drone palat raha hai ---
    // use pakadna koi kamaal nahi, aur uska credit lena khud ko dhoka dena hai.
    if (f.crash > 0 && D.windowTimes[i] > f.crash) break;
    run = f.score[i] > D.threshold ? run + 1 : 0;
    if (run >= 3) return D.windowTimes[i];
  }
  return null;
}

function scoreAt(f, time){
  // window ka score uske END time pe available hota hai --- usse pehle nahi
  let v = null;
  for (let i = 0; i < D.windowTimes.length; i++){
    if (D.windowTimes[i] <= time) v = f.score[i]; else break;
  }
  return v;
}

// ---------------------------------------------------------------- drawing
function fit(cv){
  const r = window.devicePixelRatio || 1;
  const w = cv.clientWidth;
  if (cv.width !== w * r){ cv.width = w * r; cv.height = cv.clientHeight * r; }
  const g = cv.getContext('2d');
  g.setTransform(r, 0, 0, r, 0, 0);
  g.clearRect(0, 0, w, cv.clientHeight);
  return [g, w, cv.clientHeight];
}

function axes(g, w, h, lo, hi){
  g.strokeStyle = css('--line'); g.lineWidth = 1;
  for (let i = 0; i <= 3; i++){
    const y = 8 + (h - 22) * i / 3;
    g.beginPath(); g.moveTo(34, y); g.lineTo(w - 6, y); g.stroke();
  }
  g.fillStyle = css('--muted'); g.font = '10px ui-sans-serif,system-ui';
  g.textAlign = 'right';
  for (let i = 0; i <= 3; i++){
    const y = 8 + (h - 22) * i / 3;
    const v = hi - (hi - lo) * i / 3;
    g.fillText(Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2), 30, y + 3.5);
  }
}

function series(g, w, h, arr, dt, lo, hi, col, upto, lw, total){
  // total = jitna waqt X-axis pe dikh raha hai. Array isse LAMBA ho sakta
  // hai (crash ke baad ka frozen data), isliye array.length se normalize
  // karna galat hai --- traces aur vline alag alag jagah aa jaate hain.
  g.strokeStyle = col; g.lineWidth = lw || 1.4; g.beginPath();
  const n = Math.min(arr.length, Math.floor(upto / dt) + 1);
  for (let i = 0; i < n; i++){
    const x = 34 + (w - 40) * (i * dt) / total;
    const y = 8 + (h - 22) * (1 - (arr[i] - lo) / (hi - lo || 1));
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  }
  g.stroke();
}

function vline(g, w, h, time, total, col, dash){
  if (time < 0) return;
  const x = 34 + (w - 40) * time / total;
  g.save(); g.strokeStyle = col; g.lineWidth = 1.6;
  if (dash) g.setLineDash([5, 4]);
  g.beginPath(); g.moveTo(x, 4); g.lineTo(x, h - 14); g.stroke(); g.restore();
}

function range(arrs, pad){
  let lo = Infinity, hi = -Infinity;
  arrs.forEach(a => a.forEach(v => { if (v < lo) lo = v; if (v > hi) hi = v; }));
  const m = (hi - lo) * (pad || 0.08) || 1;
  return [lo - m, hi + m];
}

function draw(){
  const f = D.flights[cur], T = dur(f);

  // --- motors ---
  let [g, w, h] = fit($('c1'));
  const ms = ['pwm_1','pwm_2','pwm_3','pwm_4'].map(k => f.ch[k]);
  axes(g, w, h, 0, 1);
  ['--m1','--m2','--m3','--m4'].forEach((c, i) => {
    const bad = (i === f.faultMotor && (f.kind === 'motor_degradation' || f.kind === 'prop_damage'));
    series(g, w, h, ms[i], f.dt, 0, 1, css(c), t, bad ? 2.2 : 1.1, T);
  });
  vline(g, w, h, f.faultStart, T, css('--warn'), true);
  if (f.crash > 0 && t >= f.crash) vline(g, w, h, f.crash, T, '#B91C1C');
  vline(g, w, h, t, T, css('--accent'));

  // --- tell channel ---
  const tc = f.ch[f.tell];
  [g, w, h] = fit($('c2'));
  // y-range sirf dikhne wale hisse se --- crash ke baad ka frozen data
  // range ko kheench kar baaki sab flat kar deta hai
  const [lo2, hi2] = range([tc.slice(0, Math.floor(T / f.dt) + 1)]);
  axes(g, w, h, lo2, hi2);
  series(g, w, h, tc, f.dt, lo2, hi2, '#7C3AED', t, 1.6, T);
  vline(g, w, h, f.faultStart, T, css('--warn'), true);
  if (f.crash > 0 && t >= f.crash) vline(g, w, h, f.crash, T, '#B91C1C');
  vline(g, w, h, t, T, css('--accent'));
  $('t2').textContent = f.tell + '  —  is fault ka asli nishaan';

  // --- score ---
  [g, w, h] = fit($('c3'));
  const hi3 = Math.max(D.threshold * 1.35, ...f.score) * 1.05;
  axes(g, w, h, 0, hi3);
  const sx = D.windowTimes;
  g.strokeStyle = css('--bad'); g.lineWidth = 1.5; g.save(); g.setLineDash([5,4]);
  const ty = 8 + (h - 22) * (1 - D.threshold / hi3);
  g.beginPath(); g.moveTo(34, ty); g.lineTo(w - 6, ty); g.stroke(); g.restore();
  g.strokeStyle = css('--ink'); g.lineWidth = 2.2; g.beginPath();
  let started = false;
  for (let i = 0; i < sx.length; i++){
    if (sx[i] > t) break;
    const x = 34 + (w - 40) * sx[i] / T;
    const y = 8 + (h - 22) * (1 - Math.min(f.score[i], hi3) / hi3);
    started ? g.lineTo(x, y) : (g.moveTo(x, y), started = true);
  }
  g.stroke();
  const firedAt = alarmBy(f, t);
  if (firedAt !== null) vline(g, w, h, firedAt, T, css('--ok'));
  vline(g, w, h, f.faultStart, T, css('--warn'), true);
  if (f.crash > 0 && t >= f.crash) vline(g, w, h, f.crash, T, '#B91C1C');
  vline(g, w, h, t, T, css('--accent'));

  // --- status ---
  const s = scoreAt(f, t);
  $('mScore').textContent = s === null ? '—' : s.toFixed(1);
  $('mThr').textContent = D.threshold.toFixed(1);
  $('mT').textContent = t.toFixed(1) + 's';
  $('tlabel').textContent = t.toFixed(1) + ' / ' + T.toFixed(1) + ' s';
  $('scrub').value = Math.round(1000 * t / T);

  const fired = alarmBy(f, t);
  $('mAlarmWrap').hidden = (fired === null);
  if (fired !== null){
    $('mAlarm').textContent = f.crash > 0
      ? (f.crash - fired).toFixed(0) + 's pehle'      // <-- YEHI asli number hai
      : '@ ' + fired.toFixed(0) + 's';
  }

  const lvl = fired !== null ? 'bad'
            : s === null ? 'none'
            : s > D.threshold ? 'warn'          // ek window paar, abhi alarm nahi
            : s > D.threshold * 0.6 ? 'warn' : 'ok';
  const box = $('status');
  box.className = 'status s-' + (lvl === 'none' ? 'ok' : lvl);
  const crashed = f.crash > 0 && t >= f.crash;
  $('stext').textContent =
      crashed ? (fired !== null ? 'CRASHED (warning di thi)' : 'CRASHED (bina warning)') :
      fired !== null ? 'FAULT DETECTED' :
      s === null ? 'STARTING…' :
      lvl === 'warn' ? 'ELEVATED' : 'NORMAL';
  $('dot').style.background = lvl === 'none' ? css('--muted') :
      lvl === 'bad' ? css('--bad') : lvl === 'warn' ? css('--warn') : css('--ok');
}

// ---------------------------------------------------------------- results table
function table(){
  const NAMES = {baseline:'Mahalanobis (stats)', lstm:'LSTM autoencoder',
                 forecaster:'LSTM forecaster', hybrid_a:'Hybrid A', hybrid_b:'Hybrid B'};
  const K = ['motor_degradation','prop_damage','battery_sag','imu_drift'];
  const rs = Object.entries(D.results);
  if (!rs.length) return;
  const best = rs.reduce((a, b) => b[1].recall > a[1].recall ? b : a)[0];
  let html = '<tr><th>detector</th><th>recall</th><th>false alarm</th>' +
             K.map(k => '<th>' + k.replace(/_/g, ' ') + '</th>').join('') + '</tr>';
  for (const [tag, r] of rs){
    html += '<tr class="' + (tag === best ? 'best' : '') + '"><td>' + (NAMES[tag] || tag) + '</td>' +
      '<td>' + (100 * r.recall).toFixed(0) + '%</td>' +
      '<td>' + (100 * r.false_alarm).toFixed(0) + '%</td>' +
      K.map(k => '<td>' + (100 * r.per_kind[k]).toFixed(0) + '%</td>').join('') + '</tr>';
  }
  $('res').innerHTML = html;
  $('resnote').textContent =
    'Recall = kitne % faults pakde gaye. False alarm = kitne % healthy flights pe ' +
    'jhootha alarm baja. Threshold hamesha validation set se chuna gaya hai, test se nahi. ' +
    'Upar ka replay "' + (NAMES[D.detector] || D.detector) + '" ke scores dikha raha hai.';
}

// ---------------------------------------------------------------- interaction
function load(i){
  cur = i; t = 0; playing = false; $('play').textContent = '▶ Play';
  const f = D.flights[i];
  $('info').textContent = f.kind === 'none'
    ? 'koi fault nahi — flight style: ' + f.style
    : f.kind.replace(/_/g,' ') + ' • motor ' + (f.faultMotor + 1) +
      ' • ramp ' + f.ramp + 's • fault @ ' + f.faultStart.toFixed(0) + 's' +
      (f.crash > 0 ? ' • CRASH @ ' + f.crash.toFixed(0) + 's (' + f.crashWhy + ')' : '');
  draw();
}

function tick(ts){
  if (!playing) return;
  const f = D.flights[cur], T = dur(f);
  if (last) t = Math.min(T, t + (ts - last) / 1000 * 6);   // 6x speed
  last = ts;
  draw();
  if (t >= T){ playing = false; $('play').textContent = '▶ Play'; return; }
  raf = requestAnimationFrame(tick);
}

$('sel').onchange = e => load(+e.target.value);
$('play').onclick = () => {
  const T = dur(D.flights[cur]);
  if (t >= T) t = 0;
  playing = !playing; last = 0;
  $('play').textContent = playing ? '❚❚ Pause' : '▶ Play';
  if (playing) raf = requestAnimationFrame(tick);
};
$('rst').onclick = () => { t = 0; playing = false; $('play').textContent = '▶ Play'; draw(); };
$('scrub').oninput = e => { playing = false; $('play').textContent = '▶ Play';
  t = dur(D.flights[cur]) * e.target.value / 1000; draw(); };
window.addEventListener('resize', draw);
matchMedia('(prefers-color-scheme:dark)').addEventListener('change', draw);

table();
load(0);
</script>
"""

if __name__ == "__main__":
    print()
    build()
    print()
