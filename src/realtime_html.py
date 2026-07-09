"""Client-side HTML/JS component for smooth real-time EMG waveform
animation. Rendered via st.components.v1.html — all animation runs in
the browser (Canvas2D + requestAnimationFrame), zero Streamlit
round-trips per frame."""
from __future__ import annotations

import json

import numpy as np
from scipy.signal import welch

from .realtime_session import rectify_envelope


def _compute_trend(
    signal: np.ndarray, fs: int,
    step_sec: float = 0.5, window_sec: float = 1.0,
) -> tuple[list[float], list[float], list[float]]:
    """Sliding-window RMS and MDF across the full signal."""
    step = int(step_sec * fs)
    win = int(window_sec * fs)
    times, rms_vals, mdf_vals = [], [], []
    for pos in range(0, len(signal) - win + 1, step):
        chunk = signal[pos : pos + win]
        times.append(round((pos + win / 2) / fs, 2))
        rms_vals.append(round(float(np.sqrt(np.mean(chunk**2))), 4))
        freqs, psd = welch(chunk, fs=fs, nperseg=min(256, win))
        cumsum = np.cumsum(psd)
        idx = np.searchsorted(cumsum, cumsum[-1] / 2) if cumsum[-1] > 0 else 0
        mdf_vals.append(round(float(freqs[min(idx, len(freqs) - 1)]), 2))
    return times, rms_vals, mdf_vals


def build_realtime_html(
    signal: np.ndarray,
    segments_info: list[dict],
    fs: int,
    display_window_sec: float = 3.0,
    playback_duration_sec: float = 30.0,
    envelope_window_sec: float = 0.05,
    target_display_hz: int = 250,
) -> str:
    """Build self-contained HTML for smooth client-side EMG animation.

    The full signal is decimated to *target_display_hz*, embedded as JSON,
    and rendered with Canvas2D.  The browser pans a *display_window_sec*
    window across the data using ``requestAnimationFrame`` — no server
    round-trips, no Plotly overhead, typically 60 fps.
    """
    env_samples = max(1, int(envelope_window_sec * fs))
    processed = rectify_envelope(signal, env_samples)

    dec = max(1, fs // target_display_hz)
    raw_list = [round(float(x), 3) for x in signal[::dec]]
    proc_list = [round(float(x), 3) for x in processed[::dec]]

    p1, p99 = float(np.percentile(signal, 1)), float(np.percentile(signal, 99))
    margin = 0.1 * (p99 - p1)

    trend_t, trend_rms, trend_mdf = _compute_trend(signal, fs)

    data = {
        "raw": raw_list,
        "proc": proc_list,
        "fs": round(fs / dec, 2),
        "yMin": round(p1 - margin, 4),
        "yMax": round(p99 + margin, 4),
        "totalSec": round(len(signal) / fs, 3),
        "displayWindow": display_window_sec,
        "playbackDuration": playback_duration_sec,
        "segments": segments_info,
        "trendT": trend_t,
        "trendRms": trend_rms,
        "trendMdf": trend_mdf,
    }

    return _TEMPLATE.replace('"__DATA__"', json.dumps(data))


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:12px 16px;overflow-x:hidden}
.title{font-size:15px;font-weight:600;margin-bottom:6px;color:#f0f6fc}
.chart-box{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:8px;margin-bottom:10px}
canvas{display:block;width:100%}
.legend{display:flex;gap:16px;font-size:11px;color:#8b949e;margin-top:4px;padding-left:50px}
.legend span::before{content:'';display:inline-block;width:14px;height:3px;margin-right:5px;vertical-align:middle;border-radius:1px}
.lg-raw::before{background:rgba(91,155,213,.6)}
.lg-proc::before{background:#F39C12}
.lg-rms::before{background:#E74C3C}
.lg-mdf::before{background:#3498DB}
.lg-ok::before{background:rgba(63,185,80,.35);width:10px;height:10px;border-radius:2px}
.lg-fat::before{background:rgba(248,81,73,.35);width:10px;height:10px;border-radius:2px}
.controls{display:flex;align-items:center;gap:10px;margin-bottom:10px}
.btn{background:#238636;color:#fff;border:none;border-radius:6px;padding:5px 14px;font-size:13px;cursor:pointer;font-weight:500;white-space:nowrap}
.btn:hover{background:#2ea043}
.btn.paused{background:#da3633}
.btn.paused:hover{background:#b62324}
.track{flex:1;height:6px;background:#30363d;border-radius:3px;cursor:pointer;position:relative}
.fill{height:100%;background:#58a6ff;border-radius:3px;pointer-events:none}
.time{font-size:12px;color:#8b949e;min-width:110px;text-align:right;font-variant-numeric:tabular-nums}
.spd{display:flex;gap:4px}
.spd button{background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:2px 7px;font-size:11px;cursor:pointer}
.spd button.on{background:#388bfd;color:#fff;border-color:#388bfd}
.cards{display:flex;gap:10px;margin-bottom:10px;flex-wrap:wrap}
.card{flex:1;min-width:120px;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px 14px}
.card .cl{font-size:11px;color:#8b949e;margin-bottom:2px}
.card .cv{font-size:18px;font-weight:600}
.ok{color:#3fb950}.bad{color:#f85149}
.reco{background:#f8514922;border:1px solid #f8514944;border-radius:6px;padding:8px 12px;font-size:13px;color:#f85149;margin-bottom:10px;display:none}
details{margin-top:6px}
summary{cursor:pointer;color:#8b949e;font-size:12px;user-select:none}
.ptbl{width:100%;border-collapse:collapse;font-size:12px;margin-top:6px}
.ptbl th{text-align:left;padding:5px 10px;border-bottom:1px solid #30363d;color:#8b949e;font-weight:500}
.ptbl td{padding:5px 10px;border-bottom:1px solid #21262d}
</style>
</head>
<body>

<div class="title">① Tín hiệu EMG — Near-Real-Time</div>
<div class="chart-box">
  <canvas id="wave"></canvas>
  <div class="legend">
    <span class="lg-raw">Tín hiệu thô</span>
    <span class="lg-proc">Rectified envelope</span>
  </div>
</div>

<div class="controls">
  <button class="btn" id="playBtn" onclick="toggle()">▶ Phát</button>
  <div class="track" id="track" onclick="seek(event)"><div class="fill" id="fill"></div></div>
  <span class="time" id="tm">0.0s / 0.0s</span>
  <div class="spd">
    <button class="on" onclick="spd(1,this)">1×</button>
    <button onclick="spd(2,this)">2×</button>
    <button onclick="spd(5,this)">5×</button>
    <button onclick="spd(10,this)">10×</button>
  </div>
</div>

<div class="cards">
  <div class="card"><div class="cl">Mức độ MVC</div><div class="cv" id="seg">—</div></div>
  <div class="card"><div class="cl">Trạng thái</div><div class="cv" id="st">—</div></div>
</div>
<div class="reco" id="reco">⚠ Khuyến nghị: giảm cường độ hoặc cho bệnh nhân nghỉ giữa hiệp</div>

<div class="title">② Xu hướng RMS & MDF</div>
<div class="chart-box">
  <canvas id="trend"></canvas>
  <div class="legend">
    <span class="lg-rms">RMS</span>
    <span class="lg-mdf">MDF (Hz)</span>
    <span class="lg-ok">Normal</span>
    <span class="lg-fat">Fatigue</span>
  </div>
</div>

<details>
  <summary>③ Chi tiết dự đoán theo model (giai đoạn hiện tại)</summary>
  <table class="ptbl" id="ptbl"><thead><tr><th>Model</th><th>Dự đoán</th><th>P(Fatigue)</th></tr></thead><tbody></tbody></table>
</details>

<script>
const D = "__DATA__";

let playing=false, curT=0, lastTs=null, sf=1;
const baseSpd=D.totalSec/D.playbackDuration;
const maxT=D.totalSec-D.displayWindow;

let wCtx,tCtx,wW,wH,tW,tH;

function setupCanvas(c,h){
  const dpr=devicePixelRatio||1;
  const w=c.parentElement.clientWidth-16;
  c.width=w*dpr; c.height=h*dpr;
  c.style.width=w+'px'; c.style.height=h+'px';
  const ctx=c.getContext('2d');
  ctx.scale(dpr,dpr);
  return [ctx,w,h];
}

function init(){
  [wCtx,wW,wH]=setupCanvas(document.getElementById('wave'),240);
  [tCtx,tW,tH]=setupCanvas(document.getElementById('trend'),170);
  drawWave(); drawTrend(); updateSeg(); updateProg();
}

function drawWave(){
  const ctx=wCtx, w=wW, h=wH;
  const L=50,R=15,T=10,B=28, pW=w-L-R, pH=h-T-B;
  const t0=curT, t1=curT+D.displayWindow;
  const y0=D.yMin, y1=D.yMax;

  ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,w,h);

  // grid
  ctx.strokeStyle='rgba(255,255,255,.07)'; ctx.lineWidth=1;
  for(let i=0;i<=4;i++){const y=T+i/4*pH; ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(L+pW,y);ctx.stroke();}
  for(let i=0;i<=6;i++){const x=L+i/6*pW; ctx.beginPath();ctx.moveTo(x,T);ctx.lineTo(x,T+pH);ctx.stroke();}

  // x labels
  ctx.fillStyle='#8b949e'; ctx.font='11px sans-serif'; ctx.textAlign='center';
  for(let i=0;i<=6;i++) ctx.fillText((t0+i/6*(t1-t0)).toFixed(1)+'s', L+i/6*pW, h-5);
  // y labels
  ctx.textAlign='right';
  for(let i=0;i<=4;i++){const v=y1-(i/4)*(y1-y0); ctx.fillText(v.toFixed(2), L-4, T+i/4*pH+4);}

  const i0=Math.max(0,Math.floor(t0*D.fs));
  const i1=Math.min(D.raw.length,Math.ceil(t1*D.fs));
  const xOf=i=>L+((i/D.fs-t0)/(t1-t0))*pW;
  const yOf=v=>T+((y1-v)/(y1-y0))*pH;

  // raw
  ctx.strokeStyle='rgba(91,155,213,.5)'; ctx.lineWidth=1;
  ctx.beginPath();
  for(let i=i0;i<i1;i++){const x=xOf(i),y=yOf(D.raw[i]); i===i0?ctx.moveTo(x,y):ctx.lineTo(x,y);}
  ctx.stroke();

  // processed
  ctx.strokeStyle='#F39C12'; ctx.lineWidth=2;
  ctx.beginPath();
  for(let i=i0;i<i1;i++){const x=xOf(i),y=yOf(D.proc[i]); i===i0?ctx.moveTo(x,y):ctx.lineTo(x,y);}
  ctx.stroke();

  // segment boundaries
  ctx.strokeStyle='rgba(255,255,255,.25)'; ctx.setLineDash([5,5]); ctx.lineWidth=1;
  for(const s of D.segments){
    if(s.start>t0&&s.start<t1){
      const x=L+((s.start-t0)/(t1-t0))*pW;
      ctx.beginPath();ctx.moveTo(x,T);ctx.lineTo(x,T+pH);ctx.stroke();
      ctx.fillStyle='rgba(255,255,255,.4)'; ctx.textAlign='left'; ctx.font='11px sans-serif';
      ctx.fillText(s.label+(s.isPostFatigue?' (sau mỏi)':''), x+3, T+14);
    }
  }
  ctx.setLineDash([]);
}

function curSeg(){
  const mid=curT+D.displayWindow/2;
  for(let i=D.segments.length-1;i>=0;i--) if(mid>=D.segments[i].start) return i;
  return 0;
}

function updateSeg(){
  const si=curSeg(), s=D.segments[si];
  document.getElementById('seg').textContent=s.label+(s.isPostFatigue?' (sau mỏi)':'');

  const knn=s.predictions.find(p=>p.model==='KNN')||s.predictions[0];
  const fat=knn.pred===1;
  const stEl=document.getElementById('st');
  stEl.textContent=fat?'⚠️ Mỏi':'✅ Không mỏi';
  stEl.className='cv '+(fat?'bad':'ok');

  document.getElementById('reco').style.display=fat?'block':'none';

  // prediction table
  const tb=document.querySelector('#ptbl tbody');
  tb.innerHTML=s.predictions.map(p=>'<tr><td>'+p.model+'</td><td>'+(p.pred===1?'Fatigue':'Normal')+'</td><td>'+(p.p_fatigue!=null?(p.p_fatigue*100).toFixed(1)+'%':'—')+'</td></tr>').join('');
}

function drawTrend(){
  const ctx=tCtx, w=tW, h=tH;
  const L=55,R=55,T=22,B=28, pW=w-L-R, pH=h-T-B;
  ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,w,h);

  const tT=D.trendT, tR=D.trendRms, tM=D.trendMdf;
  if(!tT.length) return;
  const now=curT+D.displayWindow/2;

  // y ranges (fixed from full data)
  const rMin=Math.min(...tR)*.9, rMax=Math.max(...tR)*1.1;
  const mMin=Math.min(...tM)*.9, mMax=Math.max(...tM)*1.1;
  const xOf=t=>L+(t/D.totalSec)*pW;
  const rY=v=>T+((rMax-v)/(rMax-rMin))*pH;
  const mY=v=>T+((mMax-v)/(mMax-mMin))*pH;

  // grid
  ctx.strokeStyle='rgba(255,255,255,.07)'; ctx.lineWidth=1;
  for(let i=0;i<=3;i++){const y=T+i/3*pH; ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(L+pW,y);ctx.stroke();}
  for(let i=0;i<=6;i++){const x=L+i/6*pW; ctx.beginPath();ctx.moveTo(x,T);ctx.lineTo(x,T+pH);ctx.stroke();}

  // x labels
  ctx.fillStyle='#8b949e'; ctx.font='10px sans-serif'; ctx.textAlign='center';
  for(let i=0;i<=6;i++){const t=i/6*D.totalSec; ctx.fillText(t.toFixed(0)+'s', L+i/6*pW, h-5);}

  // segment fatigue background shading + boundaries
  for(let si=0;si<D.segments.length;si++){
    const s=D.segments[si];
    const knn=s.predictions.find(p=>p.model==='KNN')||s.predictions[0];
    const fat=knn.pred===1;
    const x0=xOf(Math.max(0,s.start)), x1=xOf(Math.min(D.totalSec,s.end));
    const clampX1=Math.min(x1, xOf(Math.min(now,D.totalSec)));
    if(clampX1>x0){
      ctx.fillStyle=fat?'rgba(248,81,73,.12)':'rgba(63,185,80,.07)';
      ctx.fillRect(x0,T,clampX1-x0,pH);
    }
    if(s.start>0&&s.start<D.totalSec){
      ctx.strokeStyle='rgba(255,255,255,.15)'; ctx.setLineDash([4,4]); ctx.lineWidth=1;
      const bx=xOf(s.start);
      ctx.beginPath();ctx.moveTo(bx,T);ctx.lineTo(bx,T+pH);ctx.stroke();
      ctx.setLineDash([]);
      if(clampX1>=xOf(s.start)){
        ctx.fillStyle='rgba(255,255,255,.35)'; ctx.font='9px sans-serif'; ctx.textAlign='center';
        ctx.fillText(s.label, xOf((s.start+Math.min(s.end,D.totalSec))/2), T+12);
      }
    }
  }

  // find last visible index
  let nVis=0;
  for(let i=0;i<tT.length;i++){if(tT[i]<=now) nVis=i+1; else break;}
  if(nVis<1) return;

  // RMS line
  ctx.strokeStyle='#E74C3C'; ctx.lineWidth=2;
  ctx.beginPath();
  for(let i=0;i<nVis;i++){const x=xOf(tT[i]),y=rY(tR[i]); i?ctx.lineTo(x,y):ctx.moveTo(x,y);}
  ctx.stroke();

  // MDF line
  ctx.strokeStyle='#3498DB'; ctx.lineWidth=2;
  ctx.beginPath();
  for(let i=0;i<nVis;i++){const x=xOf(tT[i]),y=mY(tM[i]); i?ctx.lineTo(x,y):ctx.moveTo(x,y);}
  ctx.stroke();

  // playback cursor
  ctx.strokeStyle='rgba(255,255,255,.4)'; ctx.lineWidth=1;
  const cx=xOf(now);
  ctx.beginPath();ctx.moveTo(cx,T);ctx.lineTo(cx,T+pH);ctx.stroke();

  // axis titles
  ctx.font='11px sans-serif';
  ctx.fillStyle='#E74C3C'; ctx.textAlign='right'; ctx.fillText('RMS', L-5, T-6);
  ctx.fillStyle='#3498DB'; ctx.textAlign='left'; ctx.fillText('MDF (Hz)', w-R+5, T-6);
}

function updateProg(){
  const pct=maxT>0?(curT/maxT)*100:0;
  document.getElementById('fill').style.width=Math.min(100,pct)+'%';
  const el=(curT+D.displayWindow/2);
  document.getElementById('tm').textContent=el.toFixed(1)+'s / '+D.totalSec.toFixed(1)+'s';
}

function toggle(){
  playing=!playing;
  const b=document.getElementById('playBtn');
  if(playing){
    if(curT>=maxT) curT=0;
    b.textContent='⏸ Tạm dừng'; b.classList.add('paused');
    lastTs=null; requestAnimationFrame(frame);
  }else{
    b.textContent='▶ Phát'; b.classList.remove('paused');
  }
}

function spd(s,el){
  sf=s;
  document.querySelectorAll('.spd button').forEach(b=>b.classList.remove('on'));
  el.classList.add('on');
}

function seek(e){
  const r=document.getElementById('track').getBoundingClientRect();
  curT=Math.max(0,Math.min(maxT,((e.clientX-r.left)/r.width)*maxT));
  drawWave(); drawTrend(); updateSeg(); updateProg();
}

function frame(ts){
  if(!playing) return;
  if(lastTs===null) lastTs=ts;
  const dt=(ts-lastTs)/1000;
  lastTs=ts;
  curT=Math.min(curT+dt*baseSpd*sf, maxT);

  drawWave(); drawTrend(); updateSeg(); updateProg();

  if(curT>=maxT){
    playing=false;
    const b=document.getElementById('playBtn');
    b.textContent='⟲ Phát lại'; b.classList.remove('paused');
  }else{
    requestAnimationFrame(frame);
  }
}

window.addEventListener('load',()=>{init(); setTimeout(toggle,400);});
window.addEventListener('resize',()=>{init(); drawTrend();});
</script>
</body>
</html>
"""
