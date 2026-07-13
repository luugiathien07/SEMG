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
    channel_layout: list[list[int | None]],
    display_window_sec: float = 3.0,
    playback_duration_sec: float = 30.0,
    envelope_window_sec: float = 0.05,
    target_display_hz: int = 250,
    model_metrics: list[dict] | None = None,
    best_model: str = "",
) -> str:
    """Build self-contained HTML for smooth client-side EMG animation.

    The full signal is decimated to *target_display_hz*, embedded as JSON,
    and rendered with Canvas2D.  The browser pans a *display_window_sec*
    window across the data using ``requestAnimationFrame`` — no server
    round-trips, no Plotly overhead, typically 60 fps.

    Each dict in `segments_info` must include a `"channelPreds"` key: a
    64-length list (index = physical channel - 1) of 0/1/None, used to color
    the 64-electrode diagram for that segment.
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
        "models": model_metrics or [],
        "bestModel": best_model,
        "channelLayout": channel_layout,
    }

    return _TEMPLATE.replace('"__DATA__"', json.dumps(data))


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<style>
:root{
  --bg:#F8FAFC; --card:#FFFFFF; --border:#E2E8F0;
  --text:#1E293B; --text-soft:#64748B;
  --blue:#348AC9; --gold:#E99D2A; --light-blue:#A8CDE8;
  --green:#16A34A; --green-tint:#DCFCE7;
  --red:#DC2626; --red-tint:#FEE2E2;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:8px 16px 20px}
.title{font-size:23px;font-weight:700;margin:22px 0 10px;color:var(--text)}
.title:first-of-type{margin-top:6px}
.chart-box{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:6px;margin-bottom:6px}
.panel-row{display:flex;gap:6px;align-items:flex-start}
.panel-row .chart-box{margin-bottom:6px}
.wave-box{flex:2;min-width:0}
.chmap-box{flex:1;min-width:180px}
.chmap-title{font-size:13px;font-weight:600;color:var(--text-soft);text-align:center;margin-bottom:4px}
canvas{display:block;width:100%}
.legend{display:flex;gap:16px;font-size:13px;color:var(--text-soft);margin-top:4px;padding-left:50px}
.legend span::before{content:'';display:inline-block;width:14px;height:3px;margin-right:5px;vertical-align:middle;border-radius:1px}
.lg-raw::before{background:var(--blue)}
.lg-proc::before{background:var(--gold)}
.lg-rms::before{background:var(--red)}
.lg-mdf::before{background:var(--blue)}
.lg-ok::before{background:var(--green);width:10px;height:10px;border-radius:2px}
.lg-fat::before{background:var(--red);width:10px;height:10px;border-radius:2px}
.lg-invalid::before{background:#94A3B8;width:10px;height:10px;border-radius:2px}
.controls{display:flex;align-items:center;gap:10px;margin-bottom:6px}
.tog{background:var(--card);color:var(--text-soft);border:1px solid var(--border);border-radius:4px;padding:3px 9px;font-size:13px;cursor:pointer;white-space:nowrap}
.tog.on{background:var(--gold);color:#fff;border-color:var(--gold)}
.btn{background:var(--blue);color:#fff;border:none;border-radius:6px;padding:6px 16px;font-size:15px;cursor:pointer;font-weight:500;white-space:nowrap}
.btn:hover{background:#2c73a8}
.btn.paused{background:#94a3b8}
.btn.paused:hover{background:#7c8ba0}
.track{flex:1;height:6px;background:var(--light-blue);border-radius:3px;cursor:pointer;position:relative}
.fill{height:100%;background:var(--blue);border-radius:3px;pointer-events:none}
.time{font-size:13px;color:var(--text-soft);min-width:110px;text-align:right;font-variant-numeric:tabular-nums}
.spd{display:flex;gap:4px}
.spd button{background:var(--card);color:var(--text-soft);border:1px solid var(--border);border-radius:4px;padding:3px 8px;font-size:13px;cursor:pointer}
.spd button.on{background:var(--gold);color:#fff;border-color:var(--gold)}

.status-row{display:flex;gap:10px;margin-bottom:6px;flex-wrap:wrap}
.status-banner{flex:2;min-width:220px;border-radius:10px;padding:14px 20px;border:2px solid var(--border);display:flex;flex-direction:column;justify-content:center}
.status-banner.ok{background:var(--green-tint);border-color:var(--green)}
.status-banner.bad{background:var(--red-tint);border-color:var(--red)}
.status-banner .sl{font-size:13px;color:var(--text-soft);margin-bottom:2px}
.status-banner .sv{font-size:28px;font-weight:700;line-height:1.1}
.status-banner.ok .sv{color:var(--green)}
.status-banner.bad .sv{color:var(--red)}
.mvc-card{flex:1;min-width:120px;background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 16px;display:flex;flex-direction:column;justify-content:center}
.mvc-card .cl{font-size:13px;color:var(--text-soft);margin-bottom:2px}
.mvc-card .cv{font-size:24px;font-weight:600;color:var(--text)}

.reco{background:var(--red-tint);border:1px solid var(--red);border-radius:6px;padding:7px 11px;font-size:13px;color:var(--red);margin-bottom:6px;display:none}

.section-toggle{display:flex;align-items:center;gap:8px;cursor:pointer;user-select:none}
.arrow{width:0;height:0;border-left:5px solid transparent;border-right:5px solid transparent;border-top:6px solid var(--text-soft);transition:transform .2s}
.arrow.open{transform:rotate(180deg)}
.ptbl{width:100%;border-collapse:collapse;font-size:13px;margin-top:6px}
.ptbl th{text-align:left;padding:6px 10px;border-bottom:1px solid var(--border);color:var(--text-soft);font-weight:500}
.ptbl td{padding:6px 10px;border-bottom:1px solid var(--border)}

.mtbl{width:100%;border-collapse:collapse;font-size:13px;margin-top:6px;margin-bottom:14px}
.mtbl th{text-align:left;padding:7px 10px;border-bottom:1px solid var(--border);color:var(--text-soft);font-weight:500;white-space:nowrap}
.mtbl td{padding:7px 10px;border-bottom:1px solid var(--border);white-space:nowrap}
.mtbl td.mname{font-weight:600;color:var(--text)}
.mtbl tr.best td{background:var(--green-tint);border-bottom-color:var(--green)}
.mtbl tr.best td.mname{color:var(--green);font-weight:700}
.mtbl tr.best td:first-child{border-left:3px solid var(--green)}
.mtbl .best-badge{display:inline-block;margin-left:8px;font-size:11px;font-weight:700;color:#fff;background:var(--green);border-radius:10px;padding:1px 8px;vertical-align:middle}
.cm-wrap{display:flex;gap:16px;flex-wrap:wrap;margin-top:4px}
.cm-block{flex:1;min-width:180px;background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.cm-title{font-size:15px;font-weight:600;color:var(--text);margin-bottom:8px;text-align:center}
.cm-grid{display:grid;grid-template-columns:auto repeat(2,1fr);gap:4px;font-size:14px;text-align:center}
.cm-grid .hd{color:var(--text-soft);display:flex;align-items:center;justify-content:center;font-size:12px;padding:2px}
.cm-grid .cell{padding:16px 4px;border-radius:6px;font-weight:700;font-size:22px}
.cm-grid .cell.tn,.cm-grid .cell.tp{background:var(--green-tint);color:var(--green)}
.cm-grid .cell.fp,.cm-grid .cell.fn{background:var(--red-tint);color:var(--red)}
</style>
</head>
<body>

<div class="title">1. Tín hiệu EMG thời gian thực</div>
<div class="panel-row">
  <div class="chart-box wave-box">
    <canvas id="wave"></canvas>
    <div class="legend">
      <span class="lg-raw">Tín hiệu thô</span>
      <span class="lg-proc">Tín hiệu đã xử lý</span>
    </div>
  </div>
  <div class="chart-box chmap-box">
    <div class="chmap-title">Sơ đồ 64 điện cực — Cơ nhị đầu tay (Biceps brachii)</div>
    <canvas id="chmap"></canvas>
    <div class="legend">
      <span class="lg-ok">Không mỏi</span>
      <span class="lg-fat">Mỏi</span>
      <span class="lg-invalid">Kênh không hợp lệ</span>
    </div>
  </div>
</div>

<div class="controls">
  <button class="btn" id="playBtn" onclick="toggle()">Phát</button>
  <button class="tog on" id="togEnv" onclick="togProc()">Envelope</button>
  <div class="track" id="track" onclick="seek(event)"><div class="fill" id="fill"></div></div>
  <span class="time" id="tm">0.0s / 0.0s</span>
  <div class="spd">
    <button class="on" onclick="spd(1,this)">1×</button>
    <button onclick="spd(2,this)">2×</button>
    <button onclick="spd(5,this)">5×</button>
    <button onclick="spd(10,this)">10×</button>
  </div>
</div>

<div class="status-row">
  <div class="status-banner" id="statusBanner">
    <div class="sl">Trạng thái</div>
    <div class="sv" id="st">—</div>
  </div>
  <div class="mvc-card">
    <div class="cl">Mức độ MVC</div>
    <div class="cv" id="seg">—</div>
  </div>
</div>
<div class="reco" id="reco">Khuyến nghị: giảm cường độ hoặc cho bệnh nhân nghỉ giữa hiệp</div>

<div class="title">2. Xu hướng RMS &amp; MDF</div>
<div class="chart-box">
  <canvas id="trend"></canvas>
  <div class="legend">
    <span class="lg-rms">RMS</span>
    <span class="lg-mdf">MDF (Hz)</span>
    <span class="lg-ok">Normal</span>
    <span class="lg-fat">Fatigue</span>
  </div>
</div>

<div class="title section-toggle" onclick="togSection('detailBody','detailArrow')">
  3. Chi tiết dự đoán theo model (giai đoạn hiện tại)
  <span class="arrow" id="detailArrow"></span>
</div>
<div id="detailBody" style="display:none">
  <table class="ptbl" id="ptbl"><thead><tr><th>Model</th><th>Dự đoán</th><th>P(Fatigue)</th></tr></thead><tbody></tbody></table>
</div>

<div class="title section-toggle" onclick="togSection('metricsBody','metricsArrow')">
  4. Kết quả phân loại của các mô hình (Classification results)
  <span class="arrow" id="metricsArrow"></span>
</div>
<div id="metricsBody" style="display:none">
  <div style="overflow-x:auto">
  <table class="mtbl" id="mtbl">
    <thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>CV-Acc</th><th>AUC</th></tr></thead>
    <tbody></tbody>
  </table>
  </div>
  <div class="cm-wrap" id="cmWrap"></div>
</div>

<script>
const D = "__DATA__";

let playing=false, curT=0, lastTs=null, sf=1, showProc=true;
const baseSpd=D.totalSec/D.playbackDuration;
const maxT=D.totalSec-D.displayWindow;

let wCtx,tCtx,cCtx,wW,wH,tW,tH,cW,cH;

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
  [wCtx,wW,wH]=setupCanvas(document.getElementById('wave'),300);
  [tCtx,tW,tH]=setupCanvas(document.getElementById('trend'),170);
  [cCtx,cW,cH]=setupCanvas(document.getElementById('chmap'),300);
  drawWave(); drawTrend(); drawChannelMap(); updateSeg(); updateProg(); renderMetrics();
}

function drawWave(){
  const ctx=wCtx, w=wW, h=wH;
  const L=50,R=15,T=10,B=28, pW=w-L-R, pH=h-T-B;
  const t0=curT, t1=curT+D.displayWindow;
  const y0=D.yMin, y1=D.yMax;

  ctx.fillStyle='#FFFFFF'; ctx.fillRect(0,0,w,h);

  // grid
  ctx.strokeStyle='rgba(30,41,59,.08)'; ctx.lineWidth=1;
  for(let i=0;i<=4;i++){const y=T+i/4*pH; ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(L+pW,y);ctx.stroke();}
  for(let i=0;i<=6;i++){const x=L+i/6*pW; ctx.beginPath();ctx.moveTo(x,T);ctx.lineTo(x,T+pH);ctx.stroke();}

  // x labels
  ctx.fillStyle='#64748B'; ctx.font='13px sans-serif'; ctx.textAlign='center';
  for(let i=0;i<=6;i++) ctx.fillText((t0+i/6*(t1-t0)).toFixed(1)+'s', L+i/6*pW, h-5);
  // y labels
  ctx.textAlign='right';
  for(let i=0;i<=4;i++){const v=y1-(i/4)*(y1-y0); ctx.fillText(v.toFixed(2), L-4, T+i/4*pH+4);}

  const i0=Math.max(0,Math.floor(t0*D.fs));
  const i1=Math.min(D.raw.length,Math.ceil(t1*D.fs));
  const xOf=i=>L+((i/D.fs-t0)/(t1-t0))*pW;
  const yOf=v=>T+((y1-v)/(y1-y0))*pH;

  // raw
  ctx.strokeStyle='#348AC9'; ctx.lineWidth=1.4;
  ctx.beginPath();
  for(let i=i0;i<i1;i++){const x=xOf(i),y=yOf(D.raw[i]); i===i0?ctx.moveTo(x,y):ctx.lineTo(x,y);}
  ctx.stroke();

  // processed (rectified envelope)
  if(showProc){
    ctx.strokeStyle='#E99D2A'; ctx.lineWidth=2;
    ctx.beginPath();
    for(let i=i0;i<i1;i++){const x=xOf(i),y=yOf(D.proc[i]); i===i0?ctx.moveTo(x,y):ctx.lineTo(x,y);}
    ctx.stroke();
  }

  // segment boundaries
  ctx.strokeStyle='rgba(30,41,59,.25)'; ctx.setLineDash([5,5]); ctx.lineWidth=1;
  for(const s of D.segments){
    if(s.start>t0&&s.start<t1){
      const x=L+((s.start-t0)/(t1-t0))*pW;
      ctx.beginPath();ctx.moveTo(x,T);ctx.lineTo(x,T+pH);ctx.stroke();
      ctx.fillStyle='rgba(30,41,59,.5)'; ctx.textAlign='left'; ctx.font='12px sans-serif';
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

function drawChannelMap(){
  const ctx=cCtx, w=cW, h=cH;
  ctx.fillStyle='#FFFFFF'; ctx.fillRect(0,0,w,h);

  const layout=D.channelLayout;
  const rows=layout.length, cols=layout[0].length;
  const pad=10, gridW=w-2*pad, gridH=h-2*pad;
  const cellW=gridW/cols, cellH=gridH/rows;
  const rad=Math.min(cellW,cellH)*0.28;

  const si=curSeg(), preds=D.segments[si].channelPreds;

  ctx.font='9px sans-serif'; ctx.textAlign='left'; ctx.textBaseline='middle';
  for(let r=0;r<rows;r++){
    for(let c=0;c<cols;c++){
      const ch=layout[r][c];
      if(ch===null) continue;
      const cx=pad+cellW*(c+0.5), cy=pad+cellH*(r+0.5);
      const pred=preds[ch-1];
      let color;
      if(pred===1) color='#DC2626';
      else if(pred===0) color='#16A34A';
      else color='#94A3B8';
      ctx.beginPath(); ctx.arc(cx,cy,rad,0,2*Math.PI);
      ctx.fillStyle=color; ctx.fill();
      ctx.fillStyle='#1E293B';
      ctx.fillText(String(ch), cx+rad+2, cy);
    }
  }
}

function updateSeg(){
  const si=curSeg(), s=D.segments[si];
  document.getElementById('seg').textContent=s.label+(s.isPostFatigue?' (sau mỏi)':'');

  const chosen=s.predictions.find(p=>p.model===D.bestModel)||s.predictions[0];
  const fat=chosen.pred===1;
  const stEl=document.getElementById('st');
  stEl.textContent=fat?'MỎI':'KHÔNG MỎI';
  const banner=document.getElementById('statusBanner');
  banner.className='status-banner '+(fat?'bad':'ok');

  document.getElementById('reco').style.display=fat?'block':'none';

  // prediction table
  const tb=document.querySelector('#ptbl tbody');
  tb.innerHTML=s.predictions.map(p=>'<tr><td>'+p.model+'</td><td>'+(p.pred===1?'Fatigue':'Normal')+'</td><td>'+(p.p_fatigue!=null?(p.p_fatigue*100).toFixed(1)+'%':'—')+'</td></tr>').join('');
}

function drawTrend(){
  const ctx=tCtx, w=tW, h=tH;
  const L=55,R=55,T=22,B=28, pW=w-L-R, pH=h-T-B;
  ctx.fillStyle='#FFFFFF'; ctx.fillRect(0,0,w,h);

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
  ctx.strokeStyle='rgba(30,41,59,.08)'; ctx.lineWidth=1;
  for(let i=0;i<=3;i++){const y=T+i/3*pH; ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(L+pW,y);ctx.stroke();}
  for(let i=0;i<=6;i++){const x=L+i/6*pW; ctx.beginPath();ctx.moveTo(x,T);ctx.lineTo(x,T+pH);ctx.stroke();}

  // x labels
  ctx.fillStyle='#64748B'; ctx.font='11px sans-serif'; ctx.textAlign='center';
  for(let i=0;i<=6;i++){const t=i/6*D.totalSec; ctx.fillText(t.toFixed(0)+'s', L+i/6*pW, h-5);}

  // segment fatigue background shading + boundaries
  for(let si=0;si<D.segments.length;si++){
    const s=D.segments[si];
    const chosen=s.predictions.find(p=>p.model===D.bestModel)||s.predictions[0];
    const fat=chosen.pred===1;
    const x0=xOf(Math.max(0,s.start)), x1=xOf(Math.min(D.totalSec,s.end));
    const clampX1=Math.min(x1, xOf(Math.min(now,D.totalSec)));
    if(clampX1>x0){
      ctx.fillStyle=fat?'rgba(220,38,38,.14)':'rgba(22,163,74,.12)';
      ctx.fillRect(x0,T,clampX1-x0,pH);
    }
    if(s.start>0&&s.start<D.totalSec){
      ctx.strokeStyle='rgba(30,41,59,.15)'; ctx.setLineDash([4,4]); ctx.lineWidth=1;
      const bx=xOf(s.start);
      ctx.beginPath();ctx.moveTo(bx,T);ctx.lineTo(bx,T+pH);ctx.stroke();
      ctx.setLineDash([]);
      if(clampX1>=xOf(s.start)){
        ctx.fillStyle='rgba(30,41,59,.45)'; ctx.font='10px sans-serif'; ctx.textAlign='center';
        ctx.fillText(s.label, xOf((s.start+Math.min(s.end,D.totalSec))/2), T+12);
      }
    }
  }

  // find last visible index
  let nVis=0;
  for(let i=0;i<tT.length;i++){if(tT[i]<=now) nVis=i+1; else break;}
  if(nVis<1) return;

  // RMS line
  ctx.strokeStyle='#DC2626'; ctx.lineWidth=2;
  ctx.beginPath();
  for(let i=0;i<nVis;i++){const x=xOf(tT[i]),y=rY(tR[i]); i?ctx.lineTo(x,y):ctx.moveTo(x,y);}
  ctx.stroke();

  // MDF line
  ctx.strokeStyle='#348AC9'; ctx.lineWidth=2;
  ctx.beginPath();
  for(let i=0;i<nVis;i++){const x=xOf(tT[i]),y=mY(tM[i]); i?ctx.lineTo(x,y):ctx.moveTo(x,y);}
  ctx.stroke();

  // playback cursor
  ctx.strokeStyle='rgba(30,41,59,.4)'; ctx.lineWidth=1;
  const cx=xOf(now);
  ctx.beginPath();ctx.moveTo(cx,T);ctx.lineTo(cx,T+pH);ctx.stroke();

  // axis titles
  ctx.font='12px sans-serif';
  ctx.fillStyle='#DC2626'; ctx.textAlign='right'; ctx.fillText('RMS', L-5, T-6);
  ctx.fillStyle='#348AC9'; ctx.textAlign='left'; ctx.fillText('MDF (Hz)', w-R+5, T-6);
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
    b.textContent='Tạm dừng'; b.classList.add('paused');
    lastTs=null; requestAnimationFrame(frame);
  }else{
    b.textContent='Phát'; b.classList.remove('paused');
  }
}

function spd(s,el){
  sf=s;
  document.querySelectorAll('.spd button').forEach(b=>b.classList.remove('on'));
  el.classList.add('on');
}

function togProc(){
  showProc=!showProc;
  document.getElementById('togEnv').classList.toggle('on',showProc);
  drawWave();
}

function togSection(bodyId,arrowId){
  const body=document.getElementById(bodyId);
  const arrow=document.getElementById(arrowId);
  const open=body.style.display==='none';
  body.style.display=open?'block':'none';
  arrow.classList.toggle('open',open);
}

function pct(v){ return v==null?'—':(v*100).toFixed(1)+'%'; }

function renderMetrics(){
  const tb=document.querySelector('#mtbl tbody');
  // Best classification result = the demo's chosen model, else the highest F1.
  let bestName=D.bestModel;
  if(!bestName || !D.models.some(m=>m.name===bestName))
    bestName=D.models.reduce((a,b)=>b.f1>a.f1?b:a, D.models[0]).name;
  tb.innerHTML=D.models.map(m=>{
    const isBest=m.name===bestName;
    const nameCell=m.name+(isBest?'<span class="best-badge">★ Tốt nhất</span>':'');
    return '<tr'+(isBest?' class="best"':'')+'><td class="mname">'+nameCell+'</td><td>'+pct(m.accuracy)+'</td><td>'+pct(m.precision)+
    '</td><td>'+pct(m.recall)+'</td><td>'+pct(m.f1)+'</td><td>'+pct(m.cvAccuracy)+'</td><td>'+pct(m.auc)+'</td></tr>';
  }).join('');

  const wrap=document.getElementById('cmWrap');
  wrap.innerHTML=D.models.map(m=>{
    const c=m.confusion; // [[TN,FP],[FN,TP]]
    return '<div class="cm-block"><div class="cm-title">'+m.name+'</div>'+
      '<div class="cm-grid">'+
      '<div class="hd"></div><div class="hd">Dự đoán: Normal</div><div class="hd">Dự đoán: Fatigue</div>'+
      '<div class="hd">Thực: Normal</div><div class="cell tn">'+c[0][0]+'</div><div class="cell fp">'+c[0][1]+'</div>'+
      '<div class="hd">Thực: Fatigue</div><div class="cell fn">'+c[1][0]+'</div><div class="cell tp">'+c[1][1]+'</div>'+
      '</div></div>';
  }).join('');
}

function seek(e){
  const r=document.getElementById('track').getBoundingClientRect();
  curT=Math.max(0,Math.min(maxT,((e.clientX-r.left)/r.width)*maxT));
  drawWave(); drawTrend(); drawChannelMap(); updateSeg(); updateProg();
}

function frame(ts){
  if(!playing) return;
  if(lastTs===null) lastTs=ts;
  const dt=(ts-lastTs)/1000;
  lastTs=ts;
  curT=Math.min(curT+dt*baseSpd*sf, maxT);

  drawWave(); drawTrend(); drawChannelMap(); updateSeg(); updateProg();

  if(curT>=maxT){
    playing=false;
    const b=document.getElementById('playBtn');
    b.textContent='Phát lại'; b.classList.remove('paused');
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
