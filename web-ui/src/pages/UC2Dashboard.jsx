import { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ReferenceLine, ResponsiveContainer, Label,
} from 'recharts'
import MetricCard from '../components/MetricCard'
import { useApi } from '../hooks/useApi'
import './UC2Dashboard.css'

/* ── Color tokens (match Streamlit version) ── */
const TEAL = '#1B7A7D'
const TEAL_LIGHT = 'rgba(27,122,125,0.25)'
const CORAL = '#E8604C'
const GRAY = '#999999'
const GREEN = '#27AE60'

/* ── Tooltip style ── */
const tooltipStyle = {
  backgroundColor: '#FFFFFF',
  border: '1px solid #E2E8F0',
  borderRadius: 8,
  fontSize: 13,
  fontFamily: 'Inter, sans-serif',
  color: '#1A202C',
  boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
}

/* ══════════════════════════════════════════════════════════════════
   MAIN COMPONENT
   ══════════════════════════════════════════════════════════════════ */
export default function UC2Dashboard() {
  const { data: endData, loading: endLoading } = useApi('/uc2/endurance')
  const { data: patientsData } = useApi('/uc2/patients')

  if (endLoading || !endData) {
    return (
      <div className="uc2-loading">
        <div className="uc2-loading__spinner" />
        <p>Đang mô phỏng và huấn luyện KNN…</p>
      </div>
    )
  }

  const { sessions, f1 } = endData

  return (
    <div className="uc2">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <PanelDashboard sessions={sessions} f1={f1} />
        <div className="uc2__divider" />
        <PanelFatigueInSession sessions={sessions} f1={f1} />
        <div className="uc2__divider" />
        <PanelEnduranceTrend sessions={sessions} />
        <div className="uc2__divider" />
        <PanelRecoveryMetrics patients={patientsData?.patients || []} />
        <div className="uc2__divider" />
        <PanelSignalExplorer patients={patientsData?.patients || []} />
      </motion.div>
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════════
   PANEL 0 — DASHBOARD
   ══════════════════════════════════════════════════════════════════ */
function PanelDashboard({ sessions, f1 }) {
  const first = sessions[0]
  const last = sessions[sessions.length - 1]
  const n = sessions.length
  const improvement = Math.round((last.endurance_sec / first.endurance_sec - 1) * 100)
  const [showExplainer, setShowExplainer] = useState(false)

  return (
    <section className="uc2__section">
      <h2 className="uc2__section-title">📊 Tổng quan phục hồi</h2>
      <p className="uc2__disclaimer">
        Dữ liệu mô phỏng cho demo — không phải bệnh nhân thật. Số thật cần thu tại Motion Lab.
      </p>

      <div className="uc2__metrics-grid">
        <MetricCard icon="📅" label="Số buổi tập" value={n} colorClass="teal" />
        <MetricCard icon="🕐" label="Sức bền ban đầu" value={`${first.endurance_sec.toFixed(0)}s`} colorClass="coral" />
        <MetricCard
          icon="💪" label="Sức bền gần nhất"
          value={`${last.endurance_sec.toFixed(0)}s`}
          delta={`+${improvement}%`}
          colorClass="green"
        />
        <MetricCard icon="🎯" label="F1 (KNN phân loại mỏi)" value={f1.toFixed(3)} colorClass="purple" />
      </div>

      <div className="uc2__info-box">
        <span className="uc2__info-icon">💪</span>
        <p>
          Sau <strong>{n} buổi</strong> trị liệu, sức bền cơ tăng từ{' '}
          <strong>{first.endurance_sec.toFixed(0)}s → {last.endurance_sec.toFixed(0)}s</strong>{' '}
          (+{improvement}%). Cơ mỏi muộn hơn = phục hồi tốt hơn.
        </p>
      </div>

      <button
        className="uc2__explainer-toggle"
        onClick={() => setShowExplainer(!showExplainer)}
      >
        💡 <strong>P(mỏi) là gì?</strong>
        <span className="uc2__chevron">{showExplainer ? '▲' : '▼'}</span>
      </button>

      {showExplainer && (
        <motion.div
          className="uc2__explainer glass-card"
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          transition={{ duration: 0.3 }}
        >
          <p>1. Cắt tín hiệu mỗi buổi thành các <strong>đoạn 1 giây</strong>. Mỗi đoạn trích 7 đặc trưng
          (RMS, MAV, WL, ZC, SSC, MDF, MNF) → thành 1 điểm trong không gian đặc trưng.</p>
          <p>2. Trong dữ liệu huấn luyện, mỗi đoạn đã có nhãn: <strong>mỏi (1)</strong> hoặc <strong>chưa mỏi (0)</strong>.</p>
          <p>3. Với 1 đoạn mới: model tìm <strong>7 đoạn GIỐNG NHẤT</strong> (khoảng cách gần nhất),
          rồi đếm bao nhiêu đoạn là "mỏi".</p>
          <blockquote>
            <strong>P(mỏi) = số hàng xóm "mỏi" / 7.</strong> Ví dụ 5/7 là mỏi → P = 0,71.
          </blockquote>
          <p><strong>Trực giác:</strong> Đầu buổi cơ khoẻ (RMS thấp, MDF cao) → giống đám "chưa mỏi" → P thấp.
          Cuối buổi cơ mỏi (RMS cao, MDF thấp) → giống đám "mỏi" → P cao. Giây P vượt <strong>0,5</strong> = lúc bắt đầu mỏi.</p>
        </motion.div>
      )}
    </section>
  )
}

/* ══════════════════════════════════════════════════════════════════
   PANEL 1 — P(MỎI) IN-SESSION
   ══════════════════════════════════════════════════════════════════ */
function PanelFatigueInSession({ sessions, f1 }) {
  const [idxA, setIdxA] = useState(0)
  const [idxB, setIdxB] = useState(sessions.length - 1)

  const sa = sessions[idxA]
  const sb = sessions[idxB]

  const chartData = useMemo(() => {
    const maxLen = Math.max(sa.t_centers.length, sb.t_centers.length)
    const rows = []
    for (let i = 0; i < maxLen; i++) {
      rows.push({
        time: sa.t_centers[i] ?? sb.t_centers[i] ?? i,
        probaA: sa.proba[i] ?? null,
        probaB: sb.proba[i] ?? null,
      })
    }
    return rows
  }, [sa, sb])

  return (
    <section className="uc2__section">
      <h2 className="uc2__section-title">📈 Xác suất mỏi (KNN) trong bài co cơ duy trì</h2>
      <p className="uc2__caption">
        So sánh đường P(mỏi) giữa 2 buổi. Buổi đầu mỏi sớm (đường lên nhanh),
        buổi sau mỏi muộn (đường lên chậm) → chứng tỏ phục hồi.
      </p>

      <div className="uc2__select-row">
        <label className="uc2__select-label">
          Buổi A
          <select value={idxA} onChange={e => setIdxA(+e.target.value)} className="uc2__select">
            {sessions.map((s, i) => <option key={i} value={i}>Buổi {s.session}</option>)}
          </select>
        </label>
        <label className="uc2__select-label">
          Buổi B
          <select value={idxB} onChange={e => setIdxB(+e.target.value)} className="uc2__select">
            {sessions.map((s, i) => <option key={i} value={i}>Buổi {s.session}</option>)}
          </select>
        </label>
      </div>

      <ResponsiveContainer width="100%" height={420}>
        <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.08)" />
          <XAxis dataKey="time" stroke="#5A6B80" fontSize={12} tickFormatter={v => `${v.toFixed(0)}s`}>
            <Label value="Thời gian trong buổi (s)" position="bottom" offset={0} fill="#5A6B80" fontSize={12} />
          </XAxis>
          <YAxis domain={[0, 1.05]} ticks={[0, 0.25, 0.5, 0.75, 1.0]} stroke="#5A6B80" fontSize={12}>
            <Label value="P(mỏi)" angle={-90} position="insideLeft" fill="#5A6B80" fontSize={12} />
          </YAxis>
          <Tooltip contentStyle={tooltipStyle} formatter={(v, name) => [v?.toFixed(3), name]} />
          <Legend verticalAlign="top" height={36} />
          <ReferenceLine y={0.5} stroke={GRAY} strokeDasharray="6 4" label={{ value: 'Ngưỡng mỏi (P=0.5)', fill: GRAY, fontSize: 11, position: 'insideTopLeft' }} />
          <ReferenceLine x={sa.endurance_sec} stroke={CORAL} strokeDasharray="3 3" label={{ value: `${sa.endurance_sec.toFixed(1)}s`, fill: CORAL, fontSize: 13 }} />
          <ReferenceLine x={sb.endurance_sec} stroke={TEAL} strokeDasharray="3 3" label={{ value: `${sb.endurance_sec.toFixed(1)}s`, fill: TEAL, fontSize: 13 }} />
          <Line
            type="monotone" dataKey="probaA" stroke={CORAL} strokeWidth={2.5}
            dot={{ r: 3, fill: CORAL }} name={`Buổi ${sa.session} (mỏi sớm)`}
            connectNulls
          />
          <Line
            type="monotone" dataKey="probaB" stroke={TEAL} strokeWidth={2.5}
            dot={{ r: 3, fill: TEAL }} name={`Buổi ${sb.session} (mỏi muộn)`}
            connectNulls
          />
        </LineChart>
      </ResponsiveContainer>

      <p className="uc2__caption" style={{ marginTop: 8 }}>
        KNN phân loại mỏi F1 = {f1.toFixed(3)} (mô phỏng). Điểm sức bền = thời điểm P(mỏi) vượt 0.5.
      </p>
    </section>
  )
}

/* ══════════════════════════════════════════════════════════════════
   PANEL 2 — ENDURANCE TREND
   ══════════════════════════════════════════════════════════════════ */
function PanelEnduranceTrend({ sessions }) {
  const chartData = sessions.map(s => ({
    session: s.session,
    endurance: s.endurance_sec,
    pctNon: s.pct_nonfatigue,
    label: `${s.endurance_sec.toFixed(0)}s`,
  }))

  return (
    <section className="uc2__section">
      <h2 className="uc2__section-title">🏋️ Điểm sức bền cơ (thời điểm khởi phát mỏi) qua các buổi</h2>
      <p className="uc2__caption">
        Phục hồi tốt hơn → mỏi đến muộn hơn → điểm sức bền tăng. Đây là chỉ số cốt lõi của Usecase 2.
      </p>

      <ResponsiveContainer width="100%" height={420}>
        <AreaChart data={chartData} margin={{ top: 30, right: 30, left: 20, bottom: 20 }}>
          <defs>
            <linearGradient id="tealGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={TEAL} stopOpacity={0.3} />
              <stop offset="95%" stopColor={TEAL} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.08)" />
          <XAxis dataKey="session" stroke="#5A6B80" fontSize={12}>
            <Label value="Buổi trị liệu" position="bottom" offset={0} fill="#5A6B80" fontSize={12} />
          </XAxis>
          <YAxis stroke="#5A6B80" fontSize={12}>
            <Label value="Thời gian tới khi mỏi (s)" angle={-90} position="insideLeft" fill="#5A6B80" fontSize={12} />
          </YAxis>
          <Tooltip
            contentStyle={tooltipStyle}
            formatter={(v) => [`${v.toFixed(1)}s`, 'Sức bền']}
            labelFormatter={(v) => `Buổi ${v}`}
          />
          <Area
            type="monotone" dataKey="endurance" stroke={TEAL} strokeWidth={3}
            fill="url(#tealGrad)" dot={{ r: 6, fill: 'white', stroke: TEAL, strokeWidth: 2.5 }}
            label={{ position: 'top', fill: TEAL, fontSize: 13, fontWeight: 600, formatter: v => `${v.toFixed(0)}s` }}
          />
        </AreaChart>
      </ResponsiveContainer>

      {/* Data table */}
      <div className="uc2__table-wrap">
        <table className="uc2__table">
          <thead>
            <tr>
              <th>Buổi</th>
              <th>Điểm sức bền (s)</th>
              <th>% cửa sổ chưa mỏi</th>
            </tr>
          </thead>
          <tbody>
            {sessions.map(s => (
              <tr key={s.session}>
                <td>{s.session}</td>
                <td>{s.endurance_sec.toFixed(1)}</td>
                <td>{s.pct_nonfatigue.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

/* ══════════════════════════════════════════════════════════════════
   PANEL 3 — RECOVERY METRICS
   ══════════════════════════════════════════════════════════════════ */
function PanelRecoveryMetrics({ patients }) {
  const [patient, setPatient] = useState(patients[0] || '')
  const apiPath = patient ? `/uc2/recovery/${patient}` : null
  const { data: recData, loading } = useApi(apiPath)

  if (!patients.length) {
    return (
      <section className="uc2__section">
        <h2 className="uc2__section-title">📉 Chỉ số phục hồi (dữ liệu PhysioMio)</h2>
        <div className="uc2__info-box">
          <p>Không tìm thấy dữ liệu PhysioMio. Chạy script mô phỏng trước.</p>
        </div>
      </section>
    )
  }

  const rows = recData?.rows || []
  const baselineMdf = recData?.baseline_mdf

  return (
    <section className="uc2__section">
      <h2 className="uc2__section-title">📉 Chỉ số phục hồi (dữ liệu PhysioMio)</h2>
      <p className="uc2__caption">
        RMS tăng = cơ huy động tốt hơn. MDF dịch lên = dẫn truyền thần kinh cải thiện.
        Symmetry Index = RMS tay liệt / RMS tay lành × 100%.
      </p>

      <div className="uc2__select-row">
        <label className="uc2__select-label">
          Bệnh nhân
          <select value={patient} onChange={e => setPatient(e.target.value)} className="uc2__select">
            {patients.map(p => <option key={p} value={p}>{p}</option>)}
          </select>
        </label>
      </div>

      {loading ? (
        <div className="uc2-loading"><div className="uc2-loading__spinner" /><p>Đang tính RMS/MDF/Symmetry…</p></div>
      ) : (
        <>
          <div className="uc2__charts-row">
            {/* Symmetry */}
            <div className="uc2__chart-half">
              <ResponsiveContainer width="100%" height={340}>
                <AreaChart data={rows} margin={{ top: 20, right: 20, left: 10, bottom: 20 }}>
                  <defs>
                    <linearGradient id="symGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={TEAL} stopOpacity={0.25} />
                      <stop offset="95%" stopColor={TEAL} stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.08)" />
                  <XAxis dataKey="Buổi" stroke="#5A6B80" fontSize={12} />
                  <YAxis domain={[0, 110]} stroke="#5A6B80" fontSize={12}>
                    <Label value="Symmetry (%)" angle={-90} position="insideLeft" fill="#5A6B80" fontSize={11} />
                  </YAxis>
                  <Tooltip contentStyle={tooltipStyle} formatter={v => [`${v?.toFixed(1)}%`, 'Symmetry']} labelFormatter={v => `Buổi ${v}`} />
                  <ReferenceLine y={100} stroke={GRAY} strokeDasharray="6 4" label={{ value: '100% (tay lành)', fill: GRAY, fontSize: 10, position: 'insideTopLeft' }} />
                  <Area type="monotone" dataKey="Symmetry (%)" stroke={TEAL} strokeWidth={3} fill="url(#symGrad)"
                    dot={{ r: 5, fill: 'white', stroke: TEAL, strokeWidth: 2.5 }} />
                </AreaChart>
              </ResponsiveContainer>
              <p className="uc2__chart-title">Chỉ số phục hồi (đối xứng RMS lành-liệt, %)</p>
            </div>

            {/* MDF */}
            <div className="uc2__chart-half">
              <ResponsiveContainer width="100%" height={340}>
                <LineChart data={rows} margin={{ top: 20, right: 20, left: 10, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.08)" />
                  <XAxis dataKey="Buổi" stroke="#5A6B80" fontSize={12} />
                  <YAxis stroke="#5A6B80" fontSize={12}>
                    <Label value="MDF (Hz)" angle={-90} position="insideLeft" fill="#5A6B80" fontSize={11} />
                  </YAxis>
                  <Tooltip contentStyle={tooltipStyle} formatter={v => [`${v?.toFixed(1)} Hz`, 'MDF']} labelFormatter={v => `Buổi ${v}`} />
                  {baselineMdf && (
                    <ReferenceLine y={baselineMdf} stroke={GREEN} strokeDasharray="6 4"
                      label={{ value: 'baseline tay lành', fill: GREEN, fontSize: 10, position: 'insideTopLeft' }} />
                  )}
                  <Line type="monotone" dataKey="MDF" stroke={CORAL} strokeWidth={3}
                    dot={{ r: 5, fill: 'white', stroke: CORAL, strokeWidth: 2.5, symbol: 'square' }} />
                </LineChart>
              </ResponsiveContainer>
              <p className="uc2__chart-title">MDF (Hz) — dịch lên khi phục hồi</p>
            </div>
          </div>

          {/* Data table */}
          <div className="uc2__table-wrap">
            <table className="uc2__table">
              <thead>
                <tr><th>Buổi</th><th>RMS</th><th>MDF</th><th>Symmetry (%)</th></tr>
              </thead>
              <tbody>
                {rows.map(r => (
                  <tr key={r['Buổi']}>
                    <td>{r['Buổi']}</td>
                    <td>{r.RMS?.toFixed(2)}</td>
                    <td>{r.MDF?.toFixed(1)}</td>
                    <td>{r['Symmetry (%)']?.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  )
}

/* ══════════════════════════════════════════════════════════════════
   PANEL 4 — SIGNAL EXPLORER
   ══════════════════════════════════════════════════════════════════ */
function PanelSignalExplorer({ patients }) {
  const [patient, setPatient] = useState(patients[0] || '')
  const [arm, setArm] = useState('impaired_arm')
  const [sessionNo, setSessionNo] = useState(null)
  const [movement, setMovement] = useState('')
  const [channel, setChannel] = useState('channel_01')

  const { data: sessionsData } = useApi(patient && arm ? `/uc2/sessions/${patient}/${arm}` : null)
  const sessions = sessionsData?.sessions || []

  // Auto-select first session
  if (sessions.length && sessionNo === null) {
    setSessionNo(sessions[0].session_no)
  }

  const { data: sigData, loading: sigLoading } = useApi(
    patient && arm && sessionNo ? `/uc2/signal/${patient}/${arm}/${sessionNo}?channel=${channel}` : null
  )

  const movements = sigData?.movements || []
  const selectedMovement = movement || movements[0] || ''
  const signalObj = sigData?.signals?.[selectedMovement]

  const chartData = useMemo(() => {
    if (!signalObj) return []
    return signalObj.time.map((t, i) => ({
      time: t,
      amplitude: signalObj.amplitude[i],
    }))
  }, [signalObj])

  if (!patients.length) return null

  return (
    <section className="uc2__section">
      <h2 className="uc2__section-title">🔬 Khám phá tín hiệu 1 buổi / 1 kênh</h2>

      <div className="uc2__select-row">
        <label className="uc2__select-label">
          Bệnh nhân
          <select value={patient} onChange={e => { setPatient(e.target.value); setSessionNo(null) }} className="uc2__select">
            {patients.map(p => <option key={p} value={p}>{p}</option>)}
          </select>
        </label>
        <label className="uc2__select-label">
          Tay
          <select value={arm} onChange={e => { setArm(e.target.value); setSessionNo(null) }} className="uc2__select">
            <option value="healthy_arm">healthy_arm</option>
            <option value="impaired_arm">impaired_arm</option>
          </select>
        </label>
      </div>

      <div className="uc2__select-row">
        <label className="uc2__select-label">
          Buổi
          <select value={sessionNo || ''} onChange={e => setSessionNo(+e.target.value)} className="uc2__select">
            {sessions.map(s => <option key={s.session_no} value={s.session_no}>Buổi {s.session_no} ({s.filename})</option>)}
          </select>
        </label>
        <label className="uc2__select-label">
          Cử chỉ
          <select value={selectedMovement} onChange={e => setMovement(e.target.value)} className="uc2__select">
            {movements.map(m => <option key={m} value={m}>{m}</option>)}
          </select>
        </label>
        <label className="uc2__select-label">
          Kênh
          <select value={channel} onChange={e => setChannel(e.target.value)} className="uc2__select">
            {(sigData?.channels || []).map(c => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>
      </div>

      {sigLoading ? (
        <div className="uc2-loading"><div className="uc2-loading__spinner" /><p>Đang tải tín hiệu…</p></div>
      ) : chartData.length > 0 ? (
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.08)" />
            <XAxis dataKey="time" stroke="#5A6B80" fontSize={12} tickFormatter={v => `${v.toFixed(1)}s`}>
              <Label value="Thời gian (s)" position="bottom" offset={0} fill="#5A6B80" fontSize={12} />
            </XAxis>
            <YAxis stroke="#5A6B80" fontSize={12}>
              <Label value="Biên độ (µV)" angle={-90} position="insideLeft" fill="#5A6B80" fontSize={12} />
            </YAxis>
            <Tooltip contentStyle={tooltipStyle} formatter={v => [`${v?.toFixed(2)} µV`, 'Biên độ']} labelFormatter={v => `${v}s`} />
            <Line type="monotone" dataKey="amplitude" stroke={TEAL} strokeWidth={1} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      ) : (
        <p className="uc2__caption">Không có dữ liệu cho lựa chọn này.</p>
      )}
    </section>
  )
}
