import './MetricCard.css'

export default function MetricCard({ icon, label, value, delta, colorClass = '' }) {
  return (
    <div className={`metric-card glass-card ${colorClass}`}>
      <div className={`metric-card__icon metric-card__icon--${colorClass || 'teal'}`}>
        {icon}
      </div>
      <div className="metric-card__label">{label}</div>
      <div className="metric-card__value">{value}</div>
      {delta && (
        <div className={`metric-card__delta ${delta.startsWith('+') ? 'metric-card__delta--up' : 'metric-card__delta--down'}`}>
          {delta}
        </div>
      )}
    </div>
  )
}
