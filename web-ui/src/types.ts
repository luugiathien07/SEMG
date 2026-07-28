export type AccentColor = 'coral' | 'teal'

export interface IntroCard {
  icon: string
  label: string
  text: string
}

export interface IntroContent {
  badge: string
  title: string
  subtitle: string
  cards: IntroCard[]
  outputs: string[]
  footer: string
  demoPath: string
  accentColor: AccentColor
}

export interface SessionData {
  session: number
  endurance_sec: number
  pct_nonfatigue: number
  t_centers: number[]
  proba: number[]
}

export interface EnduranceResponse {
  sessions: SessionData[]
  f1: number
}

export interface PatientsResponse {
  patients: string[]
}

export interface RecoveryRow {
  'Buổi': number
  RMS: number
  MDF: number
  'Symmetry (%)': number
}

export interface RecoveryResponse {
  rows: RecoveryRow[]
  baseline_mdf?: number
}

export interface SessionMeta {
  session_no: number
  filename: string
}

export interface SessionsResponse {
  sessions: SessionMeta[]
}

export interface SignalSeries {
  time: number[]
  amplitude: number[]
}

export interface SignalResponse {
  movements: string[]
  channels: string[]
  signals: Record<string, SignalSeries>
}
