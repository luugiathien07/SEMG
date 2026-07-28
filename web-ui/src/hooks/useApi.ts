import { useState, useEffect } from 'react'

const API_BASE = '/api'

export function useApi<T = unknown>(
  path: string | null | undefined,
  options: RequestInit = {},
): { data: T | null; loading: boolean; error: string | null } {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!path) { setLoading(false); return }

    let cancelled = false
    setLoading(true)
    setError(null)

    fetch(`${API_BASE}${path}`, options)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json() as Promise<T>
      })
      .then(json => { if (!cancelled) { setData(json); setLoading(false) } })
      .catch((err: Error) => { if (!cancelled) { setError(err.message); setLoading(false) } })

    return () => { cancelled = true }
  }, [path]) // eslint-disable-line react-hooks/exhaustive-deps

  return { data, loading, error }
}
