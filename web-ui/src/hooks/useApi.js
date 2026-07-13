import { useState, useEffect } from 'react'

const API_BASE = '/api'

export function useApi(path, options = {}) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!path) { setLoading(false); return }

    let cancelled = false
    setLoading(true)
    setError(null)

    fetch(`${API_BASE}${path}`, options)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then(json => { if (!cancelled) { setData(json); setLoading(false) } })
      .catch(err => { if (!cancelled) { setError(err.message); setLoading(false) } })

    return () => { cancelled = true }
  }, [path])

  return { data, loading, error }
}
