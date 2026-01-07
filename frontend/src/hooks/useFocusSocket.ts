import { useEffect, useRef, useState, useCallback } from 'react'

export interface SessionData {
  duration: number
  focused_time: number
  focus_percentage: number
}

export interface HeadPosition {
  x: number  // 0-1, horizontal position
  y: number  // 0-1, vertical position
  tilt: number  // degrees, head tilt
}

export interface FocusState {
  focusScore: number
  rawScore: number
  graceActive: boolean
  faceDetected: boolean
  head: HeadPosition
  session: SessionData
  connected: boolean
}

export interface SessionSummary extends SessionData {
  timeline: Array<{ t: number; score: number }>
}

const INITIAL_STATE: FocusState = {
  focusScore: 1.0,
  rawScore: 1.0,
  graceActive: false,
  faceDetected: false,
  head: {
    x: 0.5,
    y: 0.5,
    tilt: 0,
  },
  session: {
    duration: 0,
    focused_time: 0,
    focus_percentage: 100,
  },
  connected: false,
}

const WS_URL = 'ws://127.0.0.1:8000/ws'
const RECONNECT_DELAY = 2000

export function useFocusSocket() {
  const [state, setState] = useState<FocusState>(INITIAL_STATE)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<number | null>(null)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      setState(prev => ({ ...prev, connected: true }))
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)

        // Skip session summary messages
        if (data.type === 'session') return

        setState({
          focusScore: data.focus_score,
          rawScore: data.raw_score,
          graceActive: data.grace_active,
          faceDetected: data.face_detected,
          head: {
            x: data.head?.x ?? 0.5,
            y: data.head?.y ?? 0.5,
            tilt: data.head?.tilt ?? 0,
          },
          session: {
            duration: data.session.duration,
            focused_time: data.session.focused_time,
            focus_percentage: data.session.focus_percentage,
          },
          connected: true,
        })
      } catch (e) {
        console.error('Failed to parse focus data:', e)
      }
    }

    ws.onclose = () => {
      setState(prev => ({ ...prev, connected: false }))
      wsRef.current = null

      // Attempt to reconnect
      reconnectTimeoutRef.current = window.setTimeout(() => {
        connect()
      }, RECONNECT_DELAY)
    }

    ws.onerror = () => {
      ws.close()
    }
  }, [])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
  }, [])

  const requestSessionSummary = useCallback((): Promise<SessionSummary | null> => {
    return new Promise((resolve) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        resolve(null)
        return
      }

      const handleMessage = (event: MessageEvent) => {
        try {
          const data = JSON.parse(event.data)
          if (data.type === 'session') {
            wsRef.current?.removeEventListener('message', handleMessage)
            resolve(data.data as SessionSummary)
          }
        } catch {
          // Ignore parse errors for other messages
        }
      }

      wsRef.current.addEventListener('message', handleMessage)
      wsRef.current.send(JSON.stringify({ type: 'get_session' }))

      // Timeout after 2 seconds
      setTimeout(() => {
        wsRef.current?.removeEventListener('message', handleMessage)
        resolve(null)
      }, 2000)
    })
  }, [])

  useEffect(() => {
    connect()
    return () => disconnect()
  }, [connect, disconnect])

  return {
    ...state,
    requestSessionSummary,
    reconnect: connect,
  }
}
