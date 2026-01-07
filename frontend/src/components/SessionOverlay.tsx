import { useEffect, useState, useCallback } from 'react'
import type { SessionData, SessionSummary } from '../hooks/useFocusSocket'

interface SessionOverlayProps {
  session: SessionData
  connected: boolean
  requestSessionSummary: () => Promise<SessionSummary | null>
}

function formatTime(seconds: number): string {
  const hrs = Math.floor(seconds / 3600)
  const mins = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)

  if (hrs > 0) {
    return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

export function SessionOverlay({ session, connected, requestSessionSummary }: SessionOverlayProps) {
  const [visible, setVisible] = useState(false)
  const [timeline, setTimeline] = useState<Array<{ t: number; score: number }>>([])
  const [hideTimeout, setHideTimeout] = useState<number | null>(null)

  const showOverlay = useCallback(async () => {
    setVisible(true)

    // Request session summary for timeline
    const summary = await requestSessionSummary()
    if (summary?.timeline) {
      setTimeline(summary.timeline)
    }

    // Clear existing timeout
    if (hideTimeout) {
      clearTimeout(hideTimeout)
    }

    // Auto-hide after 5 seconds
    const timeout = window.setTimeout(() => {
      setVisible(false)
    }, 5000)
    setHideTimeout(timeout)
  }, [requestSessionSummary, hideTimeout])

  const hideOverlay = useCallback(() => {
    setVisible(false)
    if (hideTimeout) {
      clearTimeout(hideTimeout)
      setHideTimeout(null)
    }
  }, [hideTimeout])

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (visible) {
          hideOverlay()
        } else {
          showOverlay()
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [visible, showOverlay, hideOverlay])

  // Reset hide timer on mouse movement when visible
  useEffect(() => {
    if (!visible) return

    const handleMouseMove = () => {
      if (hideTimeout) {
        clearTimeout(hideTimeout)
      }
      const timeout = window.setTimeout(() => {
        setVisible(false)
      }, 5000)
      setHideTimeout(timeout)
    }

    window.addEventListener('mousemove', handleMouseMove)
    return () => window.removeEventListener('mousemove', handleMouseMove)
  }, [visible, hideTimeout])

  if (!visible) return null

  return (
    <div className="fixed inset-0 flex items-center justify-center z-50 pointer-events-none">
      <div
        className="bg-black/60 backdrop-blur-xl rounded-3xl p-8 pointer-events-auto
                   border border-white/10 shadow-2xl
                   animate-in fade-in duration-300"
        style={{ minWidth: '320px' }}
      >
        {/* Connection status */}
        <div className="flex items-center gap-2 mb-6">
          <div
            className={`w-2 h-2 rounded-full ${connected ? 'bg-green-400' : 'bg-red-400'}`}
          />
          <span className="text-white/50 text-xs tracking-wide uppercase">
            {connected ? 'Connected' : 'Disconnected'}
          </span>
        </div>

        {/* Session time */}
        <div className="text-center mb-6">
          <div className="text-5xl font-light text-white tracking-tight">
            {formatTime(session.duration)}
          </div>
          <div className="text-white/40 text-sm mt-1 tracking-wide">
            Session Duration
          </div>
        </div>

        {/* Focus stats */}
        <div className="flex justify-between gap-8 mb-6">
          <div className="text-center">
            <div className="text-2xl font-light text-white">
              {formatTime(session.focused_time)}
            </div>
            <div className="text-white/40 text-xs tracking-wide">
              Focused
            </div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-light text-amber-400">
              {session.focus_percentage.toFixed(0)}%
            </div>
            <div className="text-white/40 text-xs tracking-wide">
              Focus Rate
            </div>
          </div>
        </div>

        {/* Timeline bar */}
        {timeline.length > 0 && (
          <div className="mt-4">
            <div className="h-1 bg-white/10 rounded-full overflow-hidden flex">
              {timeline.map((point, i) => (
                <div
                  key={i}
                  className="h-full transition-colors"
                  style={{
                    flex: 1,
                    backgroundColor:
                      point.score >= 0.7
                        ? 'rgb(232, 168, 124)' // Amber
                        : point.score >= 0.4
                        ? 'rgb(180, 150, 120)' // Muted
                        : 'rgb(74, 85, 104)',  // Cool gray
                  }}
                />
              ))}
            </div>
            <div className="flex justify-between mt-1">
              <span className="text-white/30 text-xs">Start</span>
              <span className="text-white/30 text-xs">Now</span>
            </div>
          </div>
        )}

        {/* Dismiss hint */}
        <div className="text-center mt-6 text-white/30 text-xs">
          Press <kbd className="px-1.5 py-0.5 bg-white/10 rounded text-white/50">Esc</kbd> to close
        </div>
      </div>
    </div>
  )
}
