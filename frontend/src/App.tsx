import { useFocusSocket } from './hooks/useFocusSocket'
import { FocusCanvas } from './components/FocusCanvas'
import { SessionOverlay } from './components/SessionOverlay'

function App() {
  const { focusScore, session, connected, requestSessionSummary } = useFocusSocket()

  return (
    <div className="w-full h-full relative">
      {/* Main ambient canvas */}
      <FocusCanvas focusScore={focusScore} />

      {/* Session stats overlay (Escape key) */}
      <SessionOverlay
        session={session}
        connected={connected}
        requestSessionSummary={requestSessionSummary}
      />

      {/* Connection indicator (bottom-left) */}
      <div className="fixed bottom-4 left-4 flex items-center gap-2 text-white/50 text-xs">
        <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500' : 'bg-amber-500 animate-pulse'}`} />
        {connected ? `Focus: ${focusScore.toFixed(2)}` : 'Connecting...'}
      </div>
    </div>
  )
}

export default App
