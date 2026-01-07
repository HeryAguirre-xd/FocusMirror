interface CameraPreviewProps {
  connected: boolean
}

export function CameraPreview({ connected }: CameraPreviewProps) {
  if (!connected) return null

  return (
    <div className="fixed bottom-4 right-4 z-40">
      <div className="relative overflow-hidden rounded-2xl shadow-2xl border border-white/10">
        <img
          src="http://127.0.0.1:8000/video"
          alt="Camera"
          className="w-40 h-auto object-cover"
          style={{ transform: 'scaleX(-1)' }}  // Mirror for natural feel
        />
        {/* Subtle gradient overlay */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/30 to-transparent pointer-events-none" />
      </div>
    </div>
  )
}
