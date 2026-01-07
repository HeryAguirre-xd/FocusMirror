import { Canvas } from '@react-three/fiber'
import { AmbientOrb } from './AmbientOrb'

interface FocusCanvasProps {
  focusScore: number
}

export function FocusCanvas({ focusScore }: FocusCanvasProps) {
  return (
    <div style={{ width: '100vw', height: '100vh', background: '#0a0a0a' }}>
      <Canvas
        camera={{ position: [0, 0, 5], fov: 45 }}
        gl={{
          antialias: true,
          alpha: true,
        }}
      >
        <color attach="background" args={['#0a0a0a']} />
        <ambientLight intensity={0.3} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        <pointLight position={[-10, -10, -10]} intensity={0.5} />
        <AmbientOrb focusScore={focusScore} />
      </Canvas>
    </div>
  )
}
