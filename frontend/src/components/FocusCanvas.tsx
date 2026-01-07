import { Canvas } from '@react-three/fiber'
import { AmbientOrb } from './AmbientOrb'
import { Particles } from './Particles'
import type { HeadPosition } from '../hooks/useFocusSocket'

interface FocusCanvasProps {
  focusScore: number
  head: HeadPosition
}

export function FocusCanvas({ focusScore, head }: FocusCanvasProps) {
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

        {/* Particles around the orb */}
        <Particles focusScore={focusScore} head={head} />

        {/* Main orb */}
        <AmbientOrb focusScore={focusScore} head={head} />
      </Canvas>
    </div>
  )
}
