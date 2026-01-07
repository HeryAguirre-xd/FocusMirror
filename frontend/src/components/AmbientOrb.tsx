import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

interface AmbientOrbProps {
  focusScore: number
}

export function AmbientOrb({ focusScore }: AmbientOrbProps) {
  const meshRef = useRef<THREE.Mesh>(null)
  const smoothedFocus = useRef(focusScore)

  // Colors: warm amber (focused) -> cool blue-gray (unfocused)
  const colorFocused = new THREE.Color('#e8a87c')   // Warm amber
  const colorUnfocused = new THREE.Color('#4a5568') // Cool blue-gray

  useFrame((state) => {
    if (!meshRef.current) return

    // Smooth focus transition (LERP)
    smoothedFocus.current += (focusScore - smoothedFocus.current) * 0.02

    // Interpolate color based on focus
    const color = colorUnfocused.clone().lerp(colorFocused, smoothedFocus.current)
    const material = meshRef.current.material as THREE.MeshStandardMaterial
    material.color = color
    material.emissive = color.clone().multiplyScalar(0.3)

    // Breathing animation - slower when focused
    const breathSpeed = 0.5 + (1 - smoothedFocus.current) * 0.5
    const breath = Math.sin(state.clock.elapsedTime * breathSpeed) * 0.1 + 1

    // Scale based on breathing
    meshRef.current.scale.setScalar(2 * breath)

    // Subtle rotation
    meshRef.current.rotation.y = state.clock.elapsedTime * 0.1
  })

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[1, 64, 64]} />
      <meshStandardMaterial
        color="#e8a87c"
        emissive="#e8a87c"
        emissiveIntensity={0.3}
        roughness={0.3}
        metalness={0.1}
      />
    </mesh>
  )
}
