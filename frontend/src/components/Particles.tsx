import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import type { HeadPosition } from '../hooks/useFocusSocket'

interface ParticlesProps {
  focusScore: number
  head: HeadPosition
  count?: number
}

export function Particles({ focusScore, head, count = 100 }: ParticlesProps) {
  const meshRef = useRef<THREE.Points>(null)
  const materialRef = useRef<THREE.ShaderMaterial>(null)

  // Smoothed values
  const smoothedFocus = useRef(focusScore)
  const smoothedX = useRef(0)
  const smoothedY = useRef(0)

  // Generate particle positions
  const [positions, velocities] = useMemo(() => {
    const pos = new Float32Array(count * 3)
    const vel = new Float32Array(count * 3)

    for (let i = 0; i < count; i++) {
      const i3 = i * 3
      // Distribute in a sphere around the orb
      const theta = Math.random() * Math.PI * 2
      const phi = Math.acos(2 * Math.random() - 1)
      const r = 2 + Math.random() * 2

      pos[i3] = r * Math.sin(phi) * Math.cos(theta)
      pos[i3 + 1] = r * Math.sin(phi) * Math.sin(theta)
      pos[i3 + 2] = r * Math.cos(phi)

      // Random velocities for orbital motion
      vel[i3] = (Math.random() - 0.5) * 0.02
      vel[i3 + 1] = (Math.random() - 0.5) * 0.02
      vel[i3 + 2] = (Math.random() - 0.5) * 0.02
    }

    return [pos, vel]
  }, [count])

  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uFocus: { value: 1.0 },
      uColorFocused: { value: new THREE.Color('#e8a87c') },
      uColorUnfocused: { value: new THREE.Color('#4a5568') },
    }),
    []
  )

  const vertexShader = `
    uniform float uTime;
    uniform float uFocus;

    attribute vec3 velocity;

    varying float vAlpha;

    void main() {
      // Orbital motion
      float speed = mix(0.3, 0.1, uFocus);
      vec3 pos = position;

      // Add swirling motion
      float angle = uTime * speed + length(position) * 0.5;
      float c = cos(angle);
      float s = sin(angle);
      pos.x = position.x * c - position.z * s;
      pos.z = position.x * s + position.z * c;

      // Pulsing distance from center
      float pulse = sin(uTime * 0.5 + length(position)) * 0.2;
      pos *= 1.0 + pulse * (1.0 - uFocus);

      // Add velocity for chaotic motion when unfocused
      pos += velocity * uTime * (1.0 - uFocus) * 10.0;

      vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
      gl_Position = projectionMatrix * mvPosition;

      // Size based on focus and distance
      float size = mix(3.0, 1.5, uFocus);
      gl_PointSize = size * (300.0 / -mvPosition.z);

      // Alpha based on distance from center
      vAlpha = mix(0.6, 0.3, uFocus) * (1.0 - length(position) / 5.0);
    }
  `

  const fragmentShader = `
    uniform float uFocus;
    uniform vec3 uColorFocused;
    uniform vec3 uColorUnfocused;

    varying float vAlpha;

    void main() {
      // Circular particle shape
      vec2 center = gl_PointCoord - vec2(0.5);
      float dist = length(center);
      if (dist > 0.5) discard;

      // Soft edges
      float alpha = (1.0 - dist * 2.0) * vAlpha;

      // Color based on focus
      vec3 color = mix(uColorUnfocused, uColorFocused, uFocus);

      gl_FragColor = vec4(color, alpha);
    }
  `

  useFrame((state) => {
    if (!meshRef.current || !materialRef.current) return

    const time = state.clock.elapsedTime

    // Smooth values
    smoothedFocus.current += (focusScore - smoothedFocus.current) * 0.03
    smoothedX.current += ((head.x - 0.5) * 2 - smoothedX.current) * 0.05
    smoothedY.current += ((head.y - 0.5) * -2 - smoothedY.current) * 0.05

    // Update uniforms
    materialRef.current.uniforms.uTime.value = time
    materialRef.current.uniforms.uFocus.value = smoothedFocus.current

    // Follow head position (more subtle than orb)
    meshRef.current.position.x = smoothedX.current * 0.4
    meshRef.current.position.y = smoothedY.current * 0.25
  })

  return (
    <points ref={meshRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={count}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-velocity"
          count={count}
          array={velocities}
          itemSize={3}
        />
      </bufferGeometry>
      <shaderMaterial
        ref={materialRef}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={uniforms}
        transparent
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  )
}
