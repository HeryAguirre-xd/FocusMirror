import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import type { HeadPosition } from '../hooks/useFocusSocket'

interface AmbientOrbProps {
  focusScore: number
  head: HeadPosition
}

export function AmbientOrb({ focusScore, head }: AmbientOrbProps) {
  const meshRef = useRef<THREE.Mesh>(null)
  const materialRef = useRef<THREE.ShaderMaterial>(null)

  // Smoothed values for transitions
  const smoothedFocus = useRef(focusScore)
  const smoothedX = useRef(0)
  const smoothedY = useRef(0)
  const smoothedTilt = useRef(0)

  // Custom shader for morphing effect
  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uFocus: { value: 1.0 },
      uMorph: { value: 0.0 },
      uColorFocused: { value: new THREE.Color('#e8a87c') },
      uColorUnfocused: { value: new THREE.Color('#4a5568') },
    }),
    []
  )

  const vertexShader = `
    uniform float uTime;
    uniform float uMorph;

    varying vec2 vUv;
    varying vec3 vNormal;

    // Simplex noise for organic deformation
    vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
    vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
    vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
    vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

    float snoise(vec3 v) {
      const vec2 C = vec2(1.0/6.0, 1.0/3.0);
      const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
      vec3 i  = floor(v + dot(v, C.yyy));
      vec3 x0 = v - i + dot(i, C.xxx);
      vec3 g = step(x0.yzx, x0.xyz);
      vec3 l = 1.0 - g;
      vec3 i1 = min(g.xyz, l.zxy);
      vec3 i2 = max(g.xyz, l.zxy);
      vec3 x1 = x0 - i1 + C.xxx;
      vec3 x2 = x0 - i2 + C.yyy;
      vec3 x3 = x0 - D.yyy;
      i = mod289(i);
      vec4 p = permute(permute(permute(
                i.z + vec4(0.0, i1.z, i2.z, 1.0))
              + i.y + vec4(0.0, i1.y, i2.y, 1.0))
              + i.x + vec4(0.0, i1.x, i2.x, 1.0));
      float n_ = 0.142857142857;
      vec3 ns = n_ * D.wyz - D.xzx;
      vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
      vec4 x_ = floor(j * ns.z);
      vec4 y_ = floor(j - 7.0 * x_);
      vec4 x = x_ *ns.x + ns.yyyy;
      vec4 y = y_ *ns.x + ns.yyyy;
      vec4 h = 1.0 - abs(x) - abs(y);
      vec4 b0 = vec4(x.xy, y.xy);
      vec4 b1 = vec4(x.zw, y.zw);
      vec4 s0 = floor(b0)*2.0 + 1.0;
      vec4 s1 = floor(b1)*2.0 + 1.0;
      vec4 sh = -step(h, vec4(0.0));
      vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
      vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
      vec3 p0 = vec3(a0.xy, h.x);
      vec3 p1 = vec3(a0.zw, h.y);
      vec3 p2 = vec3(a1.xy, h.z);
      vec3 p3 = vec3(a1.zw, h.w);
      vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
      p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
      vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
      m = m * m;
      return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
    }

    void main() {
      vUv = uv;
      vNormal = normal;

      // Morphing displacement when unfocused
      float noise = snoise(position * 2.0 + uTime * 0.5);
      vec3 displaced = position + normal * noise * uMorph * 0.3;

      gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced, 1.0);
    }
  `

  const fragmentShader = `
    uniform float uTime;
    uniform float uFocus;
    uniform vec3 uColorFocused;
    uniform vec3 uColorUnfocused;

    varying vec2 vUv;
    varying vec3 vNormal;

    void main() {
      // Base color interpolation
      vec3 color = mix(uColorUnfocused, uColorFocused, uFocus);

      // Fresnel effect for glow
      vec3 viewDir = normalize(cameraPosition);
      float fresnel = pow(1.0 - max(dot(vNormal, viewDir), 0.0), 2.0);

      // Breathing pulse
      float breathSpeed = mix(0.8, 0.3, uFocus);
      float pulse = sin(uTime * breathSpeed) * 0.15 + 0.85;

      // Apply effects
      color *= pulse;
      color += fresnel * color * 0.5;

      // Add subtle iridescence when focused
      float iridescence = sin(vUv.x * 10.0 + uTime * 0.5) * 0.05 * uFocus;
      color += iridescence;

      gl_FragColor = vec4(color, 0.95);
    }
  `

  useFrame((state) => {
    if (!meshRef.current || !materialRef.current) return

    const time = state.clock.elapsedTime

    // Smooth all values (LERP)
    smoothedFocus.current += (focusScore - smoothedFocus.current) * 0.03
    smoothedX.current += ((head.x - 0.5) * 2 - smoothedX.current) * 0.08
    smoothedY.current += ((head.y - 0.5) * -2 - smoothedY.current) * 0.08
    smoothedTilt.current += (head.tilt - smoothedTilt.current) * 0.08

    // Update shader uniforms
    materialRef.current.uniforms.uTime.value = time
    materialRef.current.uniforms.uFocus.value = smoothedFocus.current
    materialRef.current.uniforms.uMorph.value = 1 - smoothedFocus.current

    // Head-following position (subtle movement)
    meshRef.current.position.x = smoothedX.current * 0.8
    meshRef.current.position.y = smoothedY.current * 0.5

    // Head tilt rotation
    meshRef.current.rotation.z = (smoothedTilt.current * Math.PI) / 180 * 0.3

    // Breathing scale
    const breathSpeed = 0.5 + (1 - smoothedFocus.current) * 0.3
    const breath = Math.sin(time * breathSpeed) * 0.1 + 1
    const baseScale = 1.8 + smoothedFocus.current * 0.4
    meshRef.current.scale.setScalar(baseScale * breath)

    // Slow rotation
    meshRef.current.rotation.y = time * 0.1
  })

  return (
    <mesh ref={meshRef}>
      <icosahedronGeometry args={[1, 4]} />
      <shaderMaterial
        ref={materialRef}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={uniforms}
        transparent
      />
    </mesh>
  )
}
