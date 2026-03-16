import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react'

// ─── Complex arithmetic (all [re, im]) ─────────────────────────────────────
const cAdd = (a, b) => [a[0] + b[0], a[1] + b[1]]
const cSub = (a, b) => [a[0] - b[0], a[1] - b[1]]
const cMul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
const cDiv = (a, b) => {
  const d = b[0] * b[0] + b[1] * b[1]
  if (d === 0) return [0, 0]
  return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]
}
const cExp = (z) => {
  const r = Math.exp(z[0])
  return [r * Math.cos(z[1]), r * Math.sin(z[1])]
}
const cLog = (z) => [Math.log(Math.sqrt(z[0] * z[0] + z[1] * z[1])), Math.atan2(z[1], z[0])]
const cSqrt = (z) => {
  const r = Math.sqrt(Math.sqrt(z[0] * z[0] + z[1] * z[1]))
  const t = Math.atan2(z[1], z[0]) / 2
  return [r * Math.cos(t), r * Math.sin(t)]
}
const cAbs = (z) => Math.sqrt(z[0] * z[0] + z[1] * z[1])
const cScale = (z, s) => [z[0] * s, z[1] * s]

function computeW(gammaTilde) {
  return cSqrt([0, -gammaTilde / 2])
}

function phiPrime(theta, w, beta) {
  const cosB2 = Math.cos(beta / 2)
  const sinB2 = Math.sin(beta / 2)
  const thetaW = cMul(theta, w)
  const expTW = cExp(thetaW)
  const iSinB2 = [0, sinB2]
  const A = cSub([cosB2, 0], cMul(iSinB2, expTW))
  const negIW = cMul([0, -1], w)
  const Aprime = cMul(cScale(negIW, sinB2), expTW)
  const negTheta2 = cScale(theta, -0.5)
  const ratio = cDiv(Aprime, A)
  return cAdd(negTheta2, ratio)
}

function phiDoublePrime(theta, w, beta) {
  const cosB2 = Math.cos(beta / 2)
  const sinB2 = Math.sin(beta / 2)
  const thetaW = cMul(theta, w)
  const expTW = cExp(thetaW)
  const A = cSub([cosB2, 0], cMul([0, sinB2], expTW))
  const A2 = cMul(A, A)
  const w2 = cMul(w, w)
  const num = cMul(cMul(cScale([0, -1], sinB2 * cosB2), w2), expTW)
  const dLogA = cDiv(num, A2)
  return cAdd([-0.5, 0], dLogA)
}

function computePoles(w, beta, R, numK = 20) {
  const cosB2 = Math.cos(beta / 2)
  const sinB2 = Math.sin(beta / 2)
  const cotB2 = cosB2 / sinB2
  const logTerm = cLog([0, -cotB2])
  const poles = []
  for (let k = -numK; k <= numK; k++) {
    const num = cAdd(logTerm, [0, 2 * Math.PI * k])
    const pole = cDiv(num, w)
    if (cAbs(pole) < R * 1.01) poles.push({ pos: pole, k })
  }
  return poles
}

function findSaddlePoints(w, beta, R, maxSaddles = 30) {
  const saddles = []
  const gridN = 12
  for (let i = -gridN; i <= gridN; i++) {
    for (let j = -gridN; j <= gridN; j++) {
      const guess = [(i * R) / gridN, (j * R) / gridN]
      let theta = [...guess]
      let converged = false
      for (let iter = 0; iter < 60; iter++) {
        const fp = phiPrime(theta, w, beta)
        const fpp = phiDoublePrime(theta, w, beta)
        if (cAbs(fpp) < 1e-15) break
        const step = cDiv(fp, fpp)
        theta = cSub(theta, step)
        if (cAbs(fp) < 1e-10) {
          converged = true
          break
        }
      }
      if (converged && cAbs(theta) < R * 0.99) {
        const isDuplicate = saddles.some((s) => cAbs(cSub(s, theta)) < 1e-5)
        if (!isDuplicate) saddles.push(theta)
      }
    }
  }
  return saddles.slice(0, maxSaddles)
}

function computeWindingNumber(curve) {
  if (!curve.length) return 0
  let totalAngle = 0
  for (let i = 0; i < curve.length; i++) {
    const curr = curve[i]
    const next = curve[(i + 1) % curve.length]
    const angle1 = Math.atan2(curr[1], curr[0])
    const angle2 = Math.atan2(next[1], next[0])
    let dAngle = angle2 - angle1
    if (dAngle > Math.PI) dAngle -= 2 * Math.PI
    if (dAngle < -Math.PI) dAngle += 2 * Math.PI
    totalAngle += dAngle
  }
  return Math.round(totalAngle / (2 * Math.PI))
}

function isGoodArc(phi, w, epsilon = 0.1) {
  const eiPhi = [Math.cos(phi), Math.sin(phi)]
  const product = cMul(eiPhi, w)
  return product[0] <= -epsilon
}

// ─── Constants ───────────────────────────────────────────────────────────
const COLORS = {
  bg: '#0a0e1a',
  grid: '#1a2035',
  axes: '#3a4560',
  good: '#00d4ff',
  bad: '#ff6b35',
  idealCircle: 'rgba(255,255,255,0.25)',
  poles: '#ff3355',
  saddles: '#00ff88',
  origin: '#ffffff',
  tracer: '#ffee00',
}

const CANVAS_SIZE = 480
const NUM_SAMPLES = 1200
const EPSILON_ARC = 0.1

export default function ArgumentPrincipleViz() {
  const [gammaTilde, setGammaTilde] = useState(2 * Math.PI)
  const [R, setR] = useState(5)
  const [beta, setBeta] = useState(-Math.PI / 2)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [gammaAnimating, setGammaAnimating] = useState(false)
  const [testPoints, setTestPoints] = useState([])
  const [mathOpen, setMathOpen] = useState(false)
  const [hoverTheta, setHoverTheta] = useState(null)
  const [phase, setPhase] = useState(0)
  const thetaCanvasRef = useRef(null)
  const imageCanvasRef = useRef(null)

  const w = useMemo(() => computeW(gammaTilde), [gammaTilde])
  const poles = useMemo(() => computePoles(w, beta, R), [w, beta, R])
  const saddles = useMemo(() => findSaddlePoints(w, beta, R), [w, beta, R])

  const contourData = useMemo(() => {
    const points = []
    for (let i = 0; i <= NUM_SAMPLES; i++) {
      const phi = (i / NUM_SAMPLES) * 2 * Math.PI
      const theta = [R * Math.cos(phi), R * Math.sin(phi)]
      const fp = phiPrime(theta, w, beta)
      const good = isGoodArc(phi, w, EPSILON_ARC)
      points.push({ phi, theta, fp, good })
    }
    return points
  }, [R, w, beta])

  const winding = useMemo(
    () => computeWindingNumber(contourData.map((p) => p.fp)),
    [contourData]
  )

  const polesInside = useMemo(
    () => poles.filter((p) => cAbs(p.pos) < R).length,
    [poles, R]
  )

  const explanation = useMemo(() => {
    const small = gammaTilde < Math.PI
    const large = gammaTilde > 4
    let text = ''
    if (small) text = 'At small γ̃, the poles are far apart and G(θ) is small everywhere on C_R. The image curve nearly matches the ideal circle −θ/2.'
    else if (large) text = 'At large γ̃, the poles are dense and G(θ) creates large distortions on the bad arcs (red). But the good arcs (blue) still force the curve to wind around the origin.'
    else text = 'On the good arcs (blue), Φ′(θ) ≈ −θ/2; on the bad arcs (orange), G(θ) can be large. The winding number stays nonzero.'
    const zeros = winding + polesInside
    text += ` Winding number = ${winding}. Since there are ${polesInside} poles inside C_R, there must be at least ${zeros} zeros of Φ′ inside C_R — i.e., at least ${zeros} saddle points.`
    return text
  }, [gammaTilde, winding, polesInside])

  const toThetaPx = useCallback(
    (theta) => {
      const scale = (CANVAS_SIZE * 0.4) / Math.max(R, 1)
      return [
        CANVAS_SIZE / 2 + theta[0] * scale,
        CANVAS_SIZE / 2 - theta[1] * scale,
      ]
    },
    [R]
  )

  const fromThetaPx = useCallback(
    (px, py) => {
      const scale = (CANVAS_SIZE * 0.4) / Math.max(R, 1)
      return [
        (px - CANVAS_SIZE / 2) / scale,
        -(py - CANVAS_SIZE / 2) / scale,
      ]
    },
    [R]
  )

  const toImagePx = useCallback((fp, scaleR) => {
    const r = scaleR ?? R / 2
    const scale = (CANVAS_SIZE * 0.4) / Math.max(r, 0.5)
    return [
      CANVAS_SIZE / 2 + fp[0] * scale,
      CANVAS_SIZE / 2 - fp[1] * scale,
    ]
  }, [R])

  useEffect(() => {
    if (!playing && !gammaAnimating) return
    let raf
    const tick = () => {
      if (gammaAnimating) {
        setGammaTilde((g) => {
          if (g >= 8 * Math.PI - 0.1) {
            setGammaAnimating(false)
            return 8 * Math.PI
          }
          return g + 0.02
        })
      } else {
        setPhase((p) => {
          let next = p + 0.004 * speed
          if (next >= 2 * Math.PI) next -= 2 * Math.PI
          return next
        })
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing, speed, gammaAnimating])

  const drawThetaPlane = useCallback(
    (ctx, phase) => {
      ctx.fillStyle = COLORS.bg
      ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE)
      const scale = (CANVAS_SIZE * 0.4) / Math.max(R, 1)
      const ox = CANVAS_SIZE / 2
      const oy = CANVAS_SIZE / 2

      for (let i = -10; i <= 10; i++) {
        ctx.strokeStyle = COLORS.grid
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(ox + i * scale * 2, 0)
        ctx.lineTo(ox + i * scale * 2, CANVAS_SIZE)
        ctx.stroke()
        ctx.beginPath()
        ctx.moveTo(0, oy + i * scale * 2)
        ctx.lineTo(CANVAS_SIZE, oy + i * scale * 2)
        ctx.stroke()
      }

      ctx.strokeStyle = COLORS.axes
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(ox, 0)
      ctx.lineTo(ox, CANVAS_SIZE)
      ctx.moveTo(0, oy)
      ctx.lineTo(CANVAS_SIZE, oy)
      ctx.stroke()

      ctx.font = '14px monospace'
      ctx.fillStyle = COLORS.axes
      ctx.fillText('Re(θ)', CANVAS_SIZE - 50, oy - 8)
      ctx.fillText('Im(θ)', ox - 8, 20)

      for (let i = 0; i < contourData.length; i++) {
        const { theta, good } = contourData[i]
        const [x, y] = toThetaPx(theta)
        ctx.strokeStyle = good ? COLORS.good : COLORS.bad
        ctx.lineWidth = 2.5
        if (i === 0) ctx.beginPath()
        else ctx.lineTo(x, y)
      }
      ctx.closePath()
      ctx.stroke()

      poles.forEach(({ pos }) => {
        const [px, py] = toThetaPx(pos)
        if (px >= -20 && px <= CANVAS_SIZE + 20 && py >= -20 && py <= CANVAS_SIZE + 20) {
          ctx.strokeStyle = COLORS.poles
          ctx.lineWidth = 2
          ctx.beginPath()
          ctx.moveTo(px - 8, py - 8)
          ctx.lineTo(px + 8, py + 8)
          ctx.moveTo(px + 8, py - 8)
          ctx.lineTo(px - 8, py + 8)
          ctx.stroke()
        }
      })

      saddles.forEach((pos) => {
        const [px, py] = toThetaPx(pos)
        if (px >= -10 && px <= CANVAS_SIZE + 10 && py >= -10 && py <= CANVAS_SIZE + 10) {
          ctx.fillStyle = COLORS.saddles
          ctx.beginPath()
          ctx.arc(px, py, 5, 0, 2 * Math.PI)
          ctx.fill()
        }
      })

      ctx.fillStyle = COLORS.origin
      ctx.beginPath()
      ctx.arc(ox, oy, 4, 0, 2 * Math.PI)
      ctx.fill()

      const tracerTheta = [R * Math.cos(phase), R * Math.sin(phase)]
      const [tx, ty] = toThetaPx(tracerTheta)
      ctx.shadowColor = COLORS.tracer
      ctx.shadowBlur = 12
      ctx.fillStyle = COLORS.tracer
      ctx.beginPath()
      ctx.arc(tx, ty, 6, 0, 2 * Math.PI)
      ctx.fill()
      ctx.shadowBlur = 0

      testPoints.forEach((theta) => {
        const [qx, qy] = toThetaPx(theta)
        ctx.strokeStyle = '#aaffaa'
        ctx.lineWidth = 1
        ctx.setLineDash([4, 4])
        ctx.beginPath()
        ctx.arc(qx, qy, 8, 0, 2 * Math.PI)
        ctx.stroke()
        ctx.setLineDash([])
      })

      if (hoverTheta) {
        const [hx, hy] = toThetaPx(hoverTheta)
        ctx.strokeStyle = 'rgba(255,255,255,0.7)'
        ctx.lineWidth = 1
        ctx.setLineDash([2, 2])
        ctx.beginPath()
        ctx.moveTo(hx, 0)
        ctx.lineTo(hx, CANVAS_SIZE)
        ctx.moveTo(0, hy)
        ctx.lineTo(CANVAS_SIZE, hy)
        ctx.stroke()
        ctx.setLineDash([])
      }
    },
    [contourData, poles, saddles, R, toThetaPx, testPoints, hoverTheta]
  )

  const drawImagePlane = useCallback(
    (ctx, phase, partialWindingVal) => {
      ctx.fillStyle = COLORS.bg
      ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE)
      const scaleR = Math.max(R / 2, 0.5)
      const scale = (CANVAS_SIZE * 0.4) / scaleR
      const ox = CANVAS_SIZE / 2
      const oy = CANVAS_SIZE / 2

      for (let i = -10; i <= 10; i++) {
        ctx.strokeStyle = COLORS.grid
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(ox + i * scale * 2, 0)
        ctx.lineTo(ox + i * scale * 2, CANVAS_SIZE)
        ctx.stroke()
        ctx.beginPath()
        ctx.moveTo(0, oy + i * scale * 2)
        ctx.lineTo(CANVAS_SIZE, oy + i * scale * 2)
        ctx.stroke()
      }

      ctx.strokeStyle = COLORS.axes
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(ox, 0)
      ctx.lineTo(ox, CANVAS_SIZE)
      ctx.moveTo(0, oy)
      ctx.lineTo(CANVAS_SIZE, oy)
      ctx.stroke()

      ctx.font = '14px monospace'
      ctx.fillStyle = COLORS.axes
      ctx.fillText("Re(Φ′)", CANVAS_SIZE - 55, oy - 8)
      ctx.fillText("Im(Φ′)", ox - 8, 20)

      ctx.setLineDash([6, 4])
      ctx.strokeStyle = COLORS.idealCircle
      ctx.lineWidth = 1.5
      ctx.beginPath()
      for (let i = 0; i <= 100; i++) {
        const phi = (i / 100) * 2 * Math.PI
        const re = (R / 2) * Math.cos(phi)
        const im = (R / 2) * Math.sin(phi)
        const [x, y] = toImagePx([re, im], R / 2)
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      }
      ctx.closePath()
      ctx.stroke()
      ctx.setLineDash([])

      for (let i = 0; i < contourData.length; i++) {
        const { fp, good } = contourData[i]
        const [x, y] = toImagePx(fp)
        ctx.strokeStyle = good ? COLORS.good : COLORS.bad
        ctx.lineWidth = 2.5
        if (i === 0) ctx.beginPath()
        else ctx.lineTo(x, y)
      }
      ctx.closePath()
      ctx.stroke()

      ctx.fillStyle = COLORS.origin
      ctx.beginPath()
      ctx.arc(ox, oy, 5, 0, 2 * Math.PI)
      ctx.fill()

      ctx.font = 'bold 28px monospace'
      ctx.fillStyle = winding !== 0 ? '#00ff88' : '#ffffff'
      if (winding !== 0) {
        ctx.shadowColor = '#00ff88'
        ctx.shadowBlur = 8
      }
      ctx.fillText(`Winding = ${winding}`, 16, 36)
      ctx.shadowBlur = 0
      ctx.font = '14px monospace'
      ctx.fillStyle = 'rgba(255,255,255,0.7)'
      ctx.fillText(`(partial: ${partialWindingVal})`, 16, 54)

      const tracerTheta = [R * Math.cos(phase), R * Math.sin(phase)]
      const tracerFp = phiPrime(tracerTheta, w, beta)
      const [tx, ty] = toImagePx(tracerFp)
      ctx.shadowColor = COLORS.tracer
      ctx.shadowBlur = 12
      ctx.fillStyle = COLORS.tracer
      ctx.beginPath()
      ctx.arc(tx, ty, 6, 0, 2 * Math.PI)
      ctx.fill()
      ctx.shadowBlur = 0

      testPoints.forEach((theta) => {
        const fp = phiPrime(theta, w, beta)
        const [qx, qy] = toImagePx(fp)
        ctx.fillStyle = 'rgba(170,255,170,0.9)'
        ctx.beginPath()
        ctx.arc(qx, qy, 6, 0, 2 * Math.PI)
        ctx.fill()
      })

      if (hoverTheta) {
        const fp = phiPrime(hoverTheta, w, beta)
        const [hx, hy] = toImagePx(fp)
        ctx.strokeStyle = 'rgba(255,255,255,0.8)'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.arc(hx, hy, 10, 0, 2 * Math.PI)
        ctx.stroke()
      }
    },
    [contourData, winding, R, w, beta, toImagePx, testPoints, hoverTheta]
  )

  const partialWinding = useMemo(() => {
    const idx = Math.min(
      Math.floor((phase / (2 * Math.PI)) * contourData.length),
      contourData.length - 1
    )
    if (idx < 2) return 0
    const partial = contourData.slice(0, idx + 1).map((p) => p.fp)
    const closed = [...partial, contourData[0].fp]
    return computeWindingNumber(closed)
  }, [phase, contourData])

  useEffect(() => {
    const thetaC = thetaCanvasRef.current
    const imageC = imageCanvasRef.current
    if (!thetaC || !imageC) return
    const ctxT = thetaC.getContext('2d')
    const ctxI = imageC.getContext('2d')
    drawThetaPlane(ctxT, phase)
    drawImagePlane(ctxI, phase, partialWinding)
  }, [phase, partialWinding, drawThetaPlane, drawImagePlane, contourData, poles, saddles, winding, testPoints, hoverTheta])

  const handleThetaMouse = useCallback(
    (e) => {
      const rect = e.currentTarget.getBoundingClientRect()
      const px = e.clientX - rect.left
      const py = e.clientY - rect.top
      const theta = fromThetaPx(px, py)
      setHoverTheta(theta)
    },
    [fromThetaPx]
  )

  const handleThetaLeave = useCallback(() => setHoverTheta(null), [])

  const handleThetaClick = useCallback(
    (e) => {
      const rect = e.currentTarget.getBoundingClientRect()
      const px = e.clientX - rect.left
      const py = e.clientY - rect.top
      const theta = fromThetaPx(px, py)
      setTestPoints((prev) => [...prev.slice(-4), theta])
    },
    [fromThetaPx]
  )

  return (
    <div className="min-h-screen p-4 md:p-6" style={{ background: COLORS.bg }}>
      <h1 className="text-xl md:text-2xl font-semibold text-center text-white mb-4">
        The Argument Principle: Why Saddle Points Always Exist
      </h1>

      <div className="flex flex-col lg:flex-row gap-4 justify-center items-start max-w-6xl mx-auto">
        <div className="flex-shrink-0">
          <div className="text-sm text-gray-400 mb-1">θ-plane (domain)</div>
          <canvas
            ref={thetaCanvasRef}
            width={CANVAS_SIZE}
            height={CANVAS_SIZE}
            className="border border-gray-700 rounded cursor-crosshair"
            onMouseMove={handleThetaMouse}
            onMouseLeave={handleThetaLeave}
            onClick={handleThetaClick}
          />
          {hoverTheta && (
            <div className="mt-2 text-xs font-mono text-cyan-300">
              θ = {hoverTheta[0].toFixed(3)} + {hoverTheta[1].toFixed(3)}i
              {' → '}
              Φ′(θ) = {(phiPrime(hoverTheta, w, beta)[0]).toFixed(3)} + {(phiPrime(hoverTheta, w, beta)[1]).toFixed(3)}i
            </div>
          )}
        </div>
        <div className="flex-shrink-0">
          <div className="text-sm text-gray-400 mb-1">Φ′-plane (image)</div>
          <canvas
            ref={imageCanvasRef}
            width={CANVAS_SIZE}
            height={CANVAS_SIZE}
            className="border border-gray-700 rounded"
          />
        </div>
      </div>

      <div className="max-w-6xl mx-auto mt-6 space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 items-end">
          <div>
            <label className="block text-sm text-gray-400 mb-1">γ̃ [0.1, 8π]</label>
            <input
              type="range"
              min={0.1}
              max={8 * Math.PI}
              step={0.05}
              value={gammaTilde}
              onChange={(e) => setGammaTilde(Number(e.target.value))}
              disabled={gammaAnimating}
              className="w-full h-2 rounded accent-cyan-500"
            />
            <span className="text-xs font-mono text-gray-500">
              {(gammaTilde / Math.PI).toFixed(2)}π
            </span>
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">R [1, 15]</label>
            <input
              type="range"
              min={1}
              max={15}
              step={0.2}
              value={R}
              onChange={(e) => setR(Number(e.target.value))}
              className="w-full h-2 rounded accent-cyan-500"
            />
            <span className="text-xs font-mono text-gray-500">{R.toFixed(1)}</span>
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">β (rad)</label>
            <input
              type="range"
              min={-Math.PI + 0.1}
              max={-0.1}
              step={0.05}
              value={beta}
              onChange={(e) => setBeta(Number(e.target.value))}
              className="w-full h-2 rounded accent-cyan-500"
            />
            <span className="text-xs font-mono text-gray-500">
              {(beta / Math.PI).toFixed(2)}π
            </span>
          </div>
          <div className="flex flex-wrap gap-2 items-center">
            <button
              onClick={() => setPlaying((p) => !p)}
              className="px-4 py-2 rounded bg-cyan-600 hover:bg-cyan-500 text-white text-sm font-medium"
            >
              {playing ? 'Pause' : 'Play'}
            </button>
            <button
              onClick={() => {
                setGammaAnimating(true)
                if (gammaTilde > 0.5) setGammaTilde(0.1)
              }}
              disabled={gammaAnimating}
              className="px-4 py-2 rounded bg-orange-600 hover:bg-orange-500 text-white text-sm font-medium disabled:opacity-50"
            >
              Animate γ̃
            </button>
            <button
              onClick={() => setTestPoints([])}
              className="px-3 py-2 rounded border border-gray-500 text-gray-300 text-sm hover:bg-gray-800"
            >
              Clear points
            </button>
          </div>
        </div>
        <div>
          <label className="block text-sm text-gray-400 mb-1">Speed</label>
          <input
            type="range"
            min={0.25}
            max={3}
            step={0.25}
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            className="w-full max-w-xs h-2 rounded accent-cyan-500"
          />
        </div>
      </div>

      <div className="max-w-6xl mx-auto mt-6 p-4 rounded border border-gray-700 bg-gray-900/50">
        <p className="text-sm text-gray-300 leading-relaxed">{explanation}</p>
      </div>

      <div className="max-w-6xl mx-auto mt-4">
        <button
          onClick={() => setMathOpen((o) => !o)}
          className="text-sm text-cyan-400 hover:text-cyan-300"
        >
          {mathOpen ? '▼' : '▶'} Mathematical Details
        </button>
        {mathOpen && (
          <div className="mt-2 p-4 rounded border border-gray-700 bg-gray-900/30 font-mono text-sm text-gray-400 overflow-x-auto">
            <p>Φ′(θ) = −θ/2 + G(θ)</p>
            <p>G(θ) = −i w sin(β/2) e^{θw} / (cos(β/2) − i sin(β/2) e^{θw})</p>
            <p>w = √(−i γ̃ / 2)</p>
            <p>Poles: θ_k = (log(−i cot(β/2)) + 2πik) / w</p>
            <p>Good arcs: Re(e^{iφ}·w) ≤ −ε ⇒ Φ′(θ) ≈ −θ/2</p>
            <p>Argument principle: (1/2πi) ∮ Φ″/Φ′ dθ = N − P (zeros − poles)</p>
          </div>
        )}
      </div>
    </div>
  )
}
