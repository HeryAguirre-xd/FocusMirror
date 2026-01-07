# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Focus Mirror is an ambient biometric feedback app that uses Computer Vision to map "Focus State" to a generative minimalist frontend. When the user loses focus, the environment subtly degrades; when focus returns, clarity blooms back.

**Platform:** Browser-based, all vision processing local (no cloud).

## Tech Stack

- **Frontend:** React, Vite, Three.js (React Three Fiber), Tailwind CSS
- **Backend:** Python 3.11+, FastAPI, WebSockets
- **Vision:** OpenCV, MediaPipe (Face Mesh/Iris Tracking)
- **Communication:** Bi-directional WebSockets for real-time `focus_score` streaming

## Commands

### Backend
```bash
pip install -r backend/requirements.txt     # Install dependencies
python backend/main.py                       # Run WebSocket server
python backend/vision_engine.py --debug      # Test vision with camera overlay
python backend/vision_engine.py --mock       # Run with synthetic focus data
```

### Frontend
```bash
npm install          # Install dependencies
npm run dev          # Development server
npm run build        # Production build
```

## Focus Score Logic

The `focus_score` (0.0 - 1.0) is calculated from:
1. **Gaze:** Eyes directed toward screen (Iris tracking)
2. **Posture:** Head centered and upright (Face geometry)
3. **Engagement:** Blink frequency, jaw tension (Stress indicators)

### Temporal Behavior
- **3-second grace period** before degradation begins (allows looking away to think)
- **Exponential smoothing** on focus_score (slow to penalize, slow to reward)
- Brief glances away = no penalty; sustained distraction = gradual fade

## Visual Design

**Aesthetic:** Apple-level minimalism (Ive/Jobs approved). Restrained, intentional, calm.

### Color
- Warm amber/gold (focused) → cool blue-gray (unfocused)
- Near-monochromatic palette, one shifting accent color

### States
- **Focused:** Calm breathing animation, warm light, clean geometry
- **Degradation:** Slow desaturation, cooling, film grain appears—entropy, not chaos
- **Recovery:** Gentle "clarity bloom" (~2s transition), warmth seeps back like sunlight through clouds

### Motion
- Slow, breathing animations
- All transitions use LERP to avoid flickering
- Nothing demands attention—it just exists

## UI Patterns

- **Main canvas:** Pure, no buttons or chrome
- **Settings:** `Space` key reveals hover-to-reveal settings panel
- **Session stats:** `Escape` key shows minimal overlay:
  - Total session time
  - Time in focus (percentage)
  - Thin timeline bar showing focus/unfocus periods
  - Auto-dismisses after 5 seconds
- **On exit:** Brief toast with session summary (optional)

## Coding Patterns

- **Smoothing:** Always LERP focus_score transitions
- **Performance:** OpenCV loop under 30ms; use threading for camera feed if needed
- **State management:** WebSocket connection via custom `useFocusSocket` hook
- **Mock mode:** `--mock` flag for frontend development without camera

## Roadmap

- [ ] Phase 1: Python vision engine + WebSocket server
- [ ] Phase 2: React frontend with Three.js shader responding to score
- [ ] Phase 3: Refine focus logic (head tilt, blink detection)
- [ ] Phase 4: Aesthetic tuning (grain, noise, color palettes)
