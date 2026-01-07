"""
Focus Mirror WebSocket Server
Streams real-time focus_score to connected clients.
"""

import asyncio
import json
import time
import argparse
import threading
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from vision_engine import FocusEngine, MockFocusEngine, FocusMetrics


class FocusServer:
    """
    Manages the vision engine and broadcasts focus updates to WebSocket clients.
    """

    def __init__(self, mock_mode: bool = False):
        self.mock_mode = mock_mode
        self.clients: list[WebSocket] = []
        self.running = False
        self._lock = threading.Lock()

        # Focus state
        self.current_score: float = 1.0
        self.raw_score: float = 1.0
        self.is_grace_active: bool = False
        self.face_detected: bool = False

        # Session tracking
        self.session_start: float = 0
        self.focused_time: float = 0
        self.focus_history: list[tuple[float, float]] = []  # (timestamp, score)
        self._last_update: float = 0
        self._was_focused: bool = True

        # Vision engine (initialized on start)
        self._engine: Optional[FocusEngine] = None
        self._mock_engine: Optional[MockFocusEngine] = None
        self._capture: Optional[cv2.VideoCapture] = None
        self._vision_thread: Optional[threading.Thread] = None

    def start(self):
        """Start the vision processing."""
        self.running = True
        self.session_start = time.time()
        self._last_update = self.session_start

        if self.mock_mode:
            self._mock_engine = MockFocusEngine()
            print("Focus Server started in MOCK mode")
        else:
            self._engine = FocusEngine()
            self._capture = cv2.VideoCapture(0)
            if not self._capture.isOpened():
                raise RuntimeError("Could not open camera")
            print("Focus Server started with camera")

        # Start vision processing in background thread
        self._vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self._vision_thread.start()

    def stop(self):
        """Stop the vision processing."""
        self.running = False
        if self._vision_thread:
            self._vision_thread.join(timeout=2.0)
        if self._capture:
            self._capture.release()
        if self._engine:
            self._engine.release()
        print("Focus Server stopped")

    def _vision_loop(self):
        """Background thread for vision processing."""
        target_fps = 30
        frame_time = 1.0 / target_fps

        while self.running:
            start = time.time()

            if self.mock_mode:
                score = self._mock_engine.get_focus_score()
                self._update_state(score, score, False, True)
            else:
                ret, frame = self._capture.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    metrics = self._engine.process_frame(frame)
                    score = self._engine.get_focus_score(metrics)
                    self._update_state(
                        score,
                        self._engine.raw_score,
                        self._engine.is_grace_active,
                        metrics.face_detected
                    )

            # Maintain target framerate
            elapsed = time.time() - start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    def _update_state(
        self,
        score: float,
        raw_score: float,
        grace_active: bool,
        face_detected: bool
    ):
        """Update focus state and track session metrics."""
        current_time = time.time()

        with self._lock:
            self.current_score = score
            self.raw_score = raw_score
            self.is_grace_active = grace_active
            self.face_detected = face_detected

            # Track focused time
            dt = current_time - self._last_update
            is_focused = score >= 0.5

            if self._was_focused:
                self.focused_time += dt

            self._was_focused = is_focused
            self._last_update = current_time

            # Record history point every second
            if not self.focus_history or current_time - self.focus_history[-1][0] >= 1.0:
                self.focus_history.append((current_time, score))
                # Keep last 30 minutes of history
                cutoff = current_time - 1800
                self.focus_history = [(t, s) for t, s in self.focus_history if t >= cutoff]

    def get_state(self) -> dict:
        """Get current focus state as dictionary."""
        with self._lock:
            session_duration = time.time() - self.session_start
            focus_percentage = (self.focused_time / session_duration * 100) if session_duration > 0 else 100

            return {
                "focus_score": round(self.current_score, 3),
                "raw_score": round(self.raw_score, 3),
                "grace_active": self.is_grace_active,
                "face_detected": self.face_detected,
                "session": {
                    "duration": round(session_duration, 1),
                    "focused_time": round(self.focused_time, 1),
                    "focus_percentage": round(focus_percentage, 1)
                },
                "timestamp": time.time()
            }

    def get_session_summary(self) -> dict:
        """Get session summary for stats overlay."""
        with self._lock:
            session_duration = time.time() - self.session_start
            focus_percentage = (self.focused_time / session_duration * 100) if session_duration > 0 else 100

            # Downsample history for timeline (max 60 points)
            timeline = []
            if self.focus_history:
                step = max(1, len(self.focus_history) // 60)
                timeline = [
                    {"t": round(t - self.session_start, 1), "score": round(s, 2)}
                    for t, s in self.focus_history[::step]
                ]

            return {
                "duration": round(session_duration, 1),
                "focused_time": round(self.focused_time, 1),
                "focus_percentage": round(focus_percentage, 1),
                "timeline": timeline
            }

    async def register(self, websocket: WebSocket):
        """Register a new WebSocket client."""
        await websocket.accept()
        self.clients.append(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")

    def unregister(self, websocket: WebSocket):
        """Unregister a WebSocket client."""
        if websocket in self.clients:
            self.clients.remove(websocket)
        print(f"Client disconnected. Total clients: {len(self.clients)}")


# Global server instance
server: Optional[FocusServer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage server lifecycle."""
    global server
    if server:
        server.start()
    yield
    if server:
        server.stop()


app = FastAPI(title="Focus Mirror", lifespan=lifespan)

# CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "running", "mock_mode": server.mock_mode if server else False}


@app.get("/session")
async def get_session():
    """Get session summary."""
    if not server:
        return {"error": "Server not initialized"}
    return server.get_session_summary()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time focus updates.

    Sends JSON messages at ~30fps:
    {
        "focus_score": 0.85,
        "raw_score": 0.82,
        "grace_active": false,
        "face_detected": true,
        "session": {
            "duration": 120.5,
            "focused_time": 95.2,
            "focus_percentage": 79.0
        },
        "timestamp": 1234567890.123
    }
    """
    if not server:
        await websocket.close(code=1011, reason="Server not initialized")
        return

    await server.register(websocket)

    try:
        while True:
            # Send current state
            state = server.get_state()
            await websocket.send_json(state)

            # Check for incoming messages (for future bidirectional features)
            try:
                # Non-blocking receive with short timeout
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=0.033  # ~30fps
                )
                # Handle client messages if needed
                data = json.loads(message)
                if data.get("type") == "get_session":
                    summary = server.get_session_summary()
                    await websocket.send_json({"type": "session", "data": summary})
            except asyncio.TimeoutError:
                pass  # No message received, continue

    except WebSocketDisconnect:
        server.unregister(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        server.unregister(websocket)


def main():
    """Entry point."""
    global server

    parser = argparse.ArgumentParser(description="Focus Mirror Server")
    parser.add_argument("--mock", action="store_true", help="Use synthetic focus data")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()

    server = FocusServer(mock_mode=args.mock)

    import uvicorn
    print(f"\nStarting Focus Mirror Server")
    print(f"  Mode: {'Mock' if args.mock else 'Camera'}")
    print(f"  WebSocket: ws://{args.host}:{args.port}/ws")
    print(f"  Health: http://{args.host}:{args.port}/")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
