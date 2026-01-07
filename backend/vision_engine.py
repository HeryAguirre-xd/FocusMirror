"""
Focus Mirror Vision Engine
Uses MediaPipe Face Mesh for gaze tracking and focus detection.
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import argparse
from dataclasses import dataclass
from typing import Optional, Callable
import math


@dataclass
class FocusMetrics:
    """Raw metrics before smoothing."""
    gaze_score: float  # 0-1: how centered the gaze is
    posture_score: float  # 0-1: how upright/centered the head is
    face_detected: bool
    timestamp: float


class FocusEngine:
    """
    Processes webcam frames to calculate focus_score.

    Uses MediaPipe Face Mesh for:
    - Iris tracking (gaze direction)
    - Head pose estimation (posture)
    """

    # MediaPipe Face Mesh landmark indices
    # Left eye
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    LEFT_IRIS_CENTER = 468

    # Right eye
    RIGHT_EYE_OUTER = 362
    RIGHT_EYE_INNER = 263
    RIGHT_IRIS_CENTER = 473

    # Face reference points for head pose
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EAR = 234
    RIGHT_EAR = 454
    FOREHEAD = 10

    def __init__(
        self,
        smoothing_factor: float = 0.15,
        grace_period: float = 3.0,
        unfocus_threshold: float = 0.4
    ):
        """
        Args:
            smoothing_factor: EMA alpha (lower = smoother, 0.1-0.2 recommended)
            grace_period: Seconds before degradation begins
            unfocus_threshold: Raw score below this triggers grace period
        """
        self.smoothing_factor = smoothing_factor
        self.grace_period = grace_period
        self.unfocus_threshold = unfocus_threshold

        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Enables iris tracking
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # State
        self._smoothed_score = 1.0  # Start focused
        self._last_focused_time = time.time()
        self._grace_active = False
        self._raw_score = 1.0

    def _get_iris_position(
        self,
        landmarks,
        eye_outer_idx: int,
        eye_inner_idx: int,
        iris_center_idx: int,
        img_width: int
    ) -> float:
        """
        Calculate normalized iris position within eye (0 = inner, 1 = outer).
        Returns 0.5 when looking straight ahead.
        """
        eye_outer = landmarks[eye_outer_idx]
        eye_inner = landmarks[eye_inner_idx]
        iris = landmarks[iris_center_idx]

        # Get x coordinates in pixel space
        outer_x = eye_outer.x * img_width
        inner_x = eye_inner.x * img_width
        iris_x = iris.x * img_width

        # Normalize iris position within eye bounds
        eye_width = abs(outer_x - inner_x)
        if eye_width < 1:
            return 0.5

        # Calculate relative position
        if outer_x > inner_x:  # Left eye
            position = (iris_x - inner_x) / eye_width
        else:  # Right eye
            position = (outer_x - iris_x) / eye_width

        return np.clip(position, 0, 1)

    def _calculate_gaze_score(self, landmarks, img_width: int) -> float:
        """
        Calculate how centered the gaze is (1.0 = looking at screen).
        """
        # Get iris positions for both eyes
        left_pos = self._get_iris_position(
            landmarks,
            self.LEFT_EYE_OUTER,
            self.LEFT_EYE_INNER,
            self.LEFT_IRIS_CENTER,
            img_width
        )
        right_pos = self._get_iris_position(
            landmarks,
            self.RIGHT_EYE_OUTER,
            self.RIGHT_EYE_INNER,
            self.RIGHT_IRIS_CENTER,
            img_width
        )

        # Average both eyes
        avg_pos = (left_pos + right_pos) / 2

        # Convert to centered score (0.5 = perfect = 1.0 score)
        # Deviation from center reduces score
        deviation = abs(avg_pos - 0.5) * 2  # 0-1 range
        gaze_score = 1.0 - deviation

        return np.clip(gaze_score, 0, 1)

    def _calculate_posture_score(self, landmarks, img_width: int, img_height: int) -> float:
        """
        Calculate head posture score based on head tilt and position.
        """
        # Get key landmarks
        nose = landmarks[self.NOSE_TIP]
        left_ear = landmarks[self.LEFT_EAR]
        right_ear = landmarks[self.RIGHT_EAR]
        forehead = landmarks[self.FOREHEAD]
        chin = landmarks[self.CHIN]

        # Check horizontal tilt (roll) - ears should be level
        ear_dy = abs(left_ear.y - right_ear.y)
        roll_score = 1.0 - min(ear_dy * 5, 1.0)  # Penalize tilt

        # Check if face is centered horizontally
        face_center_x = nose.x
        center_deviation_x = abs(face_center_x - 0.5) * 2
        horizontal_score = 1.0 - min(center_deviation_x, 1.0)

        # Check vertical head tilt (pitch) using nose-forehead angle
        # When looking down, chin moves up relative to nose
        vertical_ratio = (chin.y - forehead.y)
        # Normal ratio is around 0.15-0.25, looking down increases it
        pitch_deviation = abs(vertical_ratio - 0.2) * 3
        pitch_score = 1.0 - min(pitch_deviation, 1.0)

        # Combine scores with weights
        posture_score = (
            roll_score * 0.3 +
            horizontal_score * 0.4 +
            pitch_score * 0.3
        )

        return np.clip(posture_score, 0, 1)

    def process_frame(self, frame: np.ndarray) -> FocusMetrics:
        """
        Process a single frame and return raw focus metrics.
        """
        img_height, img_width = frame.shape[:2]

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return FocusMetrics(
                gaze_score=0.0,
                posture_score=0.0,
                face_detected=False,
                timestamp=time.time()
            )

        landmarks = results.multi_face_landmarks[0].landmark

        gaze_score = self._calculate_gaze_score(landmarks, img_width)
        posture_score = self._calculate_posture_score(landmarks, img_width, img_height)

        return FocusMetrics(
            gaze_score=gaze_score,
            posture_score=posture_score,
            face_detected=True,
            timestamp=time.time()
        )

    def get_focus_score(self, metrics: FocusMetrics) -> float:
        """
        Calculate smoothed focus score with grace period logic.

        Returns a value 0.0-1.0 representing current focus state.
        """
        current_time = metrics.timestamp

        if not metrics.face_detected:
            # No face = unfocused, but apply grace period
            raw_score = 0.0
        else:
            # Combine gaze and posture (gaze weighted higher)
            raw_score = metrics.gaze_score * 0.7 + metrics.posture_score * 0.3

        self._raw_score = raw_score

        # Grace period logic
        if raw_score >= self.unfocus_threshold:
            # User is focused - reset grace timer
            self._last_focused_time = current_time
            self._grace_active = False
            target_score = raw_score
        else:
            # User may be unfocused
            time_since_focus = current_time - self._last_focused_time

            if time_since_focus < self.grace_period:
                # Within grace period - maintain current smoothed score
                self._grace_active = True
                target_score = self._smoothed_score
            else:
                # Grace period expired - allow degradation
                self._grace_active = False
                target_score = raw_score

        # Exponential moving average smoothing
        self._smoothed_score = (
            self.smoothing_factor * target_score +
            (1 - self.smoothing_factor) * self._smoothed_score
        )

        return np.clip(self._smoothed_score, 0, 1)

    @property
    def is_grace_active(self) -> bool:
        """Whether we're currently in the grace period."""
        return self._grace_active

    @property
    def raw_score(self) -> float:
        """The unsmoothed focus score."""
        return self._raw_score

    def draw_debug_overlay(self, frame: np.ndarray, metrics: FocusMetrics, focus_score: float) -> np.ndarray:
        """
        Draw debug visualization on frame.
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Semi-transparent background for text
        cv2.rectangle(overlay, (10, 10), (300, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Colors
        green = (0, 255, 0)
        yellow = (0, 255, 255)
        red = (0, 0, 255)
        white = (255, 255, 255)

        def score_color(score):
            if score > 0.7:
                return green
            elif score > 0.4:
                return yellow
            return red

        # Draw metrics
        y = 35
        cv2.putText(frame, f"Face: {'Yes' if metrics.face_detected else 'No'}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 1)

        y += 25
        cv2.putText(frame, f"Gaze: {metrics.gaze_score:.2f}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_color(metrics.gaze_score), 1)

        y += 25
        cv2.putText(frame, f"Posture: {metrics.posture_score:.2f}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_color(metrics.posture_score), 1)

        y += 25
        grace_text = " (grace)" if self.is_grace_active else ""
        cv2.putText(frame, f"Focus: {focus_score:.2f}{grace_text}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_color(focus_score), 1)

        # Focus bar at bottom
        bar_height = 10
        bar_y = h - 20
        bar_width = int(w * 0.8)
        bar_x = int(w * 0.1)

        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)

        # Focus level bar
        fill_width = int(bar_width * focus_score)
        color = score_color(focus_score)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)

        return frame

    def release(self):
        """Clean up resources."""
        self.face_mesh.close()


class MockFocusEngine:
    """
    Generates synthetic focus scores for frontend development.
    Simulates realistic focus patterns with occasional dips.
    """

    def __init__(self):
        self._score = 0.8
        self._target = 0.8
        self._last_change = time.time()
        self._change_interval = 5.0  # Change target every ~5 seconds

    def get_focus_score(self) -> float:
        """Generate synthetic focus score."""
        current_time = time.time()

        # Periodically change target
        if current_time - self._last_change > self._change_interval:
            # Usually focused, occasionally unfocused
            if np.random.random() < 0.7:
                self._target = np.random.uniform(0.7, 1.0)
            else:
                self._target = np.random.uniform(0.2, 0.5)
            self._last_change = current_time
            self._change_interval = np.random.uniform(3.0, 8.0)

        # Smooth transition to target
        self._score += (self._target - self._score) * 0.05

        # Add small noise
        self._score += np.random.uniform(-0.02, 0.02)

        return np.clip(self._score, 0, 1)


def run_debug_mode():
    """Run vision engine with debug overlay."""
    print("Starting Focus Mirror Vision Engine (Debug Mode)")
    print("Press 'q' to quit")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    engine = FocusEngine()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror the frame for more intuitive feedback
            frame = cv2.flip(frame, 1)

            # Process frame
            metrics = engine.process_frame(frame)
            focus_score = engine.get_focus_score(metrics)

            # Draw debug overlay
            frame = engine.draw_debug_overlay(frame, metrics, focus_score)

            cv2.imshow("Focus Mirror - Debug", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        engine.release()


def run_mock_mode():
    """Run with synthetic focus data (no camera)."""
    print("Starting Focus Mirror Vision Engine (Mock Mode)")
    print("Press Ctrl+C to stop")

    engine = MockFocusEngine()

    try:
        while True:
            score = engine.get_focus_score()
            print(f"\rFocus Score: {score:.3f}", end="", flush=True)
            time.sleep(0.033)  # ~30fps
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Focus Mirror Vision Engine")
    parser.add_argument("--debug", action="store_true", help="Run with debug overlay")
    parser.add_argument("--mock", action="store_true", help="Run with synthetic data")
    args = parser.parse_args()

    if args.mock:
        run_mock_mode()
    elif args.debug:
        run_debug_mode()
    else:
        print("Usage: python vision_engine.py --debug or --mock")
