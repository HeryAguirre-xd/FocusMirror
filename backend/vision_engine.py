"""
Focus Mirror Vision Engine
Uses MediaPipe Face Landmarker for gaze tracking and focus detection.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import argparse
import urllib.request
import os
from dataclasses import dataclass
from typing import Optional


# Model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def download_model():
    """Download the face landmarker model if not present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading face landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


@dataclass
class FocusMetrics:
    """Raw metrics before smoothing."""
    gaze_score: float  # 0-1: how centered the gaze is
    posture_score: float  # 0-1: how upright/centered the head is
    face_detected: bool
    timestamp: float
    # Head position for interactive orb (normalized 0-1)
    head_x: float = 0.5  # Horizontal position (0=left, 1=right)
    head_y: float = 0.5  # Vertical position (0=top, 1=bottom)
    head_tilt: float = 0.0  # Head tilt in degrees (-45 to 45)


class FocusEngine:
    """
    Processes webcam frames to calculate focus_score.

    Uses MediaPipe Face Landmarker for:
    - Iris tracking (gaze direction)
    - Head pose estimation (posture)
    """

    # Face Landmarker indices (based on MediaPipe face mesh)
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

        # Download model if needed
        download_model()

        # MediaPipe Face Landmarker setup
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

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
        try:
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
        except (IndexError, AttributeError):
            return 0.5

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
        try:
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
            vertical_ratio = (chin.y - forehead.y)
            pitch_deviation = abs(vertical_ratio - 0.2) * 3
            pitch_score = 1.0 - min(pitch_deviation, 1.0)

            # Combine scores with weights
            posture_score = (
                roll_score * 0.3 +
                horizontal_score * 0.4 +
                pitch_score * 0.3
            )

            return np.clip(posture_score, 0, 1)
        except (IndexError, AttributeError):
            return 0.5

    def _calculate_head_position(self, landmarks) -> tuple[float, float, float]:
        """
        Calculate head position and tilt for interactive orb.
        Returns (head_x, head_y, head_tilt) all normalized.
        """
        try:
            nose = landmarks[self.NOSE_TIP]
            left_ear = landmarks[self.LEFT_EAR]
            right_ear = landmarks[self.RIGHT_EAR]

            # Head position (nose tip position, normalized 0-1)
            head_x = nose.x
            head_y = nose.y

            # Head tilt (roll) based on ear positions
            ear_dy = left_ear.y - right_ear.y
            ear_dx = right_ear.x - left_ear.x
            tilt_radians = np.arctan2(ear_dy, ear_dx)
            head_tilt = np.degrees(tilt_radians)

            return head_x, head_y, np.clip(head_tilt, -45, 45)
        except (IndexError, AttributeError):
            return 0.5, 0.5, 0.0

    def process_frame(self, frame: np.ndarray) -> FocusMetrics:
        """
        Process a single frame and return raw focus metrics.
        """
        img_height, img_width = frame.shape[:2]

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect face landmarks
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            return FocusMetrics(
                gaze_score=0.0,
                posture_score=0.0,
                face_detected=False,
                timestamp=time.time()
            )

        landmarks = result.face_landmarks[0]

        gaze_score = self._calculate_gaze_score(landmarks, img_width)
        posture_score = self._calculate_posture_score(landmarks, img_width, img_height)
        head_x, head_y, head_tilt = self._calculate_head_position(landmarks)

        return FocusMetrics(
            gaze_score=gaze_score,
            posture_score=posture_score,
            face_detected=True,
            timestamp=time.time(),
            head_x=head_x,
            head_y=head_y,
            head_tilt=head_tilt
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
        h, w = frame.shape[:2]

        # Semi-transparent background for text
        overlay = frame.copy()
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
        self.landmarker.close()


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
    print("Press Ctrl+C to quit")
    print("-" * 50)

    # Open camera (use camera 1 on this Mac)
    print("Opening camera...")
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Give camera time to warm up - macOS needs this
    print("Warming up camera (this may take a few seconds)...")
    time.sleep(2)

    # Try reading frames until we get one
    for attempt in range(50):
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"Camera ready! (took {attempt + 1} attempts)")
            break
        time.sleep(0.1)
    else:
        print("Error: Could not read from camera after 50 attempts")
        cap.release()
        return

    engine = FocusEngine()
    frame_count = 0
    failed_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                failed_frames += 1
                if failed_frames > 30:
                    print("\nError: Could not read frames from camera")
                    break
                time.sleep(0.1)
                continue

            failed_frames = 0

            # Mirror the frame for more intuitive feedback
            frame = cv2.flip(frame, 1)

            # Process frame
            metrics = engine.process_frame(frame)
            focus_score = engine.get_focus_score(metrics)

            # Print metrics to terminal
            frame_count += 1
            if frame_count % 10 == 0:  # Print every 10 frames
                grace = "(grace)" if engine.is_grace_active else ""
                face = "YES" if metrics.face_detected else "NO "
                print(f"\rFace: {face} | Gaze: {metrics.gaze_score:.2f} | Posture: {metrics.posture_score:.2f} | Focus: {focus_score:.2f} {grace}   ", end="", flush=True)

            # Try to show window (may not work on all systems)
            try:
                frame = engine.draw_debug_overlay(frame, metrics, focus_score)
                cv2.imshow("Focus Mirror - Debug", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception:
                pass  # Window display not available

    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
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
