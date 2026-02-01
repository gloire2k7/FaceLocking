# src/lock.py
"""
Face Locking Feature

Extends the face recognition system to:
- Lock onto a specific enrolled identity
- Track the locked face consistently
- Detect actions (movement, blinks, smiles)
- Record action history to files

Run:
python -m src.lock

Keys:
q : quit
l : lock/unlock (when target face is detected)
r : reload DB
+/- : adjust recognition threshold
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque
import cv2
import numpy as np
import onnxruntime as ort

try:
    import mediapipe as mp
except Exception as e:
    mp = None
    _MP_IMPORT_ERROR = e

from .haar_5pt import align_face_5pt
from .recognize import (
    HaarFaceMesh5pt,
    ArcFaceEmbedderONNX,
    FaceDBMatcher,
    load_db_npz,
    cosine_distance,
    FaceDet,
)


# -------------------------
# Action Detection
# -------------------------
@dataclass
class ActionEvent:
    timestamp: float
    action_type: str
    description: str
    value: Optional[float] = None


class ActionDetector:
    """Detects face actions: movement, blinks, smiles."""

    def __init__(self):
        if mp is None:
            raise RuntimeError(
                f"mediapipe import failed: {_MP_IMPORT_ERROR}\n"
                f"Install: pip install mediapipe==0.10.21"
            )
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # MediaPipe landmark indices for eye and mouth
        # Left eye landmarks (upper and lower)
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        # Right eye landmarks
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        # Mouth landmarks (for smile detection)
        self.MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]

        # Blink detection state
        self.eye_ar_history = deque(maxlen=5)  # Eye aspect ratio history
        self.blink_threshold = 0.25  # Threshold for blink detection
        self.eye_ar_normal = 0.30  # Normal eye aspect ratio

        # Smile detection state
        self.mouth_ar_history = deque(maxlen=5)  # Mouth aspect ratio history
        self.smile_threshold = 0.50  # Threshold for smile detection

    def _eye_aspect_ratio(self, landmarks, eye_indices, frame_w, frame_h):
        """Calculate Eye Aspect Ratio (EAR) for blink detection."""
        eye_points = []
        for idx in eye_indices:
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                eye_points.append([lm.x * frame_w, lm.y * frame_h])

        if len(eye_points) < 6:
            return None

        eye_points = np.array(eye_points)
        # Calculate vertical distances
        vertical_1 = np.linalg.norm(eye_points[1] - eye_points[5])
        vertical_2 = np.linalg.norm(eye_points[2] - eye_points[4])
        # Calculate horizontal distance
        horizontal = np.linalg.norm(eye_points[0] - eye_points[3])

        if horizontal == 0:
            return None

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

    def _mouth_aspect_ratio(self, landmarks, frame_w, frame_h):
        """Calculate Mouth Aspect Ratio (MAR) for smile detection."""
        mouth_points = []
        for idx in self.MOUTH_INDICES:
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                mouth_points.append([lm.x * frame_w, lm.y * frame_h])

        if len(mouth_points) < 6:
            return None

        mouth_points = np.array(mouth_points)
        # Calculate vertical distance (mouth opening)
        vertical = np.linalg.norm(mouth_points[2] - mouth_points[10])
        # Calculate horizontal distance (mouth width)
        horizontal = np.linalg.norm(mouth_points[0] - mouth_points[6])

        if horizontal == 0:
            return None

        mar = vertical / horizontal
        return mar

    def detect_actions(self, frame_bgr: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> List[ActionEvent]:
        """Detect actions from the face region."""
        x1, y1, x2, y2 = face_bbox
        # Expand ROI slightly for better landmark detection
        h, w = frame_bgr.shape[:2]
        margin = 20
        roi_x1 = max(0, x1 - margin)
        roi_y1 = max(0, y1 - margin)
        roi_x2 = min(w, x2 + margin)
        roi_y2 = min(h, y2 + margin)

        roi = frame_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            return []

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        actions = []
        timestamp = time.time()

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            roi_w, roi_h = roi_x2 - roi_x1, roi_y2 - roi_y1

            # Blink detection
            left_ear = self._eye_aspect_ratio(landmarks, self.LEFT_EYE_INDICES, roi_w, roi_h)
            right_ear = self._eye_aspect_ratio(landmarks, self.RIGHT_EYE_INDICES, roi_w, roi_h)

            if left_ear is not None and right_ear is not None:
                avg_ear = (left_ear + right_ear) / 2.0
                self.eye_ar_history.append(avg_ear)

                # Detect blink: EAR drops below threshold
                if len(self.eye_ar_history) >= 3:
                    if avg_ear < self.blink_threshold:
                        # Check if it was normal before (blink, not closed eyes)
                        if len(self.eye_ar_history) >= 2 and self.eye_ar_history[-2] > self.eye_ar_normal:
                            actions.append(ActionEvent(
                                timestamp=timestamp,
                                action_type="blink",
                                description="Eye blink detected",
                                value=avg_ear
                            ))

            # Smile detection
            mar = self._mouth_aspect_ratio(landmarks, roi_w, roi_h)
            if mar is not None:
                self.mouth_ar_history.append(mar)
                # Smile: MAR increases (mouth opens wider)
                if len(self.mouth_ar_history) >= 3:
                    avg_mar = np.mean(list(self.mouth_ar_history)[-3:])
                    if avg_mar > self.smile_threshold:
                        # Check if it increased from previous state
                        if len(self.mouth_ar_history) >= 4:
                            prev_avg = np.mean(list(self.mouth_ar_history)[-4:-1])
                            if avg_mar > prev_avg * 1.1:  # 10% increase
                                actions.append(ActionEvent(
                                    timestamp=timestamp,
                                    action_type="smile",
                                    description="Smile or laugh detected",
                                    value=avg_mar
                                ))

        return actions


# -------------------------
# Face Locking State
# -------------------------
@dataclass
class LockState:
    target_name: Optional[str] = None
    is_locked: bool = False
    locked_face_id: Optional[int] = None  # Track which face index is locked
    last_seen_time: float = 0.0
    lock_timeout: float = 2.0  # Release lock if face not seen for 2 seconds
    position_history: deque = field(default_factory=lambda: deque(maxlen=10))  # Track face center position
    last_position: Optional[Tuple[float, float]] = None
    movement_threshold: float = 30.0  # Pixels to consider movement

    def update_position(self, face_center: Tuple[float, float]):
        """Update face position and detect movement."""
        if self.last_position is not None:
            dx = face_center[0] - self.last_position[0]
            dy = face_center[1] - self.last_position[1]
            dist = np.sqrt(dx**2 + dy**2)

            if dist > self.movement_threshold:
                # Determine direction
                if abs(dx) > abs(dy):
                    direction = "left" if dx < 0 else "right"
                else:
                    direction = "up" if dy < 0 else "down"

                self.position_history.append((time.time(), direction, dist))
                self.last_position = face_center
                return direction, dist

        self.last_position = face_center
        self.position_history.append((time.time(), None, 0.0))
        return None, 0.0

    def get_recent_movement(self) -> Optional[Tuple[str, float]]:
        """Get the most recent significant movement."""
        if len(self.position_history) < 2:
            return None
        # Check last few entries for movement
        for entry in reversed(list(self.position_history)[-3:]):
            if entry[1] is not None:  # Has direction
                return entry[1], entry[2]  # direction, distance
        return None


# -------------------------
# Action History Recorder
# -------------------------
class ActionHistoryRecorder:
    """Records action history to files."""

    def __init__(self, output_dir: Path = Path("data/history")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_file: Optional[Path] = None
        self.file_handle = None

    def start_recording(self, face_name: str):
        """Start recording for a specific face."""
        self.stop_recording()  # Close any existing file

        timestamp = time.strftime("%Y%m%d%H%M%S")
        filename = f"{face_name.lower()}_history_{timestamp}.txt"
        self.current_file = self.output_dir / filename

        self.file_handle = open(self.current_file, "w", encoding="utf-8")
        self.file_handle.write(f"Face Locking History for: {face_name}\n")
        self.file_handle.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.file_handle.write("-" * 60 + "\n\n")

    def record_action(self, action: ActionEvent):
        """Record an action to the current file."""
        if self.file_handle is None:
            return

        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(action.timestamp))
        value_str = f" (value: {action.value:.3f})" if action.value is not None else ""
        line = f"{timestamp_str} | {action.action_type:10s} | {action.description}{value_str}\n"
        self.file_handle.write(line)
        self.file_handle.flush()

    def stop_recording(self):
        """Stop recording and close the file."""
        if self.file_handle is not None:
            self.file_handle.write(f"\nEnded: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.file_handle.close()
            self.file_handle = None
            if self.current_file:
                print(f"[lock] History saved to: {self.current_file}")


# -------------------------
# Main Face Locking System
# -------------------------
def main():
    db_path = Path("data/db/face_db.npz")
    
    # Load database
    db = load_db_npz(db_path)
    if not db:
        print("No enrolled faces found. Please run enrollment first (python -m src.enroll)")
        return

    # Select target face
    print("\nAvailable enrolled faces:")
    names = sorted(db.keys())
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}")
    
    while True:
        try:
            choice = input(f"\nSelect face to lock (1-{len(names)}) or name: ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(names):
                    target_name = names[idx]
                    break
            elif choice in names:
                target_name = choice
                break
            else:
                print(f"Invalid choice. Please enter 1-{len(names)} or a valid name.")
        except (ValueError, KeyboardInterrupt):
            print("\nExiting.")
            return

    print(f"\nLocking onto: {target_name}")
    print("Controls:")
    print("  q : quit")
    print("  l : lock/unlock manually")
    print("  r : reload database")
    print("  +/- : adjust recognition threshold")

    # Initialize components
    det = HaarFaceMesh5pt(min_size=(70, 70), debug=False)
    embedder = ArcFaceEmbedderONNX(
        model_path="models/embedder_arcface.onnx",
        input_size=(112, 112),
        debug=False,
    )
    matcher = FaceDBMatcher(db=db, dist_thresh=0.34)
    action_detector = ActionDetector()
    recorder = ActionHistoryRecorder()
    lock_state = LockState(target_name=target_name)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")

    t0 = time.time()
    frames = 0
    fps: Optional[float] = None
    manual_lock_toggle = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            faces = det.detect(frame, max_faces=5)
            vis = frame.copy()

            # FPS
            frames += 1
            dt = time.time() - t0
            if dt >= 1.0:
                fps = frames / dt
                frames = 0
                t0 = time.time()

            # Find target face
            target_face: Optional[FaceDet] = None
            target_match: Optional[Tuple[str, float, float, bool]] = None

            for i, f in enumerate(faces):
                aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                emb = embedder.embed(aligned)
                mr = matcher.match(emb)

                # Check if this is the target face
                if mr.accepted and mr.name == target_name:
                    target_face = f
                    target_match = (mr.name, mr.distance, mr.similarity, mr.accepted)
                    break

            # Locking logic
            current_time = time.time()

            if target_face is not None and target_match is not None:
                # Target face detected
                lock_state.last_seen_time = current_time

                # Auto-lock if not locked and confidence is high
                if not lock_state.is_locked and target_match[2] > 0.7:  # High similarity
                    lock_state.is_locked = True
                    lock_state.locked_face_id = 0  # Track first matching face
                    recorder.start_recording(target_name)
                    print(f"[lock] LOCKED onto {target_name}")

                # If locked, track this face
                if lock_state.is_locked:
                    # Calculate face center
                    face_center = (
                        (target_face.x1 + target_face.x2) / 2.0,
                        (target_face.y1 + target_face.y2) / 2.0,
                    )

                    # Detect movement
                    movement = lock_state.update_position(face_center)
                    if movement[0] is not None:
                        direction, distance = movement
                        if direction in ["left", "right"]:
                            action = ActionEvent(
                                timestamp=current_time,
                                action_type="movement",
                                description=f"Face moved {direction}",
                                value=distance,
                            )
                            recorder.record_action(action)

                    # Detect actions (blinks, smiles)
                    face_bbox = (target_face.x1, target_face.y1, target_face.x2, target_face.y2)
                    actions = action_detector.detect_actions(vis, face_bbox)
                    for action in actions:
                        recorder.record_action(action)

            else:
                # Target face not detected
                if lock_state.is_locked:
                    # Check timeout
                    if current_time - lock_state.last_seen_time > lock_state.lock_timeout:
                        lock_state.is_locked = False
                        lock_state.locked_face_id = None
                        recorder.stop_recording()
                        print(f"[lock] UNLOCKED (timeout)")

            # Manual lock toggle
            if manual_lock_toggle:
                if lock_state.is_locked:
                    lock_state.is_locked = False
                    lock_state.locked_face_id = None
                    recorder.stop_recording()
                    print(f"[lock] UNLOCKED (manual)")
                elif target_face is not None:
                    lock_state.is_locked = True
                    lock_state.locked_face_id = 0
                    recorder.start_recording(target_name)
                    print(f"[lock] LOCKED (manual)")
                manual_lock_toggle = False

            # Draw all faces
            for i, f in enumerate(faces):
                aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                emb = embedder.embed(aligned)
                mr = matcher.match(emb)

                # Determine if this is the locked face
                is_locked_face = (
                    lock_state.is_locked
                    and target_face is not None
                    and f.x1 == target_face.x1
                    and f.y1 == target_face.y1
                )

                # Color: locked=cyan, target=green, other=blue, unknown=red
                if is_locked_face:
                    color = (255, 255, 0)  # Cyan
                    thickness = 4
                elif mr.accepted and mr.name == target_name:
                    color = (0, 255, 0)  # Green
                    thickness = 3
                elif mr.accepted:
                    color = (255, 0, 0)  # Blue
                    thickness = 2
                else:
                    color = (0, 0, 255)  # Red
                    thickness = 2

                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, thickness)

                # Draw landmarks
                for x, y in f.kps.astype(int):
                    cv2.circle(vis, (int(x), int(y)), 2, color, -1)

                # Label
                label = mr.name if mr.name is not None else "Unknown"
                if is_locked_face:
                    label = f"[LOCKED] {label}"
                elif mr.accepted and mr.name == target_name:
                    label = f"[TARGET] {label}"

                cv2.putText(
                    vis,
                    label,
                    (f.x1, max(0, f.y1 - 28)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )
                cv2.putText(
                    vis,
                    f"sim={mr.similarity:.3f}",
                    (f.x1, max(0, f.y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            # Status overlay
            status_lines = [
                f"Target: {target_name}",
                f"Locked: {'YES' if lock_state.is_locked else 'NO'}",
                f"DB: {len(matcher._names)} identities",
                f"Threshold: {matcher.dist_thresh:.2f}",
            ]
            if fps is not None:
                status_lines.append(f"FPS: {fps:.1f}")

            y_offset = 30
            for line in status_lines:
                cv2.putText(
                    vis,
                    line,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                y_offset += 25

            if lock_state.is_locked:
                cv2.putText(
                    vis,
                    "*** LOCKED ***",
                    (vis.shape[1] - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    3,
                )

            cv2.imshow("Face Locking", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("l"):
                manual_lock_toggle = True
            elif key == ord("r"):
                db = load_db_npz(db_path)
                matcher.reload_from(db_path)
                print(f"[lock] Reloaded DB: {len(matcher._names)} identities")
            elif key in (ord("+"), ord("=")):
                matcher.dist_thresh = float(min(1.20, matcher.dist_thresh + 0.01))
                print(f"[lock] Threshold: {matcher.dist_thresh:.2f}")
            elif key == ord("-"):
                matcher.dist_thresh = float(max(0.05, matcher.dist_thresh - 0.01))
                print(f"[lock] Threshold: {matcher.dist_thresh:.2f}")

    finally:
        recorder.stop_recording()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

