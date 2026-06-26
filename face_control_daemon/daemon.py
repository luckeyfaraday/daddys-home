from __future__ import annotations

import argparse
import importlib
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from gesture_control_daemon.daemon import (
    ActionBackend,
    DryRunBackend,
    PyAutoGuiBackend,
    average_points as hand_center_point,
    clamp,
    missing_required_modules,
    parse_camera_source,
    prepare_mediapipe_vision_import,
    shorten,
)


Point = Tuple[float, float]


LOGGER = logging.getLogger("face_control_daemon")

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
MOUTH_LEFT = 78
MOUTH_RIGHT = 308
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
NOSE_TIP = 1
# Wrist plus the four finger MCP joints. Their average is a stable palm center
# that tracks the whole hand regardless of which fingers are extended.
HAND_CENTER_LANDMARKS = (0, 5, 9, 13, 17)


@dataclass(frozen=True)
class FaceMetrics:
    left_ear: float
    right_ear: float
    mouth_ratio: float
    roll_degrees: float
    nose_point: Point


@dataclass(frozen=True)
class FaceGesture:
    name: str
    confidence: float
    metrics: Optional[FaceMetrics] = None
    cursor_point: Optional[Point] = None


@dataclass
class RuntimeConfig:
    camera: str
    start_enabled: bool
    dry_run: bool
    debug_overlay: bool
    enable_tilt_scroll: bool
    smoothing: float
    cursor_deadzone: float
    cursor_gain: float
    click_cooldown: float
    wink_hold_seconds: float
    drag_hold_seconds: float
    min_confidence: float
    wink_threshold: float
    eye_open_threshold: float
    wink_gap: float
    mouth_open_threshold: float
    tilt_scroll_threshold: float
    scroll_interval: float
    scroll_scale: float
    scroll_max_step: int


@dataclass
class RuntimeState:
    control_enabled: bool = False
    active_gesture: str = "none"
    gesture_started_at: float = field(default_factory=time.monotonic)
    drag_active: bool = False
    click_armed: bool = True
    last_click_at: float = 0.0
    last_scroll_at: float = 0.0
    cursor_pos: Optional[Point] = None
    last_hand_point: Optional[Point] = None
    cursor_velocity: Point = (0.0, 0.0)
    hand_present: bool = False
    debug_overlay: bool = False
    last_event: str = "starting"
    last_event_at: float = field(default_factory=time.monotonic)

    def set_event(self, event: str) -> None:
        self.last_event = event
        self.last_event_at = time.monotonic()
        LOGGER.info(event)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safety-first daemon: hand moves the cursor, face triggers actions."
    )
    parser.add_argument(
        "--camera",
        default="0",
        help="OpenCV camera index or capture URL/path. Default: 0.",
    )
    parser.add_argument(
        "--start-enabled",
        action="store_true",
        help="Start with control mode enabled. Default is disabled.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run vision and overlay without sending mouse or keyboard events.",
    )
    parser.add_argument(
        "--debug-overlay",
        action="store_true",
        help="Start with expanded face diagnostics visible. Can also toggle with D.",
    )
    parser.add_argument(
        "--enable-tilt-scroll",
        action="store_true",
        help="Allow head tilt left/right to scroll. Default is disabled.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.35,
        help="Cursor smoothing alpha from 0.05 to 1.0. Default: 0.35.",
    )
    parser.add_argument(
        "--cursor-deadzone",
        type=float,
        default=2.0,
        help="Ignore per-frame cursor motion below this many screen pixels. Default: 2.",
    )
    parser.add_argument(
        "--cursor-gain",
        type=float,
        default=1.5,
        help="Relative cursor speed: hand motion times this maps to screen pixels. Default: 1.5.",
    )
    parser.add_argument(
        "--click-cooldown",
        type=float,
        default=0.65,
        help="Minimum seconds between wink clicks. Default: 0.65.",
    )
    parser.add_argument(
        "--wink-hold-seconds",
        type=float,
        default=0.16,
        help="Wink dwell before click. Default: 0.16.",
    )
    parser.add_argument(
        "--drag-hold-seconds",
        type=float,
        default=0.35,
        help="Mouth-open dwell before drag starts. Default: 0.35.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.65,
        help="Minimum gesture confidence for actions. Default: 0.65.",
    )
    parser.add_argument(
        "--wink-threshold",
        type=float,
        default=0.18,
        help="Eye-aspect ratio below which an eye counts as closed. Default: 0.18.",
    )
    parser.add_argument(
        "--eye-open-threshold",
        type=float,
        default=0.23,
        help="Eye-aspect ratio above which the other eye must remain open. Default: 0.23.",
    )
    parser.add_argument(
        "--wink-gap",
        type=float,
        default=0.07,
        help="Minimum eye-aspect-ratio gap between eyes for wink. Default: 0.07.",
    )
    parser.add_argument(
        "--mouth-open-threshold",
        type=float,
        default=0.32,
        help="Mouth open ratio required for drag. Default: 0.32.",
    )
    parser.add_argument(
        "--tilt-scroll-threshold",
        type=float,
        default=16.0,
        help="Head roll degrees required for tilt scroll. Default: 16.",
    )
    parser.add_argument(
        "--scroll-interval",
        type=float,
        default=0.12,
        help="Minimum seconds between tilt-scroll events. Default: 0.12.",
    )
    parser.add_argument(
        "--scroll-scale",
        type=float,
        default=0.35,
        help="Tilt-scroll sensitivity multiplier. Default: 0.35.",
    )
    parser.add_argument(
        "--scroll-max-step",
        type=int,
        default=4,
        help="Maximum scroll units emitted per tilt event. Default: 4.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    config = RuntimeConfig(
        camera=args.camera,
        start_enabled=args.start_enabled,
        dry_run=args.dry_run,
        debug_overlay=args.debug_overlay,
        enable_tilt_scroll=args.enable_tilt_scroll,
        smoothing=clamp(args.smoothing, 0.05, 1.0),
        cursor_deadzone=max(0.0, args.cursor_deadzone),
        cursor_gain=max(0.1, args.cursor_gain),
        click_cooldown=max(0.0, args.click_cooldown),
        wink_hold_seconds=max(0.0, args.wink_hold_seconds),
        drag_hold_seconds=max(0.0, args.drag_hold_seconds),
        min_confidence=clamp(args.min_confidence, 0.0, 1.0),
        wink_threshold=clamp(args.wink_threshold, 0.01, 1.0),
        eye_open_threshold=clamp(args.eye_open_threshold, 0.01, 1.0),
        wink_gap=max(0.0, args.wink_gap),
        mouth_open_threshold=max(0.0, args.mouth_open_threshold),
        tilt_scroll_threshold=max(0.0, args.tilt_scroll_threshold),
        scroll_interval=max(0.0, args.scroll_interval),
        scroll_scale=max(0.01, args.scroll_scale),
        scroll_max_step=max(1, args.scroll_max_step),
    )

    try:
        return run(config)
    except KeyboardInterrupt:
        LOGGER.info("stopped by keyboard interrupt")
        return 130


def run(config: RuntimeConfig) -> int:
    missing = missing_required_modules(("cv2", "mediapipe"))
    if missing:
        LOGGER.error(
            "Missing required vision dependencies: %s. Install with: pip install -r requirements.txt",
            ", ".join(missing),
        )
        return 2

    cv2 = importlib.import_module("cv2")

    backend = build_backend(config.dry_run)
    state = RuntimeState(
        control_enabled=config.start_enabled,
        debug_overlay=config.debug_overlay,
    )
    state.set_event("control enabled" if state.control_enabled else "control disabled")

    camera_source = parse_camera_source(config.camera)
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        LOGGER.error("Could not open camera source %r", config.camera)
        return 3

    prepare_mediapipe_vision_import()
    mp = importlib.import_module("mediapipe")
    mp_hands = mp.solutions.hands
    face_mesh_module = mp.solutions.face_mesh
    drawing = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles

    window_name = "Tony Stark Hand + Face Control"
    LOGGER.info("hand moves the cursor; face winks click, mouth drags, tilt scrolls")
    LOGGER.info("press E to toggle control mode; press Q or Esc to quit")
    LOGGER.info("action backend: %s", backend.name)

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.6,
        ) as hands, face_mesh_module.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as face_mesh:
            while True:
                ok, frame = cap.read()
                if not ok:
                    LOGGER.error("Camera frame read failed")
                    return 4

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_result = hands.process(rgb)
                face_result = face_mesh.process(rgb)

                hand_point: Optional[Point] = None
                state.hand_present = False
                if hand_result.multi_hand_landmarks:
                    hand_landmarks = hand_result.multi_hand_landmarks[0]
                    drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        styles.get_default_hand_landmarks_style(),
                        styles.get_default_hand_connections_style(),
                    )
                    hand_point = hand_center_point(
                        hand_landmarks.landmark, HAND_CENTER_LANDMARKS
                    )
                    state.hand_present = True

                gesture = FaceGesture("no_face", 0.0)
                if face_result.multi_face_landmarks:
                    face_landmarks = face_result.multi_face_landmarks[0]
                    drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=face_mesh_module.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=styles.get_default_face_mesh_contours_style(),
                    )
                    gesture = classify_face(
                        face_landmarks.landmark,
                        wink_threshold=config.wink_threshold,
                        eye_open_threshold=config.eye_open_threshold,
                        wink_gap=config.wink_gap,
                        mouth_open_threshold=config.mouth_open_threshold,
                        tilt_scroll_threshold=config.tilt_scroll_threshold,
                    )

                try:
                    apply_actions(config, state, backend, hand_point, gesture)
                except Exception as exc:  # pragma: no cover - desktop dependent.
                    try:
                        release_drag_if_needed(state, backend)
                    except Exception:
                        state.drag_active = False
                    state.control_enabled = False
                    state.set_event(f"actions disabled after error: {exc}")

                draw_overlay(cv2, frame, config, state, backend, gesture)
                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    state.set_event("kill switch requested")
                    break
                if key in (ord("e"), ord("E")):
                    state.control_enabled = not state.control_enabled
                    if not state.control_enabled:
                        release_drag_if_needed(state, backend)
                    state.set_event(
                        "control enabled" if state.control_enabled else "control disabled"
                    )
                if key in (ord("d"), ord("D")):
                    state.debug_overlay = not state.debug_overlay
                    state.set_event(
                        "debug overlay enabled"
                        if state.debug_overlay
                        else "debug overlay disabled"
                    )
    finally:
        release_drag_if_needed(state, backend)
        cap.release()
        cv2.destroyAllWindows()

    return 0


def build_backend(force_dry_run: bool) -> ActionBackend:
    if force_dry_run:
        LOGGER.info("dry-run requested; OS input actions are disabled")
        return DryRunBackend()

    try:
        return PyAutoGuiBackend()
    except Exception as exc:  # pragma: no cover - depends on desktop session.
        LOGGER.warning("PyAutoGUI unavailable; falling back to dry-run: %s", exc)
        return DryRunBackend()


def classify_face(
    landmarks: Sequence[object],
    *,
    wink_threshold: float = 0.18,
    eye_open_threshold: float = 0.23,
    wink_gap: float = 0.07,
    mouth_open_threshold: float = 0.32,
    tilt_scroll_threshold: float = 16.0,
) -> FaceGesture:
    metrics = measure_face(landmarks)
    left_closed = metrics.left_ear <= wink_threshold
    right_closed = metrics.right_ear <= wink_threshold
    left_open = metrics.left_ear >= eye_open_threshold
    right_open = metrics.right_ear >= eye_open_threshold
    eye_gap = abs(metrics.left_ear - metrics.right_ear)

    if metrics.mouth_ratio >= mouth_open_threshold:
        confidence = clamp(metrics.mouth_ratio / max(mouth_open_threshold, 0.001), 0.0, 1.0)
        return FaceGesture("mouth_open", confidence, metrics, metrics.nose_point)

    if left_closed and right_closed:
        return FaceGesture("blink", 0.5, metrics, metrics.nose_point)

    if left_closed and right_open and eye_gap >= wink_gap:
        confidence = wink_confidence(metrics.right_ear, metrics.left_ear, wink_gap)
        return FaceGesture("left_wink", confidence, metrics, metrics.nose_point)

    if right_closed and left_open and eye_gap >= wink_gap:
        confidence = wink_confidence(metrics.left_ear, metrics.right_ear, wink_gap)
        return FaceGesture("right_wink", confidence, metrics, metrics.nose_point)

    if metrics.roll_degrees <= -tilt_scroll_threshold:
        confidence = tilt_confidence(metrics.roll_degrees, tilt_scroll_threshold)
        return FaceGesture("tilt_left", confidence, metrics, metrics.nose_point)

    if metrics.roll_degrees >= tilt_scroll_threshold:
        confidence = tilt_confidence(metrics.roll_degrees, tilt_scroll_threshold)
        return FaceGesture("tilt_right", confidence, metrics, metrics.nose_point)

    return FaceGesture("face_neutral", 0.85, metrics, metrics.nose_point)


def measure_face(landmarks: Sequence[object]) -> FaceMetrics:
    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
    mouth_ratio = distance(landmarks[MOUTH_TOP], landmarks[MOUTH_BOTTOM]) / max(
        distance(landmarks[MOUTH_LEFT], landmarks[MOUTH_RIGHT]), 0.001
    )
    left_eye_outer = landmark_point(landmarks[LEFT_EYE[0]])
    right_eye_outer = landmark_point(landmarks[RIGHT_EYE[3]])
    roll_degrees = math.degrees(
        math.atan2(right_eye_outer[1] - left_eye_outer[1], right_eye_outer[0] - left_eye_outer[0])
    )
    return FaceMetrics(
        left_ear=left_ear,
        right_ear=right_ear,
        mouth_ratio=mouth_ratio,
        roll_degrees=roll_degrees,
        nose_point=landmark_point(landmarks[NOSE_TIP]),
    )


def eye_aspect_ratio(landmarks: Sequence[object], indexes: Sequence[int]) -> float:
    p1, p2, p3, p4, p5, p6 = (landmarks[index] for index in indexes)
    vertical = distance(p2, p6) + distance(p3, p5)
    horizontal = 2.0 * max(distance(p1, p4), 0.001)
    return vertical / horizontal


def wink_confidence(open_eye_ear: float, closed_eye_ear: float, wink_gap: float) -> float:
    return clamp((open_eye_ear - closed_eye_ear) / max(wink_gap * 2.0, 0.001), 0.0, 1.0)


def tilt_confidence(roll_degrees: float, threshold: float) -> float:
    return clamp((abs(roll_degrees) - threshold) / max(threshold, 1.0), 0.0, 1.0)


def apply_actions(
    config: RuntimeConfig,
    state: RuntimeState,
    backend: ActionBackend,
    hand_point: Optional[Point],
    gesture: FaceGesture,
) -> None:
    now = time.monotonic()
    if gesture.name != state.active_gesture:
        state.active_gesture = gesture.name
        state.gesture_started_at = now
    stable_for = now - state.gesture_started_at

    if not state.control_enabled:
        release_drag_if_needed(state, backend)
        forget_hand_reference(state)
        return

    # Cursor movement is hand-driven, pose-independent, and relative: the pointer
    # moves by however much the hand moves from the previous frame, starting from
    # wherever the cursor already is. A missing hand drops the reference so the
    # next sighting resumes without a jump.
    if hand_point is not None:
        move_cursor_relative(config, state, backend, hand_point)
    else:
        forget_hand_reference(state)

    if gesture.metrics is None:
        release_drag_if_needed(state, backend)
        state.click_armed = True
        return

    if gesture.name == "mouth_open" and gesture.confidence >= config.min_confidence:
        if stable_for >= config.drag_hold_seconds and not state.drag_active:
            backend.mouse_down()
            state.drag_active = True
            state.set_event("mouth drag started")
        state.click_armed = True
        return

    release_drag_if_needed(state, backend)

    if gesture.name in ("left_wink", "right_wink") and gesture.confidence >= config.min_confidence:
        if (
            stable_for >= config.wink_hold_seconds
            and state.click_armed
            and now - state.last_click_at >= config.click_cooldown
        ):
            backend.click()
            state.last_click_at = now
            state.click_armed = False
            state.set_event(f"{gesture.name} click")
        return

    state.click_armed = True

    if (
        config.enable_tilt_scroll
        and gesture.name in ("tilt_left", "tilt_right")
        and gesture.confidence >= config.min_confidence
    ):
        apply_tilt_scroll(config, state, backend, gesture.metrics.roll_degrees)


def move_cursor_relative(
    config: RuntimeConfig,
    state: RuntimeState,
    backend: ActionBackend,
    hand_point: Point,
) -> None:
    if state.last_hand_point is None:
        # First sighting (or resuming after the hand left): anchor the reference
        # to the hand and the cursor to the live OS pointer so nothing jumps.
        state.last_hand_point = hand_point
        state.cursor_pos = backend.position()
        state.cursor_velocity = (0.0, 0.0)
        return

    width, height = backend.screen_size
    raw_dx = (hand_point[0] - state.last_hand_point[0]) * width * config.cursor_gain
    raw_dy = (hand_point[1] - state.last_hand_point[1]) * height * config.cursor_gain
    state.last_hand_point = hand_point

    alpha = config.smoothing
    vx = state.cursor_velocity[0] * (1.0 - alpha) + raw_dx * alpha
    vy = state.cursor_velocity[1] * (1.0 - alpha) + raw_dy * alpha
    state.cursor_velocity = (vx, vy)

    if math.hypot(vx, vy) < config.cursor_deadzone:
        return

    if state.cursor_pos is None:
        state.cursor_pos = backend.position()
    state.cursor_pos = (
        clamp(state.cursor_pos[0] + vx, 0.0, float(width)),
        clamp(state.cursor_pos[1] + vy, 0.0, float(height)),
    )
    backend.move_to(state.cursor_pos)


def forget_hand_reference(state: RuntimeState) -> None:
    state.last_hand_point = None
    state.cursor_velocity = (0.0, 0.0)


def apply_tilt_scroll(
    config: RuntimeConfig,
    state: RuntimeState,
    backend: ActionBackend,
    roll_degrees: float,
) -> None:
    now = time.monotonic()
    if now - state.last_scroll_at < config.scroll_interval:
        return

    excess = abs(roll_degrees) - config.tilt_scroll_threshold
    amount = int(
        clamp(
            excess * config.scroll_scale,
            1,
            config.scroll_max_step,
        )
    )
    if roll_degrees > 0:
        amount = -amount
    backend.scroll(amount)
    state.last_scroll_at = now
    state.set_event(f"tilt scroll {amount}")


def release_drag_if_needed(state: RuntimeState, backend: ActionBackend) -> None:
    if state.drag_active:
        backend.mouse_up()
        state.drag_active = False
        state.set_event("drag released")


def draw_overlay(
    cv2: object,
    frame: object,
    config: RuntimeConfig,
    state: RuntimeState,
    backend: ActionBackend,
    gesture: FaceGesture,
) -> None:
    enabled = state.control_enabled
    color = (40, 180, 70) if enabled else (40, 40, 220)
    panel_height = 300 if state.debug_overlay else 180
    cv2.rectangle(frame, (12, 12), (650, panel_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (12, 12), (650, panel_height), color, 2)

    lines = [
        f"Control: {'ENABLED' if enabled else 'DISABLED'}",
        f"Gesture: {gesture.name} ({gesture.confidence:.2f})",
        f"Backend: {backend.name}",
        f"Hand cursor: {'tracking' if state.hand_present else 'no hand'}",
        f"Tilt scroll: {'on' if config.enable_tilt_scroll else 'off'}",
        f"Last: {shorten(state.last_event, 58)}",
        "E toggle | D debug | Q/Esc kill switch",
    ]
    if state.debug_overlay:
        lines.extend(debug_overlay_lines(config, gesture))

    for idx, line in enumerate(lines):
        y = 38 + idx * 20
        cv2.putText(
            frame,
            line,
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def debug_overlay_lines(config: RuntimeConfig, gesture: FaceGesture) -> list[str]:
    metrics = gesture.metrics
    if metrics is None:
        return ["Face: not detected"]

    return [
        f"EAR L/R: {metrics.left_ear:.2f} / {metrics.right_ear:.2f}",
        f"Wink closed/open/gap: {config.wink_threshold:.2f} / {config.eye_open_threshold:.2f} / {config.wink_gap:.2f}",
        f"Mouth: {metrics.mouth_ratio:.2f} / {config.mouth_open_threshold:.2f}",
        f"Roll: {metrics.roll_degrees:.1f} deg / {config.tilt_scroll_threshold:.1f}",
        f"Nose: {metrics.nose_point[0]:.2f}, {metrics.nose_point[1]:.2f}",
    ]


def landmark_point(landmark: object) -> Point:
    return landmark.x, landmark.y


def distance(a: object, b: object) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)
