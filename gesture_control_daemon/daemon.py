from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Sequence, Tuple


Point = Tuple[float, float]


LOGGER = logging.getLogger("gesture_control_daemon")
DEFAULT_MPLCONFIGDIR = "/tmp/tony-stark-matplotlib"


@dataclass(frozen=True)
class Gesture:
    name: str
    confidence: float
    cursor_point: Optional[Point] = None
    scroll_point: Optional[Point] = None
    drag_point: Optional[Point] = None
    center_point: Optional[Point] = None
    finger_state: str = "----"
    pinch_ratio: Optional[float] = None
    index_extension: Optional[float] = None


@dataclass
class RuntimeConfig:
    camera: str
    start_enabled: bool
    dry_run: bool
    enable_swipe_hotkeys: bool
    smoothing: float
    click_cooldown: float
    click_hold_seconds: float
    drag_hold_seconds: float
    min_confidence: float
    margin: float
    debug_overlay: bool
    cursor_deadzone: float
    cursor_extension_threshold: float
    finger_extension_threshold: float
    pinch_extension_threshold: float
    pinch_threshold: float
    scroll_deadzone: float
    scroll_scale: float
    scroll_max_step: int


@dataclass
class RuntimeState:
    control_enabled: bool = False
    active_gesture: str = "none"
    gesture_started_at: float = field(default_factory=time.monotonic)
    drag_active: bool = False
    pinch_armed: bool = True
    last_click_at: float = 0.0
    last_swipe_at: float = 0.0
    scroll_anchor_y: Optional[float] = None
    smoothed_cursor: Optional[Point] = None
    pinch_anchor_cursor: Optional[Point] = None
    debug_overlay: bool = False
    swipe_history: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=16)
    )
    last_event: str = "starting"
    last_event_at: float = field(default_factory=time.monotonic)

    def set_event(self, event: str) -> None:
        self.last_event = event
        self.last_event_at = time.monotonic()
        LOGGER.info(event)


class ActionBackend:
    name = "base"

    def __init__(self) -> None:
        self.screen_size = (1920, 1080)

    def position(self) -> Point:
        """Current pointer position. Defaults to screen center if unknown."""
        return (self.screen_size[0] / 2.0, self.screen_size[1] / 2.0)

    def move_to(self, point: Point) -> None:
        raise NotImplementedError

    def click(self) -> None:
        raise NotImplementedError

    def mouse_down(self) -> None:
        raise NotImplementedError

    def mouse_up(self) -> None:
        raise NotImplementedError

    def scroll(self, amount: int) -> None:
        raise NotImplementedError

    def hotkey(self, *keys: str) -> None:
        raise NotImplementedError


class DryRunBackend(ActionBackend):
    name = "dry-run"

    def __init__(self) -> None:
        super().__init__()
        self._pos: Point = (self.screen_size[0] / 2.0, self.screen_size[1] / 2.0)

    def position(self) -> Point:
        return self._pos

    def move_to(self, point: Point) -> None:
        self._pos = point

    def click(self) -> None:
        LOGGER.info("dry-run click")

    def mouse_down(self) -> None:
        LOGGER.info("dry-run mouse down")

    def mouse_up(self) -> None:
        LOGGER.info("dry-run mouse up")

    def scroll(self, amount: int) -> None:
        LOGGER.info("dry-run scroll %s", amount)

    def hotkey(self, *keys: str) -> None:
        LOGGER.info("dry-run hotkey %s", "+".join(keys))


class PyAutoGuiBackend(ActionBackend):
    name = "pyautogui"

    def __init__(self) -> None:
        super().__init__()
        self._pyautogui = importlib.import_module("pyautogui")
        self._pyautogui.FAILSAFE = True
        self._pyautogui.PAUSE = 0
        self.screen_size = tuple(self._pyautogui.size())

    def position(self) -> Point:
        return tuple(self._pyautogui.position())

    def move_to(self, point: Point) -> None:
        self._pyautogui.moveTo(point[0], point[1], duration=0)

    def click(self) -> None:
        self._pyautogui.click()

    def mouse_down(self) -> None:
        self._pyautogui.mouseDown()

    def mouse_up(self) -> None:
        previous_failsafe = self._pyautogui.FAILSAFE
        try:
            self._pyautogui.FAILSAFE = False
            self._pyautogui.mouseUp()
        finally:
            self._pyautogui.FAILSAFE = previous_failsafe

    def scroll(self, amount: int) -> None:
        self._pyautogui.scroll(amount)

    def hotkey(self, *keys: str) -> None:
        self._pyautogui.hotkey(*keys)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safety-first webcam hand gesture control daemon."
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
        "--enable-swipe-hotkeys",
        action="store_true",
        help="Allow fast index-finger swipes to send Ctrl+Tab/Ctrl+Shift+Tab.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.35,
        help="Cursor smoothing alpha from 0.05 to 1.0. Default: 0.35.",
    )
    parser.add_argument(
        "--click-cooldown",
        type=float,
        default=0.55,
        help="Minimum seconds between pinch clicks. Default: 0.55.",
    )
    parser.add_argument(
        "--click-hold-seconds",
        type=float,
        default=0.12,
        help="Pinch dwell before click. Default: 0.12.",
    )
    parser.add_argument(
        "--drag-hold-seconds",
        type=float,
        default=0.28,
        help="Fist dwell before mouse drag starts. Default: 0.28.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.65,
        help="Minimum gesture confidence for actions. Default: 0.65.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.08,
        help="Ignored camera edge margin for cursor mapping. Default: 0.08.",
    )
    parser.add_argument(
        "--debug-overlay",
        action="store_true",
        help="Start with expanded gesture diagnostics visible. Can also toggle with D.",
    )
    parser.add_argument(
        "--cursor-deadzone",
        type=float,
        default=6.0,
        help="Ignore cursor movements below this many screen pixels. Default: 6.",
    )
    parser.add_argument(
        "--cursor-extension-threshold",
        type=float,
        default=0.06,
        help="Normalized index-finger extension required for cursor mode. Default: 0.06.",
    )
    parser.add_argument(
        "--finger-extension-threshold",
        type=float,
        default=0.015,
        help="Normalized y-distance used to decide if a finger is extended. Default: 0.015.",
    )
    parser.add_argument(
        "--pinch-threshold",
        type=float,
        default=0.38,
        help="Thumb/index distance ratio below which pinch is detected. Default: 0.38.",
    )
    parser.add_argument(
        "--pinch-extension-threshold",
        type=float,
        default=0.025,
        help="Minimum index extension required before thumb/index contact counts as pinch. Default: 0.025.",
    )
    parser.add_argument(
        "--scroll-deadzone",
        type=float,
        default=0.022,
        help="Ignore two-finger vertical motion below this normalized distance. Default: 0.022.",
    )
    parser.add_argument(
        "--scroll-scale",
        type=float,
        default=700.0,
        help="Scroll sensitivity multiplier. Default: 700.",
    )
    parser.add_argument(
        "--scroll-max-step",
        type=int,
        default=5,
        help="Maximum scroll units emitted per frame. Default: 5.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = RuntimeConfig(
        camera=args.camera,
        start_enabled=args.start_enabled,
        dry_run=args.dry_run,
        enable_swipe_hotkeys=args.enable_swipe_hotkeys,
        smoothing=clamp(args.smoothing, 0.05, 1.0),
        click_cooldown=max(0.0, args.click_cooldown),
        click_hold_seconds=max(0.0, args.click_hold_seconds),
        drag_hold_seconds=max(0.0, args.drag_hold_seconds),
        min_confidence=clamp(args.min_confidence, 0.0, 1.0),
        margin=clamp(args.margin, 0.0, 0.35),
        debug_overlay=args.debug_overlay,
        cursor_deadzone=max(0.0, args.cursor_deadzone),
        cursor_extension_threshold=max(0.0, args.cursor_extension_threshold),
        finger_extension_threshold=max(0.0, args.finger_extension_threshold),
        pinch_extension_threshold=max(0.0, args.pinch_extension_threshold),
        pinch_threshold=clamp(args.pinch_threshold, 0.05, 1.0),
        scroll_deadzone=max(0.0, args.scroll_deadzone),
        scroll_scale=max(1.0, args.scroll_scale),
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
    drawing = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles

    window_name = "Tony Stark Gesture Control"
    LOGGER.info("press E to toggle control mode; press Q or Esc to quit")
    LOGGER.info("action backend: %s", backend.name)

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.6,
        ) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    LOGGER.error("Camera frame read failed")
                    return 4

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                gesture = Gesture("no_hand", 0.0)
                if result.multi_hand_landmarks:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        styles.get_default_hand_landmarks_style(),
                        styles.get_default_hand_connections_style(),
                    )
                    gesture = classify_gesture(
                        hand_landmarks.landmark,
                        cursor_extension_threshold=config.cursor_extension_threshold,
                        finger_extension_threshold=config.finger_extension_threshold,
                        pinch_extension_threshold=config.pinch_extension_threshold,
                        pinch_threshold=config.pinch_threshold,
                    )

                try:
                    apply_actions(config, state, backend, gesture)
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


def missing_required_modules(module_names: Sequence[str]) -> list[str]:
    missing: list[str] = []
    for module_name in module_names:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return missing


def build_backend(force_dry_run: bool) -> ActionBackend:
    if force_dry_run:
        LOGGER.info("dry-run requested; OS input actions are disabled")
        return DryRunBackend()

    try:
        return PyAutoGuiBackend()
    except Exception as exc:  # pragma: no cover - depends on desktop session.
        LOGGER.warning("PyAutoGUI unavailable; falling back to dry-run: %s", exc)
        return DryRunBackend()


def parse_camera_source(camera: str) -> object:
    return int(camera) if camera.isdigit() else camera


def prepare_mediapipe_vision_import() -> None:
    os.environ.setdefault("MPLCONFIGDIR", DEFAULT_MPLCONFIGDIR)
    # MediaPipe imports its audio task package at module import time. On some
    # Linux desktops, importing sounddevice can block while probing PortAudio.
    # This daemon only uses vision, so make audio capture look unavailable.
    if os.environ.get("GESTURE_DAEMON_ALLOW_SOUNDDEVICE") != "1":
        sys.modules.setdefault("sounddevice", None)


def classify_gesture(
    landmarks: Sequence[object],
    *,
    cursor_extension_threshold: float = 0.06,
    finger_extension_threshold: float = 0.015,
    pinch_extension_threshold: float = 0.025,
    pinch_threshold: float = 0.38,
) -> Gesture:
    index_extension = finger_extension_amount(landmarks, tip=8, pip=6)
    index = finger_is_extended(
        landmarks, tip=8, pip=6, threshold=finger_extension_threshold
    )
    middle = finger_is_extended(
        landmarks, tip=12, pip=10, threshold=finger_extension_threshold
    )
    ring = finger_is_extended(
        landmarks, tip=16, pip=14, threshold=finger_extension_threshold
    )
    pinky = finger_is_extended(
        landmarks, tip=20, pip=18, threshold=finger_extension_threshold
    )
    finger_state = format_finger_state(index, middle, ring, pinky)
    center = average_points(landmarks, (0, 5, 9, 13, 17))
    index_is_deliberate_cursor = index_extension >= cursor_extension_threshold

    pinch_ratio = distance(landmarks[4], landmarks[8]) / max(
        distance(landmarks[0], landmarks[5]), 0.001
    )
    if pinch_ratio < pinch_threshold and index_extension >= pinch_extension_threshold:
        confidence = clamp((pinch_threshold + 0.10 - pinch_ratio) / 0.28, 0.0, 1.0)
        return Gesture(
            "pinch",
            confidence,
            cursor_point=midpoint(landmarks[4], landmarks[8]),
            center_point=center,
            finger_state=finger_state,
            pinch_ratio=pinch_ratio,
            index_extension=index_extension,
        )

    if index and middle and ring and pinky:
        return Gesture(
            "open_palm",
            0.85,
            center_point=center,
            finger_state=finger_state,
            pinch_ratio=pinch_ratio,
            index_extension=index_extension,
        )

    if index and not middle and not ring and not pinky and index_is_deliberate_cursor:
        return Gesture(
            "cursor",
            0.85,
            cursor_point=landmark_point(landmarks[8]),
            center_point=center,
            finger_state=finger_state,
            pinch_ratio=pinch_ratio,
            index_extension=index_extension,
        )

    if index and not middle and not ring and not pinky:
        return Gesture(
            "fist",
            0.65,
            drag_point=center,
            center_point=center,
            finger_state=finger_state,
            pinch_ratio=pinch_ratio,
            index_extension=index_extension,
        )

    if index and middle and not ring and not pinky:
        return Gesture(
            "scroll",
            0.8,
            scroll_point=average_points(landmarks, (8, 12)),
            center_point=center,
            finger_state=finger_state,
            pinch_ratio=pinch_ratio,
            index_extension=index_extension,
        )

    if not index and not middle and not ring and not pinky:
        return Gesture(
            "fist",
            0.8,
            drag_point=center,
            center_point=center,
            finger_state=finger_state,
            pinch_ratio=pinch_ratio,
            index_extension=index_extension,
        )

    return Gesture(
        "unknown",
        0.4,
        center_point=center,
        finger_state=finger_state,
        pinch_ratio=pinch_ratio,
        index_extension=index_extension,
    )


def finger_is_extended(
    landmarks: Sequence[object], *, tip: int, pip: int, threshold: float = 0.015
) -> bool:
    return finger_extension_amount(landmarks, tip=tip, pip=pip) > threshold


def finger_extension_amount(landmarks: Sequence[object], *, tip: int, pip: int) -> float:
    return landmarks[pip].y - landmarks[tip].y


def format_finger_state(index: bool, middle: bool, ring: bool, pinky: bool) -> str:
    return "".join(
        (
            "I" if index else "-",
            "M" if middle else "-",
            "R" if ring else "-",
            "P" if pinky else "-",
        )
    )


def apply_actions(
    config: RuntimeConfig,
    state: RuntimeState,
    backend: ActionBackend,
    gesture: Gesture,
) -> None:
    now = time.monotonic()
    if gesture.name != state.active_gesture:
        state.active_gesture = gesture.name
        state.gesture_started_at = now
        if gesture.name != "scroll":
            state.scroll_anchor_y = None
        if gesture.name == "pinch":
            state.pinch_anchor_cursor = state.smoothed_cursor
        else:
            state.pinch_anchor_cursor = None
    stable_for = now - state.gesture_started_at

    if not state.control_enabled:
        release_drag_if_needed(state, backend)
        return

    if gesture.confidence < config.min_confidence:
        release_drag_if_needed(state, backend)
        state.pinch_armed = True
        return

    if gesture.name == "cursor" and gesture.cursor_point:
        move_cursor(config, state, backend, gesture.cursor_point)
        maybe_apply_swipe(config, state, backend, gesture.cursor_point)
        release_drag_if_needed(state, backend)
        state.pinch_armed = True
        return

    if gesture.name == "pinch" and gesture.cursor_point:
        release_drag_if_needed(state, backend)
        if state.pinch_anchor_cursor is None and state.smoothed_cursor is None:
            move_cursor(config, state, backend, gesture.cursor_point)
            state.pinch_anchor_cursor = state.smoothed_cursor
        if (
            stable_for >= config.click_hold_seconds
            and state.pinch_armed
            and now - state.last_click_at >= config.click_cooldown
        ):
            backend.click()
            state.last_click_at = now
            state.pinch_armed = False
            state.set_event("pinch click")
        return

    state.pinch_armed = True

    if gesture.name == "fist" and gesture.drag_point:
        move_cursor(config, state, backend, gesture.drag_point)
        if stable_for >= config.drag_hold_seconds and not state.drag_active:
            backend.mouse_down()
            state.drag_active = True
            state.set_event("drag started")
        return

    if gesture.name == "scroll" and gesture.scroll_point:
        release_drag_if_needed(state, backend)
        apply_scroll(config, state, backend, gesture.scroll_point)
        return

    release_drag_if_needed(state, backend)


def move_cursor(
    config: RuntimeConfig,
    state: RuntimeState,
    backend: ActionBackend,
    camera_point: Point,
) -> None:
    target = map_camera_to_screen(camera_point, backend.screen_size, config.margin)
    previous = state.smoothed_cursor
    if previous is None:
        state.smoothed_cursor = target
    else:
        alpha = config.smoothing
        state.smoothed_cursor = (
            previous[0] * (1.0 - alpha) + target[0] * alpha,
            previous[1] * (1.0 - alpha) + target[1] * alpha,
        )
        if point_distance(previous, state.smoothed_cursor) < config.cursor_deadzone:
            return
    backend.move_to(state.smoothed_cursor)


def apply_scroll(
    config: RuntimeConfig, state: RuntimeState, backend: ActionBackend, point: Point
) -> None:
    if state.scroll_anchor_y is None:
        state.scroll_anchor_y = point[1]
        return

    dy = point[1] - state.scroll_anchor_y
    if abs(dy) < config.scroll_deadzone:
        return

    amount = int(
        clamp(
            -dy * config.scroll_scale,
            -config.scroll_max_step,
            config.scroll_max_step,
        )
    )
    if amount:
        backend.scroll(amount)
        state.scroll_anchor_y = point[1]
        state.set_event(f"scroll {amount}")


def maybe_apply_swipe(
    config: RuntimeConfig,
    state: RuntimeState,
    backend: ActionBackend,
    point: Point,
) -> None:
    if not config.enable_swipe_hotkeys:
        return

    now = time.monotonic()
    state.swipe_history.append((now, point[0]))
    while state.swipe_history and now - state.swipe_history[0][0] > 0.45:
        state.swipe_history.popleft()

    if len(state.swipe_history) < 4 or now - state.last_swipe_at < 0.9:
        return

    oldest_time, oldest_x = state.swipe_history[0]
    dx = point[0] - oldest_x
    if now - oldest_time > 0.45 or abs(dx) < 0.33:
        return

    if dx > 0:
        backend.hotkey("ctrl", "tab")
        state.set_event("swipe right: ctrl+tab")
    else:
        backend.hotkey("ctrl", "shift", "tab")
        state.set_event("swipe left: ctrl+shift+tab")
    state.last_swipe_at = now
    state.swipe_history.clear()


def release_drag_if_needed(state: RuntimeState, backend: ActionBackend) -> None:
    if state.drag_active:
        backend.mouse_up()
        state.drag_active = False
        state.set_event("drag released")


def map_camera_to_screen(
    point: Point, screen_size: Tuple[int, int], margin: float
) -> Point:
    x, y = point
    width, height = screen_size
    usable = max(1.0 - margin * 2.0, 0.01)
    mapped_x = clamp((x - margin) / usable, 0.0, 1.0) * width
    mapped_y = clamp((y - margin) / usable, 0.0, 1.0) * height
    return mapped_x, mapped_y


def draw_overlay(
    cv2: object,
    frame: object,
    config: RuntimeConfig,
    state: RuntimeState,
    backend: ActionBackend,
    gesture: Gesture,
) -> None:
    enabled = state.control_enabled
    color = (40, 180, 70) if enabled else (40, 40, 220)
    panel_height = 260 if state.debug_overlay else 180
    cv2.rectangle(frame, (12, 12), (600, panel_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (12, 12), (600, panel_height), color, 2)

    lines = [
        f"Control: {'ENABLED' if enabled else 'DISABLED'}",
        f"Gesture: {gesture.name} ({gesture.confidence:.2f})",
        f"Backend: {backend.name}",
        f"Swipe hotkeys: {'on' if config.enable_swipe_hotkeys else 'off'}",
        f"Last: {shorten(state.last_event, 56)}",
        "E toggle | D debug | Q/Esc kill switch",
    ]
    if state.debug_overlay:
        lines.extend(debug_overlay_lines(config, state, gesture))
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


def debug_overlay_lines(
    config: RuntimeConfig, state: RuntimeState, gesture: Gesture
) -> list[str]:
    stable_for = time.monotonic() - state.gesture_started_at
    lines = [
        f"Stable: {stable_for:.2f}s | fingers: {gesture.finger_state}",
        f"Pinch ratio: {format_optional_float(gesture.pinch_ratio)} / {config.pinch_threshold:.2f}",
        f"Index ext: {format_optional_float(gesture.index_extension)} / {config.cursor_extension_threshold:.2f}",
        f"Pinch ext min: {config.pinch_extension_threshold:.3f}",
        f"Cursor dz: {config.cursor_deadzone:.1f}px | scroll dz: {config.scroll_deadzone:.3f}",
        f"Pinch cursor: {'locked' if state.pinch_anchor_cursor else 'free'}",
    ]
    if gesture.cursor_point:
        lines.append(format_point("Cursor point", gesture.cursor_point))
    elif gesture.scroll_point:
        lines.append(format_point("Scroll point", gesture.scroll_point))
    elif gesture.center_point:
        lines.append(format_point("Hand center", gesture.center_point))
    return lines


def average_points(landmarks: Sequence[object], indexes: Sequence[int]) -> Point:
    x = sum(landmarks[index].x for index in indexes) / len(indexes)
    y = sum(landmarks[index].y for index in indexes) / len(indexes)
    return x, y


def landmark_point(landmark: object) -> Point:
    return landmark.x, landmark.y


def midpoint(a: object, b: object) -> Point:
    return (a.x + b.x) / 2.0, (a.y + b.y) / 2.0


def distance(a: object, b: object) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def point_distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def format_optional_float(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def format_point(label: str, point: Point) -> str:
    return f"{label}: {point[0]:.2f}, {point[1]:.2f}"


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def shorten(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    return value[: max_length - 3] + "..."
