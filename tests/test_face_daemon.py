import unittest

from face_control_daemon.daemon import (
    FaceGesture,
    FaceMetrics,
    RuntimeConfig,
    RuntimeState,
    apply_actions,
    classify_face,
)
from gesture_control_daemon.daemon import ActionBackend


class Landmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class RecordingBackend(ActionBackend):
    name = "recording"

    def __init__(self):
        super().__init__()
        self.screen_size = (1000, 1000)
        self.moves = []
        self.clicks = 0
        self.downs = 0
        self.ups = 0
        self.scrolls = []

    def move_to(self, point):
        self.moves.append(point)

    def click(self):
        self.clicks += 1

    def mouse_down(self):
        self.downs += 1

    def mouse_up(self):
        self.ups += 1

    def scroll(self, amount):
        self.scrolls.append(amount)

    def hotkey(self, *keys):
        return


def make_config(**overrides):
    values = {
        "camera": "0",
        "start_enabled": False,
        "dry_run": True,
        "debug_overlay": False,
        "enable_tilt_scroll": False,
        "smoothing": 1.0,
        "cursor_deadzone": 2.0,
        "cursor_gain": 1.0,
        "click_cooldown": 0.65,
        "wink_hold_seconds": 0.16,
        "drag_hold_seconds": 0.35,
        "min_confidence": 0.65,
        "wink_threshold": 0.18,
        "eye_open_threshold": 0.23,
        "wink_gap": 0.07,
        "mouth_open_threshold": 0.32,
        "tilt_scroll_threshold": 16.0,
        "scroll_interval": 0.12,
        "scroll_scale": 0.35,
        "scroll_max_step": 4,
    }
    values.update(overrides)
    return RuntimeConfig(**values)


def make_face(left_ear=0.30, right_ear=0.30, mouth_ratio=0.10, roll_degrees=0.0):
    landmarks = [Landmark(0.5, 0.5) for _ in range(478)]
    set_eye(landmarks, (33, 160, 158, 133, 153, 144), 0.42, 0.42, left_ear)
    set_eye(landmarks, (362, 385, 387, 263, 373, 380), 0.58, 0.42, right_ear)
    set_mouth(landmarks, mouth_ratio)
    landmarks[1] = Landmark(0.5, 0.5)

    if roll_degrees:
        left = landmarks[33]
        right = landmarks[263]
        width = right.x - left.x
        dy = width * __import__("math").tan(__import__("math").radians(roll_degrees))
        for index in (362, 385, 387, 263, 373, 380):
            landmarks[index].y += dy

    return landmarks


def set_eye(landmarks, indexes, cx, cy, ear, width=0.08):
    p1, p2, p3, p4, p5, p6 = indexes
    half_width = width / 2.0
    half_height = ear * width / 2.0
    landmarks[p1] = Landmark(cx - half_width, cy)
    landmarks[p4] = Landmark(cx + half_width, cy)
    landmarks[p2] = Landmark(cx - width * 0.2, cy - half_height)
    landmarks[p3] = Landmark(cx + width * 0.2, cy - half_height)
    landmarks[p5] = Landmark(cx + width * 0.2, cy + half_height)
    landmarks[p6] = Landmark(cx - width * 0.2, cy + half_height)


def set_mouth(landmarks, ratio, width=0.16):
    cx = 0.5
    cy = 0.67
    half_width = width / 2.0
    half_open = ratio * width / 2.0
    landmarks[78] = Landmark(cx - half_width, cy)
    landmarks[308] = Landmark(cx + half_width, cy)
    landmarks[13] = Landmark(cx, cy - half_open)
    landmarks[14] = Landmark(cx, cy + half_open)


class FaceGestureTests(unittest.TestCase):
    def test_classifies_neutral_face(self):
        gesture = classify_face(make_face())

        self.assertEqual(gesture.name, "face_neutral")
        self.assertGreater(gesture.metrics.left_ear, 0.23)
        self.assertGreater(gesture.metrics.right_ear, 0.23)

    def test_classifies_left_wink(self):
        gesture = classify_face(make_face(left_ear=0.12, right_ear=0.30))

        self.assertEqual(gesture.name, "left_wink")
        self.assertGreaterEqual(gesture.confidence, 0.65)

    def test_both_eyes_closed_is_blink_not_wink(self):
        gesture = classify_face(make_face(left_ear=0.12, right_ear=0.12))

        self.assertEqual(gesture.name, "blink")

    def test_mouth_open_takes_drag_priority(self):
        gesture = classify_face(make_face(left_ear=0.12, right_ear=0.30, mouth_ratio=0.40))

        self.assertEqual(gesture.name, "mouth_open")

    def test_classifies_head_tilt(self):
        gesture = classify_face(make_face(roll_degrees=24.0))

        self.assertEqual(gesture.name, "tilt_right")


class FaceActionTests(unittest.TestCase):
    def test_wink_click_does_not_move_cursor_without_hand(self):
        config = make_config(wink_hold_seconds=0.0)
        state = RuntimeState(
            control_enabled=True,
            active_gesture="left_wink",
            cursor_pos=(500.0, 500.0),
        )
        backend = RecordingBackend()
        metrics = FaceMetrics(0.12, 0.30, 0.10, 0.0, (0.1, 0.1))
        gesture = FaceGesture("left_wink", 0.9, metrics, metrics.nose_point)

        apply_actions(config, state, backend, None, gesture)

        self.assertEqual(backend.moves, [])
        self.assertEqual(backend.clicks, 1)
        self.assertEqual(state.cursor_pos, (500.0, 500.0))

    def test_hand_moves_cursor_relative_to_current_position(self):
        config = make_config()
        state = RuntimeState(control_enabled=True, active_gesture="face_neutral")
        backend = RecordingBackend()  # 1000x1000, starts centered at (500, 500)
        metrics = FaceMetrics(0.30, 0.30, 0.10, 0.0, (0.5, 0.5))
        gesture = FaceGesture("face_neutral", 0.85, metrics, metrics.nose_point)

        # First frame anchors the reference; no movement yet.
        apply_actions(config, state, backend, (0.5, 0.5), gesture)
        self.assertEqual(backend.moves, [])

        # Hand moves +0.1 in x: cursor nudges from 500 by 0.1*1000*gain(1.0)=100.
        apply_actions(config, state, backend, (0.6, 0.5), gesture)
        self.assertEqual(len(backend.moves), 1)
        self.assertAlmostEqual(backend.moves[-1][0], 600.0)
        self.assertAlmostEqual(backend.moves[-1][1], 500.0)

    def test_cursor_does_not_jump_when_hand_reappears(self):
        config = make_config()
        state = RuntimeState(control_enabled=True, active_gesture="face_neutral")
        backend = RecordingBackend()
        metrics = FaceMetrics(0.30, 0.30, 0.10, 0.0, (0.5, 0.5))
        gesture = FaceGesture("face_neutral", 0.85, metrics, metrics.nose_point)

        apply_actions(config, state, backend, (0.5, 0.5), gesture)  # anchor
        apply_actions(config, state, backend, None, gesture)  # hand leaves
        moves_before = len(backend.moves)

        # Hand reappears far away: should re-anchor, not teleport the cursor.
        apply_actions(config, state, backend, (0.9, 0.9), gesture)
        self.assertEqual(len(backend.moves), moves_before)

    def test_mouth_open_starts_drag_and_hand_keeps_moving(self):
        config = make_config(drag_hold_seconds=0.0)
        state = RuntimeState(control_enabled=True, active_gesture="mouth_open")
        backend = RecordingBackend()
        metrics = FaceMetrics(0.30, 0.30, 0.40, 0.0, (0.52, 0.50))
        gesture = FaceGesture("mouth_open", 0.9, metrics, metrics.nose_point)

        apply_actions(config, state, backend, (0.5, 0.5), gesture)  # anchor + drag down
        apply_actions(config, state, backend, (0.6, 0.5), gesture)  # drag while moving

        self.assertEqual(backend.downs, 1)
        self.assertEqual(len(backend.moves), 1)


if __name__ == "__main__":
    unittest.main()
