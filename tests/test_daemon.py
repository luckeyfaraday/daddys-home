import unittest

from gesture_control_daemon.daemon import (
    ActionBackend,
    Gesture,
    RuntimeConfig,
    RuntimeState,
    apply_actions,
    apply_scroll,
    classify_gesture,
    move_cursor,
)


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
        self.scrolls = []
        self.clicks = 0

    def move_to(self, point):
        self.moves.append(point)

    def click(self):
        self.clicks += 1

    def mouse_down(self):
        return

    def mouse_up(self):
        return

    def scroll(self, amount):
        self.scrolls.append(amount)

    def hotkey(self, *keys):
        return


def make_config(**overrides):
    values = {
        "camera": "0",
        "start_enabled": False,
        "dry_run": True,
        "enable_swipe_hotkeys": False,
        "smoothing": 1.0,
        "click_cooldown": 0.55,
        "click_hold_seconds": 0.12,
        "drag_hold_seconds": 0.28,
        "min_confidence": 0.65,
        "margin": 0.0,
        "debug_overlay": False,
        "cursor_deadzone": 6.0,
        "cursor_extension_threshold": 0.06,
        "finger_extension_threshold": 0.015,
        "pinch_extension_threshold": 0.025,
        "pinch_threshold": 0.38,
        "scroll_deadzone": 0.022,
        "scroll_scale": 700.0,
        "scroll_max_step": 5,
    }
    values.update(overrides)
    return RuntimeConfig(**values)


def make_landmarks(index=False, middle=False, ring=False, pinky=False, pinch=False):
    landmarks = [Landmark(0.5, 0.8) for _ in range(21)]
    landmarks[0] = Landmark(0.5, 0.9)
    landmarks[5] = Landmark(0.45, 0.65)
    landmarks[9] = Landmark(0.5, 0.65)
    landmarks[13] = Landmark(0.55, 0.65)
    landmarks[17] = Landmark(0.6, 0.65)
    landmarks[4] = Landmark(0.2, 0.6)

    set_finger(landmarks, 8, 6, 0.45, index)
    set_finger(landmarks, 12, 10, 0.5, middle)
    set_finger(landmarks, 16, 14, 0.55, ring)
    set_finger(landmarks, 20, 18, 0.6, pinky)

    if pinch:
        landmarks[4] = Landmark(0.45, 0.45)
        landmarks[8] = Landmark(0.46, 0.45)

    return landmarks


def set_finger(landmarks, tip, pip, x, extended):
    landmarks[pip] = Landmark(x, 0.55)
    landmarks[tip] = Landmark(x, 0.45 if extended else 0.62)


class GestureHeuristicTests(unittest.TestCase):
    def test_classifies_index_only_as_cursor(self):
        gesture = classify_gesture(make_landmarks(index=True))

        self.assertEqual(gesture.name, "cursor")
        self.assertEqual(gesture.finger_state, "I---")

    def test_weak_index_extension_with_folded_fingers_stays_fist(self):
        landmarks = make_landmarks(index=True)
        landmarks[6].y = 0.55
        landmarks[8].y = 0.525

        gesture = classify_gesture(landmarks)

        self.assertEqual(gesture.name, "fist")
        self.assertEqual(gesture.finger_state, "I---")
        self.assertLess(gesture.index_extension, 0.06)

    def test_classifies_index_and_middle_as_scroll(self):
        gesture = classify_gesture(make_landmarks(index=True, middle=True))

        self.assertEqual(gesture.name, "scroll")
        self.assertEqual(gesture.finger_state, "IM--")

    def test_classifies_all_fingers_extended_as_open_palm(self):
        gesture = classify_gesture(
            make_landmarks(index=True, middle=True, ring=True, pinky=True)
        )

        self.assertEqual(gesture.name, "open_palm")
        self.assertEqual(gesture.finger_state, "IMRP")

    def test_pinch_overrides_finger_state(self):
        gesture = classify_gesture(make_landmarks(index=True, pinch=True))

        self.assertEqual(gesture.name, "pinch")
        self.assertLess(gesture.pinch_ratio, 0.38)

    def test_closed_fist_with_close_thumb_and_index_stays_fist(self):
        landmarks = make_landmarks(pinch=True)
        landmarks[6].y = 0.55
        landmarks[8].y = 0.54

        gesture = classify_gesture(landmarks)

        self.assertEqual(gesture.name, "fist")
        self.assertLess(gesture.pinch_ratio, 0.38)
        self.assertLess(gesture.index_extension, 0.025)


class MotionTuningTests(unittest.TestCase):
    def test_pinch_click_does_not_move_previously_aimed_cursor(self):
        config = make_config(click_hold_seconds=0.0)
        state = RuntimeState(
            control_enabled=True,
            active_gesture="pinch",
            smoothed_cursor=(500.0, 500.0),
        )
        backend = RecordingBackend()
        gesture = Gesture("pinch", 0.9, cursor_point=(0.1, 0.1))

        apply_actions(config, state, backend, gesture)

        self.assertEqual(backend.moves, [])
        self.assertEqual(backend.clicks, 1)
        self.assertEqual(state.smoothed_cursor, (500.0, 500.0))

    def test_cursor_deadzone_suppresses_small_moves(self):
        config = make_config(cursor_deadzone=10.0)
        state = RuntimeState(smoothed_cursor=(500.0, 500.0))
        backend = RecordingBackend()

        move_cursor(config, state, backend, (0.503, 0.504))
        self.assertEqual(backend.moves, [])

        move_cursor(config, state, backend, (0.53, 0.54))
        self.assertEqual(len(backend.moves), 1)

    def test_scroll_deadzone_and_max_step(self):
        config = make_config(scroll_deadzone=0.05, scroll_scale=100.0, scroll_max_step=3)
        state = RuntimeState(scroll_anchor_y=0.5)
        backend = RecordingBackend()

        apply_scroll(config, state, backend, (0.5, 0.52))
        self.assertEqual(backend.scrolls, [])

        apply_scroll(config, state, backend, (0.5, 0.6))
        self.assertEqual(backend.scrolls, [-3])


if __name__ == "__main__":
    unittest.main()
