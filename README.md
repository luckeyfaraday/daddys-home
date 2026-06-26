# Tony Stark Gesture Control Prototype

This workspace starts with a minimal, safety-first gesture control daemon prototype.
It uses a webcam plus MediaPipe hand landmarks, then maps deliberate hand gestures to
mouse and keyboard actions through PyAutoGUI when available.

## Safety Defaults

- Control mode starts disabled.
- A visible OpenCV overlay shows status, gesture, backend, and last action.
- Press `E` in the overlay window to enable or disable control mode.
- Press `D` in the overlay window to show or hide gesture diagnostics.
- Press `Q` or `Esc` in the overlay window to quit immediately.
- PyAutoGUI fail-safe remains enabled, so moving the pointer to the top-left corner
  should raise a fail-safe exception and stop actions.
- Swipe hotkeys are disabled unless `--enable-swipe-hotkeys` is passed.
- If PyAutoGUI is unavailable, the daemon falls back to dry-run action logging.

## Setup

No dependencies are installed automatically. Install them in your chosen environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Linux, PyAutoGUI may also need desktop automation support from the host OS. If it
cannot initialize, the daemon will keep vision running in dry-run mode.

## Run

```bash
python3 -m gesture_control_daemon
```

Useful options:

```bash
python3 -m gesture_control_daemon --dry-run
python3 -m gesture_control_daemon --camera 1
python3 -m gesture_control_daemon --debug-overlay
python3 -m gesture_control_daemon --start-enabled
python3 -m gesture_control_daemon --enable-swipe-hotkeys
```

Useful tuning options:

```bash
python3 -m gesture_control_daemon --cursor-deadzone 10
python3 -m gesture_control_daemon --cursor-extension-threshold 0.08
python3 -m gesture_control_daemon --pinch-threshold 0.34
python3 -m gesture_control_daemon --pinch-extension-threshold 0.04
python3 -m gesture_control_daemon --finger-extension-threshold 0.02
python3 -m gesture_control_daemon --scroll-deadzone 0.03 --scroll-scale 550
```

Hand + face daemon (hand moves the cursor, face triggers actions):

```bash
python3 -m face_control_daemon --dry-run --debug-overlay
python3 -m face_control_daemon --camera 1
python3 -m face_control_daemon --enable-tilt-scroll
```

Hand + face tuning options:

```bash
python3 -m face_control_daemon --cursor-gain 2.0 --cursor-deadzone 3
python3 -m face_control_daemon --wink-threshold 0.16 --eye-open-threshold 0.24
python3 -m face_control_daemon --mouth-open-threshold 0.38
```

## MVP Gestures

Hand daemon:

- Open palm: neutral, no control action.
- Index finger: move cursor while control mode is enabled.
- Pinch: single click with debounce at the last aimed cursor position.
- Fist hold: hold and drag after a short dwell.
- Two fingers vertical: scroll.
- Fast index swipe left/right: switch browser/application tab only when
  `--enable-swipe-hotkeys` is set.

Hand + face daemon:

- Hand in view: move cursor relative to its current position, like a trackpad.
  The pointer moves by however much your hand moves, independent of finger pose;
  hold your hand still and the cursor rests. Take your hand out of frame and
  bring it back to re-center without the cursor jumping.
- One-eye wink: click with debounce at the current cursor position.
- Both eyes closed: blink only, no click.
- Mouth open hold: press and hold the mouse button after a short dwell, so you
  can drag by moving your hand while your mouth stays open.
- Head tilt left/right: scroll only when `--enable-tilt-scroll` is passed.

## Refinement Workflow

Start in dry-run mode:

```bash
python3 -m gesture_control_daemon --dry-run --debug-overlay
python3 -m face_control_daemon --dry-run --debug-overlay
```

Watch the overlay for `Gesture`, `fingers`, and `Pinch ratio`. Tune thresholds one
at a time. If normal cursor movement jitters, raise `--cursor-deadzone`. If scroll
fires too easily, raise `--scroll-deadzone` or lower `--scroll-scale`. If pinch is
too sensitive, lower `--pinch-threshold`.

If a fist is mistaken for cursor, raise `--cursor-extension-threshold`. This
requires a more deliberate index-finger extension before cursor movement starts.
If a fist is mistaken for pinch, raise `--pinch-extension-threshold` or lower
`--pinch-threshold`. This keeps closed-hand thumb/index contact from firing a
click.

The intended click flow is: point with the index finger to aim, then pinch without
moving the cursor. Pinch freezes cursor updates so the click does not get pulled
toward the closing index finger.

For the hand + face daemon, the cursor moves relative to your hand like a
trackpad, so no neutral calibration is needed. Raise `--cursor-gain` to cover
the screen with less hand travel, lower it for finer control, and raise
`--cursor-deadzone` if the pointer jitters while your hand is still. If wink
clicks fire during ordinary blinks, lower `--wink-threshold`, raise
`--eye-open-threshold`, or raise `--wink-gap`.

This is a local prototype, not an OS shell replacement. Gesture thresholds are simple
heuristics and should be tuned with real camera feedback before relying on them.
