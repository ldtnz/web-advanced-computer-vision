# WACV

WACV is a browser sketch for hand-tracked video effects.

It tracks both hands with MediaPipe, connects matching fingers, and uses the spaces between them as live masks for visual filters. The current web version runs directly in Next.js; the older OpenCV prototype is still in `python/` for reference.

## Run

```bash
npm install
npm run dev
```

Open `http://localhost:3000` and allow camera access.

## Controls

- Show two hands to frame the filter regions.
- Pinch thumb and index finger on both hands twice to toggle the effects.
- Press `Space + D` to show or hide the debug overlay.

The active effects are split across finger bands: halftone between thumb and index, ASCII between index and middle finger, and blueprint between middle and pinky. Thumb and index markers flash green when effects turn on and red when they turn off.

## Check

```bash
npm run typecheck
npm run build
```

There is also a shortcut:

```bash
npm run check
```

## Python Prototype

The Python version uses the same hand-landmark model with OpenCV rendering.

```bash
cd python
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python gestureEffects.py
```

Press `e` to switch effects and `q` or `Esc` to quit. On macOS, allow camera access for the app you use to launch Python, such as Terminal, iTerm, or Visual Studio Code.

## License

MIT. See `LICENSE`.
