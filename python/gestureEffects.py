import math
import threading
import time

import cv2
import numpy as np

import HandTracking as ht


displayW, displayH = 960, 720
detectW, detectH = 480, 360
targetFps = 60
fpsSmoothing = 0.9
polySmoothing = 0.35
effectMode = 0
effectNames = ["particles", "ripple", "scanner"]


def clamp(value, low, high):
    return max(low, min(high, value))


def hand_space_polygon(leftHand, rightHand):
    leftIndex = leftHand["lmList"][8][1:]
    leftThumb = leftHand["lmList"][4][1:]
    rightIndex = rightHand["lmList"][8][1:]
    rightThumb = rightHand["lmList"][4][1:]

    return np.array(
        [leftIndex, rightIndex, rightThumb, leftThumb],
        dtype=np.int32,
    )


def scale_polygon(poly, scaleX, scaleY):
    scaled = poly.astype(np.float32)
    scaled[:, 0] *= scaleX
    scaled[:, 1] *= scaleY
    return scaled


def smooth_polygon(previous, current):
    if previous is None:
        return current.astype(np.float32)

    return previous * polySmoothing + current.astype(np.float32) * (1 - polySmoothing)


def polygon_bounds(poly, width, height):
    x, y, w, h = cv2.boundingRect(poly)
    x1 = clamp(x, 0, width - 1)
    y1 = clamp(y, 0, height - 1)
    x2 = clamp(x + w, 0, width)
    y2 = clamp(y + h, 0, height)
    return x1, y1, x2, y2


def make_mask(shape, poly):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly, 255)
    return mask


def draw_particles(canvas, mask, bounds, t, intensity):
    x1, y1, x2, y2 = bounds
    if x2 <= x1 or y2 <= y1:
        return

    count = int(35 + intensity * 80)
    spanX = max(1, x2 - x1)
    spanY = max(1, y2 - y1)

    for i in range(count):
        phase = t * (0.8 + (i % 7) * 0.08) + i * 12.989
        px = x1 + int((math.sin(phase) * 0.5 + 0.5) * spanX)
        py = y1 + int((math.cos(phase * 1.37) * 0.5 + 0.5) * spanY)
        px = clamp(px, x1, x2 - 1)
        py = clamp(py, y1, y2 - 1)
        if mask[py, px] == 0:
            continue

        radius = 2 + int(4 * (math.sin(phase * 1.9) * 0.5 + 0.5))
        color = (
            80 + int(120 * math.sin(phase * 0.7) ** 2),
            180 + int(60 * math.sin(phase) ** 2),
            255,
        )
        cv2.circle(canvas, (px, py), radius, color, -1, cv2.LINE_AA)


def draw_ripple(canvas, mask, bounds, t, intensity):
    x1, y1, x2, y2 = bounds
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    maxRadius = int(math.hypot(x2 - x1, y2 - y1))
    step = max(22, int(65 - intensity * 25))

    for radius in range(step, maxRadius + step, step):
        wave = int((radius + t * 120) % maxRadius)
        color = (
            255,
            120 + int(80 * math.sin(t + radius) ** 2),
            60 + int(120 * intensity),
        )
        ring = np.zeros(mask.shape, dtype=np.uint8)
        cv2.circle(ring, (cx, cy), wave, 255, 2, cv2.LINE_AA)
        ring = cv2.bitwise_and(ring, mask)
        canvas[ring > 0] = color


def draw_scanner(canvas, mask, bounds, t, intensity):
    x1, y1, x2, y2 = bounds
    spanY = max(1, y2 - y1)
    scanY = y1 + int(((math.sin(t * 2.2) + 1) / 2) * spanY)
    lineMask = np.zeros(mask.shape, dtype=np.uint8)

    for offset in range(-45, 46, 9):
        y = scanY + offset
        if y1 <= y < y2:
            alpha = 1 - abs(offset) / 55
            cv2.line(lineMask, (x1, y), (x2, y), int(255 * alpha), 2)

    lineMask = cv2.bitwise_and(lineMask, mask)
    canvas[lineMask > 0] = (
        60,
        255,
        160 + int(80 * intensity),
    )


def apply_effect(img, poly, mode, t):
    h, w, _ = img.shape
    bounds = polygon_bounds(poly, w, h)
    x1, y1, x2, y2 = bounds
    if x2 <= x1 or y2 <= y1:
        return

    localPoly = poly.copy()
    localPoly[:, 0] -= x1
    localPoly[:, 1] -= y1

    roi = img[y1:y2, x1:x2]
    mask = make_mask(roi.shape, localPoly)
    localBounds = (0, 0, x2 - x1, y2 - y1)
    area = cv2.contourArea(poly)
    intensity = clamp(area / (w * h * 0.22), 0.15, 1.0)

    overlay = np.zeros_like(roi)
    if mode == 0:
        draw_particles(overlay, mask, localBounds, t, intensity)
    elif mode == 1:
        draw_ripple(overlay, mask, localBounds, t, intensity)
    else:
        draw_scanner(overlay, mask, localBounds, t, intensity)

    combined = cv2.addWeighted(roi, 1.0, overlay, 0.95, 0)
    roi[:] = np.where(mask[..., None] > 0, combined, roi)

    cv2.polylines(img, [poly], True, (255, 255, 255), 3, cv2.LINE_AA)
    for point in poly:
        cv2.circle(img, tuple(point), 9, (0, 255, 255), -1, cv2.LINE_AA)


def capture_frames(shared, frameLock, stopEvent):
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    cap.set(3, displayW)
    cap.set(4, displayH)
    cap.set(cv2.CAP_PROP_FPS, targetFps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while not stopEvent.is_set():
        success, frame = cap.read()
        if not success:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        with frameLock:
            shared["frame"] = frame
            shared["frame_id"] += 1

    cap.release()


def detect_hands(shared, frameLock, polyLock, stopEvent):
    detector = ht.HandDetector(maxHands=2, detectionCon=0.6, trackCon=0.5)
    lastFrameId = -1
    smoothedPoly = None

    while not stopEvent.is_set():
        with frameLock:
            frame = shared["frame"]
            frameId = shared["frame_id"]
            if frame is None or frameId == lastFrameId:
                frame = None
            else:
                frame = frame.copy()

        if frame is None:
            time.sleep(0.001)
            continue

        detectImg = cv2.resize(frame, (detectW, detectH), interpolation=cv2.INTER_AREA)
        detector.findHands(detectImg)
        hands = detector.getAllHands(detectImg)

        if len(hands) >= 2:
            hands = sorted(hands[:2], key=lambda hand: hand["center"][0])
            rawPoly = scale_polygon(
                hand_space_polygon(hands[0], hands[1]),
                frame.shape[1] / detectW,
                frame.shape[0] / detectH,
            )
            smoothedPoly = smooth_polygon(smoothedPoly, rawPoly)
            poly = smoothedPoly.copy()
        else:
            smoothedPoly = None
            poly = None

        with polyLock:
            shared["poly"] = poly

        lastFrameId = frameId


def main():
    global effectMode

    shared = {
        "frame": None,
        "frame_id": 0,
        "poly": None,
    }
    frameLock = threading.Lock()
    polyLock = threading.Lock()
    stopEvent = threading.Event()

    captureThread = threading.Thread(
        target=capture_frames,
        args=(shared, frameLock, stopEvent),
        daemon=True,
    )
    detectionThread = threading.Thread(
        target=detect_hands,
        args=(shared, frameLock, polyLock, stopEvent),
        daemon=True,
    )
    captureThread.start()
    detectionThread.start()

    smoothedFps = targetFps
    targetFrameTime = 1 / targetFps

    try:
        while True:
            loopStart = time.monotonic()
            with frameLock:
                sourceFrame = shared["frame"]
                img = sourceFrame.copy() if sourceFrame is not None else None

            if img is None:
                img = np.zeros((displayH, displayW, 3), dtype=np.uint8)
            else:
                with polyLock:
                    poly = shared["poly"].copy() if shared["poly"] is not None else None

                if poly is not None:
                    displayPoly = np.rint(poly).astype(np.int32)
                    apply_effect(img, displayPoly, effectMode, time.time())

            cv2.imshow("WACV", img)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("e"):
                effectMode = (effectMode + 1) % len(effectNames)

            elapsed = time.monotonic() - loopStart
            if elapsed < targetFrameTime:
                time.sleep(targetFrameTime - elapsed)
                elapsed = time.monotonic() - loopStart

            currentFps = 1 / elapsed if elapsed else targetFps
            smoothedFps = smoothedFps * fpsSmoothing + currentFps * (1 - fpsSmoothing)
    finally:
        stopEvent.set()
        captureThread.join(timeout=1)
        detectionThread.join(timeout=1)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
