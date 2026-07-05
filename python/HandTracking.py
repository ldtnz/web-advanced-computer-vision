import os
import time

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "hand_landmarker.task",
)


class HandDetector:
    def __init__(self, maxHands=2, detectionCon=0.5, trackCon=0.5):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=maxHands,
            min_hand_detection_confidence=detectionCon,
            min_hand_presence_confidence=trackCon,
            min_tracking_confidence=trackCon,
        )
        self.hands = vision.HandLandmarker.create_from_options(options)
        self.results = None
        self.lastTimestampMs = 0

    def findHands(self, img):
        if img is None:
            return

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        timestampMs = int(time.monotonic() * 1000)
        if timestampMs <= self.lastTimestampMs:
            timestampMs = self.lastTimestampMs + 1

        self.lastTimestampMs = timestampMs
        self.results = self.hands.detect_for_video(mpImage, timestampMs)

    def getAllHands(self, img):
        hands = []
        if img is None or not self.results or not self.results.hand_landmarks:
            return hands

        h, w, _ = img.shape
        for handNum, handLandmarks in enumerate(self.results.hand_landmarks):
            lmList = []
            xList = []
            yList = []

            for id, lm in enumerate(handLandmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                lmList.append([id, cx, cy])

            bbox = [min(xList), min(yList), max(xList), max(yList)]
            handType = None
            if len(self.results.handedness) > handNum and self.results.handedness[handNum]:
                handType = self.results.handedness[handNum][0].category_name

            hands.append({
                "lmList": lmList,
                "bbox": bbox,
                "type": handType,
                "center": ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
            })

        return hands
