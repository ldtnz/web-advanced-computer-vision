"use client";

import {
  FilesetResolver,
  HandLandmarker,
} from "@mediapipe/tasks-vision";
import { useEffect, useRef, useState } from "react";

type Point = { x: number; y: number };
type Landmark = { x: number; y: number };
type Poly = [Point, Point, Point, Point];
type FingerMarkers = Point[];
type GestureState = "pinched" | "released" | null;
type HandLandmarkerResult = { landmarks: Landmark[][] };
type Rgb = [number, number, number];
type EffectMode = "ascii" | "blueprint" | "halftone";
type EffectRegion = { mode: EffectMode; poly: Poly };
type SortedHand = { landmarks: Landmark[]; centerX: number };
type HandMergeState = {
  active: boolean;
  lastMatchedAt: number;
  point: Point | null;
};
type HandPointResolver = (id: number) => Point;
type DebugHandInfo = {
  label: string;
  pinchRatio: number;
  openFingers: number;
  okFingers: number;
  center: Point;
};
type DebugMetrics = {
  renderFps: number;
  detectionMs: number;
  hands: number;
  markers: number;
  effectEnabled: boolean;
  effectsHidden: boolean;
  effectModes: string;
  effectRegions: number;
  gesture: GestureState;
  pinchArmed: boolean;
  pinchTaps: number;
  debugVisible: boolean;
  canvas: string;
  video: string;
  detection: string;
  polygonArea: number;
  handInfo: DebugHandInfo[];
};

const polySmoothing = 0.35;
const toggleFeedbackMs = 1500;
const doublePinchWindowMs = 1400;
const detectionWidth = 480;
const pinchThreshold = 0.34;
const releaseThreshold = 0.45;
const threeFingerMergeEnterThreshold = 0.52;
const threeFingerMergeExitThreshold = 0.76;
const threeFingerMergeOpenThreshold = 0.98;
const threeFingerMergeHoldMs = 260;
const asciiChars = " .:-=+*#%@";
const effectModes: EffectMode[] = ["halftone", "ascii", "blueprint"];
const effectFingerBands: Record<EffectMode, [number, number]> = {
  halftone: [4, 8],
  ascii: [8, 12],
  blueprint: [12, 20],
};
const wasmUrl =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm";

function clamp(value: number, low: number, high: number) {
  return Math.max(low, Math.min(high, value));
}

function lerp(start: number, end: number, amount: number) {
  return start + (end - start) * amount;
}

function mixColor(from: Rgb, to: Rgb, amount: number) {
  return `rgb(${Math.round(lerp(from[0], to[0], amount))}, ${Math.round(
    lerp(from[1], to[1], amount),
  )}, ${Math.round(lerp(from[2], to[2], amount))})`;
}

function smoothPoly(previous: Poly | null, current: Poly): Poly {
  if (!previous) {
    return current;
  }

  return current.map((point, index) => ({
    x: previous[index].x * polySmoothing + point.x * (1 - polySmoothing),
    y: previous[index].y * polySmoothing + point.y * (1 - polySmoothing),
  })) as Poly;
}

function smoothEffectRegions(
  previous: EffectRegion[] | null,
  current: EffectRegion[],
): EffectRegion[] {
  if (!previous || previous.length !== current.length) {
    return current;
  }

  return current.map((region, index) => ({
    ...region,
    poly: smoothPoly(previous[index].poly, region.poly),
  }));
}

function smoothPoints(previous: FingerMarkers | null, current: FingerMarkers): FingerMarkers {
  if (!previous || previous.length !== current.length) {
    return current;
  }

  return current.map((point, index) => ({
    x: previous[index].x * polySmoothing + point.x * (1 - polySmoothing),
    y: previous[index].y * polySmoothing + point.y * (1 - polySmoothing),
  }));
}

function getCover(video: HTMLVideoElement, canvas: HTMLCanvasElement) {
  const sourceW = video.videoWidth || 1280;
  const sourceH = video.videoHeight || 720;
  const scale = Math.max(canvas.width / sourceW, canvas.height / sourceH);
  const width = sourceW * scale;
  const height = sourceH * scale;

  return {
    x: (canvas.width - width) / 2,
    y: (canvas.height - height) / 2,
    width,
    height,
  };
}

function landmarkToPoint(
  landmark: { x: number; y: number },
  cover: ReturnType<typeof getCover>,
) {
  return {
    x: cover.x + (1 - landmark.x) * cover.width,
    y: cover.y + landmark.y * cover.height,
  };
}

function getSortedHands(result: HandLandmarkerResult): SortedHand[] {
  return result.landmarks
    .slice(0, 2)
    .map((landmarks: Landmark[]) => {
      const center = landmarks.reduce(
        (acc: Point, point: Landmark) => ({ x: acc.x + point.x, y: acc.y + point.y }),
        { x: 0, y: 0 },
      );

      return {
        landmarks,
        centerX: 1 - center.x / landmarks.length,
      };
    })
    .sort((a: { centerX: number }, b: { centerX: number }) => a.centerX - b.centerX);
}

function getEffectRegions(hands: SortedHand[], getHandPoints: HandPointResolver[]) {
  if (hands.length < 2 || getHandPoints.length < 2) {
    return [];
  }

  return effectModes.map((mode) => {
    const [fromId, toId] = effectFingerBands[mode];

    return {
      mode,
      poly: [
        getHandPoints[0](toId),
        getHandPoints[1](toId),
        getHandPoints[1](fromId),
        getHandPoints[0](fromId),
      ] as Poly,
    };
  });
}

function getFingerMarkers(hands: SortedHand[], getHandPoints: HandPointResolver[]) {
  const tipIds = [4, 8, 12, 16, 20];
  return hands.flatMap((_, handIndex) => tipIds.map((id) => getHandPoints[handIndex](id)));
}

function getThreeFingerMergeRatio(hand: Landmark[]) {
  const palmWidth = getPalmWidth(hand);

  return Math.max(
    distance(hand[4], hand[8]),
    distance(hand[8], hand[12]),
    distance(hand[4], hand[12]),
  ) / palmWidth;
}

function getThreeFingerMergePoint(
  hand: Landmark[],
  cover: ReturnType<typeof getCover>,
) {
  const center = [4, 8, 12].reduce(
    (acc, id) => ({ x: acc.x + hand[id].x, y: acc.y + hand[id].y }),
    { x: 0, y: 0 },
  );

  return landmarkToPoint({ x: center.x / 3, y: center.y / 3 }, cover);
}

function getTrackedHandPointResolver(
  hand: Landmark[],
  cover: ReturnType<typeof getCover>,
  state: HandMergeState,
  now: number,
) {
  const ratio = getThreeFingerMergeRatio(hand);
  const currentMergePoint = getThreeFingerMergePoint(hand, cover);
  const shouldEnter = ratio < threeFingerMergeEnterThreshold;
  const shouldStayMatched = ratio < threeFingerMergeExitThreshold;
  const isClearlyOpen = ratio > threeFingerMergeOpenThreshold;

  if (shouldEnter || (state.active && shouldStayMatched)) {
    state.active = true;
    state.lastMatchedAt = now;
    state.point = currentMergePoint;
  } else if (
    state.active &&
    !isClearlyOpen &&
    now - state.lastMatchedAt <= threeFingerMergeHoldMs
  ) {
    state.point = currentMergePoint;
  } else {
    state.active = false;
    state.point = null;
  }

  return (id: number) => state.point ?? landmarkToPoint(hand[id], cover);
}

function getHandPointResolvers(
  hands: SortedHand[],
  cover: ReturnType<typeof getCover>,
  mergeStates: HandMergeState[],
  now: number,
) {
  return hands.map((hand, index) =>
    getTrackedHandPointResolver(hand.landmarks, cover, mergeStates[index], now),
  );
}

function distance(a: Landmark, b: Landmark) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function getPalmWidth(hand: Landmark[]) {
  return distance(hand[5], hand[17]) || distance(hand[0], hand[9]) || 1;
}

function getPinchRatio(hand: Landmark[]) {
  return distance(hand[4], hand[8]) / getPalmWidth(hand);
}

function countOkFingers(hand: Landmark[]) {
  return [
    hand[12].y < hand[10].y,
    hand[16].y < hand[14].y,
    hand[20].y < hand[18].y,
  ].filter(Boolean).length;
}

function countOpenFingers(hand: Landmark[]) {
  const thumbOpen = distance(hand[4], hand[9]) > distance(hand[3], hand[9]) * 1.2;

  return [
    thumbOpen,
    hand[8].y < hand[6].y,
    hand[12].y < hand[10].y,
    hand[16].y < hand[14].y,
    hand[20].y < hand[18].y,
  ].filter(Boolean).length;
}

function getHandDebugInfo(result: HandLandmarkerResult): DebugHandInfo[] {
  return result.landmarks.slice(0, 2).map((hand, index) => {
    const center = hand.reduce(
      (acc, point) => ({ x: acc.x + point.x, y: acc.y + point.y }),
      { x: 0, y: 0 },
    );

    return {
      label: `hand ${index + 1}`,
      pinchRatio: getPinchRatio(hand),
      openFingers: countOpenFingers(hand),
      okFingers: countOkFingers(hand),
      center: {
        x: center.x / hand.length,
        y: center.y / hand.length,
      },
    };
  });
}

function getGestureState(result: HandLandmarkerResult): GestureState {
  const pinchRatios = result.landmarks.slice(0, 2).map(getPinchRatio);

  if (pinchRatios.length < 2) {
    return "released";
  }

  if (pinchRatios.every((ratio) => ratio < pinchThreshold)) {
    return "pinched";
  }

  if (pinchRatios.every((ratio) => ratio > releaseThreshold)) {
    return "released";
  }

  return null;
}

function polygonArea(poly: Poly) {
  return Math.abs(
    poly.reduce((sum, point, index) => {
      const next = poly[(index + 1) % poly.length];
      return sum + point.x * next.y - next.x * point.y;
    }, 0) / 2,
  );
}

function effectRegionsArea(regions: EffectRegion[] | null) {
  return regions?.reduce((sum, region) => sum + polygonArea(region.poly), 0) ?? 0;
}

function polygonBounds(poly: Poly, canvas: HTMLCanvasElement) {
  const xs = poly.map((point) => point.x);
  const ys = poly.map((point) => point.y);

  return {
    x1: clamp(Math.floor(Math.min(...xs)), 0, canvas.width - 1),
    y1: clamp(Math.floor(Math.min(...ys)), 0, canvas.height - 1),
    x2: clamp(Math.ceil(Math.max(...xs)), 0, canvas.width),
    y2: clamp(Math.ceil(Math.max(...ys)), 0, canvas.height),
  };
}

function tracePoly(ctx: CanvasRenderingContext2D, poly: Poly) {
  ctx.beginPath();
  poly.forEach((point, index) => {
    if (index === 0) {
      ctx.moveTo(point.x, point.y);
    } else {
      ctx.lineTo(point.x, point.y);
    }
  });
  ctx.closePath();
}

function drawAsciiFilter(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  poly: Poly,
  t: number,
) {
  const bounds = polygonBounds(poly, canvas);
  const width = bounds.x2 - bounds.x1;
  const height = bounds.y2 - bounds.y1;
  if (width <= 0 || height <= 0) {
    return;
  }

  const cellW = 16;
  const cellH = 22;
  const cols = Math.max(1, Math.floor(width / cellW));
  const rows = Math.max(1, Math.floor(height / cellH));
  const sample = document.createElement("canvas");
  sample.width = cols;
  sample.height = rows;
  const sampleCtx = sample.getContext("2d", { willReadFrequently: true });
  if (!sampleCtx) {
    return;
  }

  sampleCtx.drawImage(canvas, bounds.x1, bounds.y1, width, height, 0, 0, cols, rows);
  const image = sampleCtx.getImageData(0, 0, cols, rows).data;

  ctx.save();
  tracePoly(ctx, poly);
  ctx.clip();
  ctx.fillStyle = "rgba(8, 8, 8, 0.88)";
  ctx.fillRect(bounds.x1, bounds.y1, width, height);

  ctx.font = `${Math.max(18, Math.floor(cellH * 0.95))}px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace`;
  ctx.textBaseline = "top";

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const index = (row * cols + col) * 4;
      const gray = image[index] * 0.299 + image[index + 1] * 0.587 + image[index + 2] * 0.114;
      const noise = (((col * 13 + row * 29 + Math.floor(t * 16) * 7) % 17) - 8) * 2.2;
      const value = clamp(gray + noise, 0, 255);
      const charIndex = Math.floor((value / 255) * (asciiChars.length - 1));
      const alpha = 0.34 + (value / 255) * 0.58;

      ctx.fillStyle = `rgba(235, 235, 226, ${alpha})`;
      ctx.fillText(
        asciiChars[charIndex],
        bounds.x1 + col * cellW,
        bounds.y1 + row * cellH,
      );
    }
  }

  ctx.restore();

  ctx.strokeStyle = "rgba(245, 245, 238, 0.76)";
  ctx.lineWidth = 2;
  tracePoly(ctx, poly);
  ctx.stroke();
}

function drawHalftoneFilter(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, poly: Poly) {
  const bounds = polygonBounds(poly, canvas);
  const width = bounds.x2 - bounds.x1;
  const height = bounds.y2 - bounds.y1;
  if (width <= 0 || height <= 0) {
    return;
  }

  const cell = 10;
  const cols = Math.max(1, Math.floor(width / cell));
  const rows = Math.max(1, Math.floor(height / cell));
  const sample = document.createElement("canvas");
  sample.width = cols;
  sample.height = rows;
  const sampleCtx = sample.getContext("2d", { willReadFrequently: true });
  if (!sampleCtx) {
    return;
  }

  sampleCtx.drawImage(canvas, bounds.x1, bounds.y1, width, height, 0, 0, cols, rows);
  const image = sampleCtx.getImageData(0, 0, cols, rows).data;

  ctx.save();
  tracePoly(ctx, poly);
  ctx.clip();
  ctx.fillStyle = "rgba(252, 250, 244, 0.98)";
  ctx.fillRect(bounds.x1, bounds.y1, width, height);

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const index = (row * cols + col) * 4;
      const r = image[index];
      const g = image[index + 1];
      const b = image[index + 2];
      const gray = r * 0.299 + g * 0.587 + b * 0.114;
      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      const darkness = 1 - gray / 255;
      const chroma = (max - min) / 255;
      const ink = clamp(darkness * 1.45 + chroma * 0.32 - 0.14, 0, 1);
      const radius = ink * cell * 0.56;

      if (radius < 0.75) {
        continue;
      }

      ctx.fillStyle = `rgba(214, 49, 38, ${0.42 + ink * 0.5})`;
      ctx.beginPath();
      ctx.arc(
        bounds.x1 + col * cell + cell * 0.5,
        bounds.y1 + row * cell + cell * 0.5,
        radius,
        0,
        Math.PI * 2,
      );
      ctx.fill();
    }
  }

  ctx.restore();

  ctx.strokeStyle = "rgba(255, 255, 255, 0.88)";
  ctx.lineWidth = 2;
  tracePoly(ctx, poly);
  ctx.stroke();
}

function drawBlueprintFilter(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  poly: Poly,
  t: number,
) {
  const bounds = polygonBounds(poly, canvas);
  const width = bounds.x2 - bounds.x1;
  const height = bounds.y2 - bounds.y1;
  if (width <= 0 || height <= 0) {
    return;
  }

  const cell = 8;
  const cols = Math.max(1, Math.floor(width / cell));
  const rows = Math.max(1, Math.floor(height / cell));
  const sample = document.createElement("canvas");
  sample.width = cols;
  sample.height = rows;
  const sampleCtx = sample.getContext("2d", { willReadFrequently: true });
  if (!sampleCtx) {
    return;
  }

  sampleCtx.drawImage(canvas, bounds.x1, bounds.y1, width, height, 0, 0, cols, rows);
  const image = sampleCtx.getImageData(0, 0, cols, rows).data;

  ctx.save();
  tracePoly(ctx, poly);
  ctx.clip();
  ctx.fillStyle = "rgba(8, 44, 92, 0.94)";
  ctx.fillRect(bounds.x1, bounds.y1, width, height);

  ctx.lineWidth = 1;
  for (let x = bounds.x1; x <= bounds.x2; x += cell * 4) {
    ctx.strokeStyle = "rgba(162, 226, 255, 0.16)";
    ctx.beginPath();
    ctx.moveTo(x, bounds.y1);
    ctx.lineTo(x, bounds.y2);
    ctx.stroke();
  }
  for (let y = bounds.y1; y <= bounds.y2; y += cell * 4) {
    ctx.strokeStyle = "rgba(162, 226, 255, 0.16)";
    ctx.beginPath();
    ctx.moveTo(bounds.x1, y);
    ctx.lineTo(bounds.x2, y);
    ctx.stroke();
  }

  ctx.lineWidth = 1.35;
  for (let row = 1; row < rows - 1; row += 1) {
    for (let col = 1; col < cols - 1; col += 1) {
      const index = (row * cols + col) * 4;
      const rightIndex = (row * cols + col + 1) * 4;
      const downIndex = ((row + 1) * cols + col) * 4;
      const gray = image[index] * 0.299 + image[index + 1] * 0.587 + image[index + 2] * 0.114;
      const rightGray =
        image[rightIndex] * 0.299 + image[rightIndex + 1] * 0.587 + image[rightIndex + 2] * 0.114;
      const downGray =
        image[downIndex] * 0.299 + image[downIndex + 1] * 0.587 + image[downIndex + 2] * 0.114;
      const edge = clamp((Math.abs(gray - rightGray) + Math.abs(gray - downGray)) / 90, 0, 1);
      const glow = clamp((gray - 92) / 150, 0, 1);
      const alpha = Math.max(edge * 0.82, glow * 0.38);

      if (alpha < 0.13) {
        continue;
      }

      const x = bounds.x1 + col * cell + cell * 0.5;
      const y = bounds.y1 + row * cell + cell * 0.5;
      const angle = edge > glow ? Math.atan2(downGray - gray, rightGray - gray) : -Math.PI / 4;
      const length = cell * (0.7 + Math.max(edge, glow) * 0.9);
      const flicker = 0.86 + (((col * 11 + row * 17 + Math.floor(t * 12)) % 9) / 8) * 0.14;

      ctx.strokeStyle = `rgba(214, 248, 255, ${alpha * flicker})`;
      ctx.beginPath();
      ctx.moveTo(x - Math.cos(angle) * length * 0.5, y - Math.sin(angle) * length * 0.5);
      ctx.lineTo(x + Math.cos(angle) * length * 0.5, y + Math.sin(angle) * length * 0.5);
      ctx.stroke();
    }
  }

  ctx.restore();

  ctx.strokeStyle = "rgba(203, 244, 255, 0.9)";
  ctx.lineWidth = 2;
  tracePoly(ctx, poly);
  ctx.stroke();
}

function drawEffect(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  poly: Poly,
  mode: EffectMode,
  t: number,
) {
  if (mode === "halftone") {
    drawHalftoneFilter(ctx, canvas, poly);
    return;
  }

  if (mode === "blueprint") {
    drawBlueprintFilter(ctx, canvas, poly, t);
    return;
  }

  drawAsciiFilter(ctx, canvas, poly, t);
}

function drawFingerMarkers(
  ctx: CanvasRenderingContext2D,
  markers: FingerMarkers,
  feedback: "on" | "off" | null,
  feedbackAmount: number,
) {
  markers.forEach((point, index) => {
    const radius = 6;
    const fingerIndex = index % 5;
    const showsFeedback = fingerIndex === 0 || fingerIndex === 1;
    const baseColor: Rgb = [205, 205, 198];
    const feedbackColor: Rgb =
      feedback === "on" ? [86, 214, 132] : feedback === "off" ? [218, 82, 82] : baseColor;
    const colorMix = showsFeedback ? feedbackAmount : 0;
    const fillColor =
      colorMix > 0 ? mixColor(baseColor, feedbackColor, colorMix) : "rgb(205, 205, 198)";
    const strokeColor =
      feedback === "on" && colorMix > 0
        ? `rgba(10, 55, 28, ${lerp(0.28, 0.48, colorMix)})`
        : feedback === "off" && colorMix > 0
          ? `rgba(65, 12, 12, ${lerp(0.28, 0.48, colorMix)})`
          : "rgba(20, 20, 20, 0.28)";

    ctx.save();
    ctx.globalAlpha = lerp(0.22, 0.46, colorMix);
    ctx.fillStyle = fillColor;
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  });
}

function drawDebugPanel(ctx: CanvasRenderingContext2D, metrics: DebugMetrics) {
  const handLines = metrics.handInfo.flatMap((hand) => [
    [`${hand.label} fingers`, `${hand.openFingers}/5 open`],
    [`${hand.label} pinch`, hand.pinchRatio.toFixed(2)],
    [`${hand.label} ok`, `${hand.okFingers}/3 fingers`],
    [`${hand.label} center`, `${hand.center.x.toFixed(2)}, ${hand.center.y.toFixed(2)}`],
  ]);
  const lines = [
    ["render fps", metrics.renderFps.toFixed(1)],
    ["detection", `${metrics.detectionMs.toFixed(1)} ms`],
    ["hands", String(metrics.hands)],
    ["markers", String(metrics.markers)],
    ["effect", metrics.effectEnabled ? "on" : "off"],
    ["effects hidden", metrics.effectsHidden ? "yes" : "no"],
    ["effect modes", metrics.effectModes],
    ["effect regions", String(metrics.effectRegions)],
    ["gesture", metrics.gesture ?? "none"],
    ["pinch armed", metrics.pinchArmed ? "yes" : "no"],
    ["pinch taps", `${metrics.pinchTaps}/2`],
    ["debug", metrics.debugVisible ? "on" : "off"],
    ["canvas", metrics.canvas],
    ["video", metrics.video],
    ["detection", metrics.detection],
    ["poly area", Math.round(metrics.polygonArea).toLocaleString()],
    ...handLines,
  ];
  const x = 24;
  const y = 24;
  const lineHeight = 24;
  const width = 460;
  const height = 42 + lines.length * lineHeight;

  ctx.save();
  ctx.fillStyle = "rgba(5, 5, 5, 0.48)";
  ctx.strokeStyle = "rgba(235, 235, 226, 0.28)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.roundRect(x, y, width, height, 10);
  ctx.fill();
  ctx.stroke();

  ctx.font = "15px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
  ctx.textBaseline = "top";
  ctx.fillStyle = "rgba(245, 245, 238, 0.86)";
  ctx.fillText("debug", x + 18, y + 14);

  lines.forEach(([label, value], index) => {
    const top = y + 44 + index * lineHeight;
    ctx.fillStyle = "rgba(198, 198, 190, 0.78)";
    ctx.fillText(label, x + 18, top);
    ctx.fillStyle = "rgba(245, 245, 238, 0.94)";
    ctx.fillText(value, x + 190, top);
  });
  ctx.restore();
}

export function GestureCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const detectionCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const effectRegionsRef = useRef<EffectRegion[] | null>(null);
  const markerRef = useRef<FingerMarkers | null>(null);
  const effectEnabledRef = useRef(false);
  const pinchArmedRef = useRef(true);
  const pinchTapCountRef = useRef(0);
  const lastPinchAtRef = useRef(0);
  const feedbackRef = useRef<"on" | "off" | null>(null);
  const feedbackStartedRef = useRef(0);
  const debugVisibleRef = useRef(false);
  const mergeStatesRef = useRef<HandMergeState[]>([
    { active: false, lastMatchedAt: 0, point: null },
    { active: false, lastMatchedAt: 0, point: null },
  ]);
  const debugMetricsRef = useRef<DebugMetrics>({
    renderFps: 0,
    detectionMs: 0,
    hands: 0,
    markers: 0,
    effectEnabled: false,
    effectsHidden: false,
    effectModes: effectModes.join(", "),
    effectRegions: 0,
    gesture: null,
    pinchArmed: true,
    pinchTaps: 0,
    debugVisible: false,
    canvas: "0x0",
    video: "0x0",
    detection: "0x0",
    polygonArea: 0,
    handInfo: [],
  });
  const lastFrameTimeRef = useRef(0);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");

  useEffect(() => {
    const pressedKeys = new Set<string>();
    let debugComboArmed = true;

    function hasDebugCombo() {
      return pressedKeys.has("Space") && pressedKeys.has("KeyD");
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.code === "Space" || event.code === "KeyD") {
        pressedKeys.add(event.code);
        if (hasDebugCombo()) {
          event.preventDefault();
          if (debugComboArmed) {
            debugVisibleRef.current = !debugVisibleRef.current;
            debugComboArmed = false;
          }
        }
      }
    }

    function handleKeyUp(event: KeyboardEvent) {
      if (event.code === "Space" || event.code === "KeyD") {
        pressedKeys.delete(event.code);
        debugComboArmed = !hasDebugCombo();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, []);

  useEffect(() => {
    let stopped = false;
    const video = videoRef.current;

    async function start() {
      if (!video) {
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 60 },
          },
          audio: false,
        });

        video.srcObject = stream;
        await video.play();

        const vision = await FilesetResolver.forVisionTasks(wasmUrl);
        landmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "/models/hand_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.6,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        if (!stopped) {
          setStatus("ready");
        }
      } catch {
        if (!stopped) {
          setStatus("error");
        }
      }
    }

    start();

    return () => {
      stopped = true;
      if (video?.srcObject instanceof MediaStream) {
        video.srcObject.getTracks().forEach((track) => track.stop());
      }
      landmarkerRef.current?.close();
    };
  }, []);

  useEffect(() => {
    const canvasEl = canvasRef.current;
    const videoEl = videoRef.current;
    if (!canvasEl || !videoEl || status !== "ready") {
      return;
    }

    const context = canvasEl.getContext("2d", { alpha: false });
    if (!context) {
      return;
    }

    const canvas = canvasEl;
    const video = videoEl;
    const ctx = context;

    function resizeCanvas() {
      const dpr = window.devicePixelRatio || 1;
      const width = Math.floor(window.innerWidth * dpr);
      const height = Math.floor(window.innerHeight * dpr);

      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
    }

    function frame(now: number) {
      resizeCanvas();
      if (lastFrameTimeRef.current) {
        const delta = now - lastFrameTimeRef.current;
        const fps = delta > 0 ? 1000 / delta : 0;
        debugMetricsRef.current.renderFps =
          debugMetricsRef.current.renderFps * 0.88 + fps * 0.12;
      }
      lastFrameTimeRef.current = now;

      const cover = getCover(video, canvas);
      ctx.fillStyle = "#050505";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.save();
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, cover.x, cover.y, cover.width, cover.height);
      ctx.restore();

      const landmarker = landmarkerRef.current;
      if (landmarker && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        if (!detectionCanvasRef.current) {
          detectionCanvasRef.current = document.createElement("canvas");
        }
        const detectionCanvas = detectionCanvasRef.current;
        const detectionScale = detectionWidth / (video.videoWidth || detectionWidth);
        detectionCanvas.width = detectionWidth;
        detectionCanvas.height = Math.max(
          1,
          Math.round((video.videoHeight || detectionWidth) * detectionScale),
        );
        const detectionCtx = detectionCanvas.getContext("2d");
        if (!detectionCtx) {
          rafRef.current = window.requestAnimationFrame(frame);
          return;
        }
        detectionCtx.drawImage(video, 0, 0, detectionCanvas.width, detectionCanvas.height);

        const detectionStart = performance.now();
        const result = landmarker.detectForVideo(detectionCanvas, now);
        const detectionMs = performance.now() - detectionStart;
        const sortedHands = getSortedHands(result);
        const getHandPoints = getHandPointResolvers(
          sortedHands,
          cover,
          mergeStatesRef.current,
          now,
        );
        const nextEffectRegions = getEffectRegions(sortedHands, getHandPoints);
        const nextMarkers = getFingerMarkers(sortedHands, getHandPoints);
        const gestureState = getGestureState(result);
        const effectsHiddenByMerge =
          effectEnabledRef.current &&
          sortedHands.length >= 2 &&
          mergeStatesRef.current.slice(0, 2).every((state) => state.active);
        if (!nextEffectRegions.length) {
          effectRegionsRef.current = null;
        } else if (!effectsHiddenByMerge) {
          effectRegionsRef.current = smoothEffectRegions(
            effectRegionsRef.current,
            nextEffectRegions,
          );
        }
        markerRef.current = nextMarkers.length
          ? smoothPoints(markerRef.current, nextMarkers)
          : null;

        if (
          pinchTapCountRef.current > 0 &&
          now - lastPinchAtRef.current > doublePinchWindowMs
        ) {
          pinchTapCountRef.current = 0;
          lastPinchAtRef.current = 0;
        }

        if (gestureState === "released") {
          pinchArmedRef.current = true;
        } else if (gestureState === "pinched" && pinchArmedRef.current) {
          if (
            pinchTapCountRef.current === 1 &&
            now - lastPinchAtRef.current <= doublePinchWindowMs
          ) {
            effectEnabledRef.current = !effectEnabledRef.current;
            feedbackRef.current = effectEnabledRef.current ? "on" : "off";
            feedbackStartedRef.current = now;
            pinchTapCountRef.current = 0;
            lastPinchAtRef.current = 0;
          } else {
            pinchTapCountRef.current = 1;
            lastPinchAtRef.current = now;
          }
          pinchArmedRef.current = false;
        }

        debugMetricsRef.current = {
          ...debugMetricsRef.current,
          detectionMs,
          hands: result.landmarks.length,
          markers: nextMarkers.length,
          effectEnabled: effectEnabledRef.current,
          effectsHidden: effectsHiddenByMerge,
          effectModes: effectModes.join(", "),
          effectRegions: nextEffectRegions.length,
          gesture: gestureState,
          pinchArmed: pinchArmedRef.current,
          pinchTaps: pinchTapCountRef.current,
          debugVisible: debugVisibleRef.current,
          canvas: `${canvas.width}x${canvas.height}`,
          video: `${video.videoWidth || 0}x${video.videoHeight || 0}`,
          detection: `${detectionCanvas.width}x${detectionCanvas.height}`,
          polygonArea: effectRegionsArea(effectRegionsRef.current),
          handInfo: getHandDebugInfo(result),
        };
      }

      if (
        effectRegionsRef.current &&
        effectEnabledRef.current &&
        !debugMetricsRef.current.effectsHidden
      ) {
        effectRegionsRef.current.forEach((region) => {
          drawEffect(ctx, canvas, region.poly, region.mode, now / 1000);
        });
      }
      if (markerRef.current) {
        let feedbackAmount = 0;
        if (feedbackRef.current) {
          const progress = clamp((now - feedbackStartedRef.current) / toggleFeedbackMs, 0, 1);
          feedbackAmount = Math.sin(progress * Math.PI);
          if (progress >= 1) {
            feedbackRef.current = null;
            feedbackAmount = 0;
          }
        }
        drawFingerMarkers(ctx, markerRef.current, feedbackRef.current, feedbackAmount);
      }
      if (debugVisibleRef.current) {
        drawDebugPanel(ctx, debugMetricsRef.current);
      }

      rafRef.current = window.requestAnimationFrame(frame);
    }

    rafRef.current = window.requestAnimationFrame(frame);

    return () => {
      if (rafRef.current !== null) {
        window.cancelAnimationFrame(rafRef.current);
      }
    };
  }, [status]);

  return (
    <main className="relative h-dvh w-dvw overflow-hidden bg-black">
      <video ref={videoRef} className="hidden" playsInline muted />
      <canvas ref={canvasRef} className="block h-full w-full" />
      {status !== "ready" ? (
        <div className="absolute inset-0 grid place-items-center bg-black text-sm uppercase tracking-[0.35em] text-white/70">
          {status === "loading" ? "Loading" : "Camera unavailable"}
        </div>
      ) : null}
    </main>
  );
}
