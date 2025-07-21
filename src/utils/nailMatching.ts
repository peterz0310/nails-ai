/**
 * Nail-Fingertip Matching and Orientation Analysis (REVISED)
 *
 * This module handles:
 * 1. Matching detected nails to specific fingertips
 * 2. Calculating a full 3D orientation basis (X, Y, Z axes) for each nail
 * 3. Calculating nail dimensions and 2D angle for drawing
 * 4. Preparing robust data for the 3D model overlay
 */

import { YoloDetection } from "./yolo";
import { HandDetection } from "./mediapipe";

// FIXED: The interface now includes a full 3D orientation basis.
// This is the key to solving rotation and tilting issues.
export interface NailFingerMatch {
  nailDetection: YoloDetection;
  fingertipIndex: number; // MediaPipe finger index (4, 8, 12, 16, 20)
  fingertipPosition: [number, number];
  nailCentroid: [number, number];
  nailWidth: number; // Length along finger direction (longitudinal)
  nailHeight: number; // Width perpendicular to finger direction (transverse)
  nailAngle: number; // 2D rotation in radians for canvas drawing and smoothing
  matchConfidence: number;
  handIndex: number;
  handedness: "Left" | "Right";
  // NEW: Full 3D orientation basis for robust 3D rendering
  orientation: {
    xAxis: [number, number, number]; // Transverse vector (nail's "right")
    yAxis: [number, number, number]; // Normal vector (out from nail surface, nail's "up")
    zAxis: [number, number, number]; // Longitudinal vector (along finger, nail's "forward")
  };
}

// MediaPipe finger landmark indices
const FINGER_TIPS = [4, 8, 12, 16, 20]; // Thumb, Index, Middle, Ring, Pinky
const FINGER_PIPS = [3, 6, 10, 14, 18]; // Proximal Interphalangeal joints for more stable direction
const FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"];

/**
 * Calculate the centroid of a nail detection
 */
function calculateNailCentroid(detection: YoloDetection): [number, number] {
  if (detection.maskPolygon && detection.maskPolygon.length > 0) {
    const sumX = detection.maskPolygon.reduce(
      (sum, point) => sum + point[0],
      0
    );
    const sumY = detection.maskPolygon.reduce(
      (sum, point) => sum + point[1],
      0
    );
    return [
      sumX / detection.maskPolygon.length,
      sumY / detection.maskPolygon.length,
    ];
  } else {
    const [x, y, width, height] = detection.bbox;
    return [x + width / 2, y + height / 2];
  }
}

/**
 * Calculate nail dimensions and 2D orientation based on a 2D projection of the finger direction.
 */
function calculateNailDimensions(
  detection: YoloDetection,
  fingerDirection3D: [number, number, number]
): { width: number; height: number; angle: number } {
  // Project the 3D finger direction onto the 2D screen for dimension calculation
  const fingerDirection2D: [number, number] = [
    fingerDirection3D[0],
    fingerDirection3D[1],
  ];
  const angle = Math.atan2(fingerDirection2D[1], fingerDirection2D[0]);

  // The perpendicular direction in 2D
  const perpDirection2D: [number, number] = [
    -fingerDirection2D[1],
    fingerDirection2D[0],
  ];

  if (detection.maskPolygon && detection.maskPolygon.length > 4) {
    const polygon = detection.maskPolygon;
    const centroidX =
      polygon.reduce((sum, p) => sum + p[0], 0) / polygon.length;
    const centroidY =
      polygon.reduce((sum, p) => sum + p[1], 0) / polygon.length;

    let minLong = Infinity,
      maxLong = -Infinity;
    let minTrans = Infinity,
      maxTrans = -Infinity;

    polygon.forEach((point) => {
      const dx = point[0] - centroidX;
      const dy = point[1] - centroidY;

      const longitudinal =
        dx * fingerDirection2D[0] + dy * fingerDirection2D[1];
      minLong = Math.min(minLong, longitudinal);
      maxLong = Math.max(maxLong, longitudinal);

      const transverse = dx * perpDirection2D[0] + dy * perpDirection2D[1];
      minTrans = Math.min(minTrans, transverse);
      maxTrans = Math.max(maxTrans, transverse);
    });

    return {
      width: maxLong - minLong, // Length along the finger
      height: maxTrans - minTrans, // Width across the finger
      angle,
    };
  } else {
    // Fallback to bounding box dimensions
    const [, , bboxWidth, bboxHeight] = detection.bbox;
    return {
      width: Math.max(bboxWidth, bboxHeight),
      height: Math.min(bboxWidth, bboxHeight),
      angle,
    };
  }
}

function distanceBetweenPoints(
  p1: [number, number],
  p2: [number, number]
): number {
  const dx = p1[0] - p2[0];
  const dy = p1[1] - p2[1];
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * FIXED: Calculate a stable 3D finger direction vector.
 * Removed distorting clamps and simplified the logic for a more reliable vector.
 */
function calculateFingerDirection(
  hand: HandDetection,
  fingerIndex: number
): [number, number, number] {
  const tipIndex = FINGER_TIPS[fingerIndex];
  const pipIndex = FINGER_PIPS[fingerIndex];

  if (tipIndex >= hand.landmarks.length || pipIndex >= hand.landmarks.length) {
    return [0, 1, 0]; // Default upward
  }

  const tip = hand.landmarks[tipIndex];
  const pip = hand.landmarks[pipIndex];

  // Vector from the PIP joint to the tip provides a stable finger direction
  let dx = tip.x - pip.x;
  let dy = tip.y - pip.y;
  let dz = tip.z - pip.z;

  // Exaggerate depth slightly for a better 3D effect. This is a common heuristic.
  // The value is reduced from 1.5 to 1.2 for more subtlety.
  dz *= 1.2;

  const length = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (length === 0) return [0, 1, 0]; // Avoid division by zero

  // Return the normalized 3D direction vector
  return [dx / length, dy / length, dz / length];
}

/**
 * FIXED: This is the core logic change.
 * It now computes a full 3D orientation basis (X, Y, Z axes) for each nail,
 * providing all the information needed for correct 3D rotation.
 */
export function matchNailsToFingertips(
  nailDetections: YoloDetection[],
  handDetections: HandDetection[],
  frameWidth: number,
  frameHeight: number
): NailFingerMatch[] {
  if (nailDetections.length === 0 || handDetections.length === 0) {
    return [];
  }

  const matches: NailFingerMatch[] = [];
  const maxDistance = Math.min(frameWidth, frameHeight) * 0.15; // 15% of min dimension

  // Vector math helpers
  const cross = (a: number[], b: number[]): number[] => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const normalize = (v: number[]): number[] => {
    const len = Math.sqrt(v.reduce((sum, val) => sum + val * val, 0));
    return len > 0
      ? (v.map((c) => c / len) as [number, number, number])
      : [0, 0, 0];
  };
  const dot = (a: number[], b: number[]): number =>
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

  handDetections.forEach((hand, handIndex) => {
    FINGER_TIPS.forEach((tipIndex, fingerIndex) => {
      if (tipIndex >= hand.landmarks.length) return;

      const fingertip = hand.landmarks[tipIndex];
      const fingertipPos: [number, number] = [
        fingertip.x * frameWidth,
        fingertip.y * frameHeight,
      ];

      let bestMatch: { detection: YoloDetection; confidence: number } | null =
        null;

      nailDetections.forEach((detection) => {
        const nailCentroid = calculateNailCentroid(detection);
        const dist = distanceBetweenPoints(fingertipPos, nailCentroid);

        if (dist < maxDistance) {
          const distanceScore = Math.max(0, 1 - dist / maxDistance);
          const detectionScore = detection.score;
          const confidence = distanceScore * 0.6 + detectionScore * 0.4;

          if (!bestMatch || confidence > bestMatch.confidence) {
            bestMatch = { detection, confidence };
          }
        }
      });

      if (bestMatch && bestMatch.confidence > 0.4) {
        // --- START OF NEW ORIENTATION LOGIC ---
        const zAxis = calculateFingerDirection(hand, fingerIndex) as [
          number,
          number,
          number
        ];

        // Define world up vector. MediaPipe's Y-axis points down.
        const worldUp: [number, number, number] = [0, -1, 0];

        // Calculate the nail's "right" vector (X-axis)
        let xAxis = normalize(cross(worldUp, zAxis));

        // If finger points straight up/down, the cross product is zero. Provide a fallback.
        if (dot(xAxis, xAxis) < 0.1) {
          const handDirection =
            hand.handedness === "Right" ? [1, 0, 0] : [-1, 0, 0];
          xAxis = normalize(cross(zAxis, handDirection));
        }

        // Calculate the nail's "up" vector (Y-axis), pointing out of the nail surface.
        // The order zAxis, xAxis ensures a right-handed coordinate system.
        const yAxis = normalize(cross(zAxis, xAxis));
        // --- END OF NEW ORIENTATION LOGIC ---

        const nailCentroid = calculateNailCentroid(bestMatch.detection);
        const nailDimensions = calculateNailDimensions(
          bestMatch.detection,
          zAxis
        );

        const match: NailFingerMatch = {
          nailDetection: bestMatch.detection,
          fingertipIndex: tipIndex,
          fingertipPosition: fingertipPos,
          nailCentroid,
          nailWidth: nailDimensions.width,
          nailHeight: nailDimensions.height,
          nailAngle: nailDimensions.angle,
          matchConfidence: bestMatch.confidence,
          handIndex: handIndex,
          handedness: hand.handedness as "Left" | "Right",
          orientation: { xAxis, yAxis, zAxis }, // Store the full basis
        };
        matches.push(match);
      }
    });
  });

  // Deduplication: Ensure each nail and finger is used only once per frame
  const uniqueMatches: NailFingerMatch[] = [];
  const usedNails = new Set<YoloDetection>();
  const usedFingers = new Set<string>();

  matches
    .sort((a, b) => b.matchConfidence - a.matchConfidence)
    .forEach((match) => {
      const fingerKey = `${match.handIndex}-${match.fingertipIndex}`;
      if (!usedNails.has(match.nailDetection) && !usedFingers.has(fingerKey)) {
        uniqueMatches.push(match);
        usedNails.add(match.nailDetection);
        usedFingers.add(fingerKey);
      }
    });

  if (Math.random() < 0.05) {
    // Reduce logging
    console.log(
      `Matching found ${uniqueMatches.length} unique matches from ${matches.length} candidates.`
    );
  }

  return uniqueMatches;
}

/**
 * Draw nail-finger matches with improved visualization
 */
export function drawNailFingerMatches(
  ctx: CanvasRenderingContext2D,
  matches: NailFingerMatch[],
  scaleX: number,
  scaleY: number
): void {
  matches.forEach((match) => {
    const scaledCentroid: [number, number] = [
      match.nailCentroid[0] * scaleX,
      match.nailCentroid[1] * scaleY,
    ];
    const scaledFingertip: [number, number] = [
      match.fingertipPosition[0] * scaleX,
      match.fingertipPosition[1] * scaleY,
    ];

    // Draw connection line
    ctx.strokeStyle = "#00ff88";
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 4]);
    ctx.beginPath();
    ctx.moveTo(scaledCentroid[0], scaledCentroid[1]);
    ctx.lineTo(scaledFingertip[0], scaledFingertip[1]);
    ctx.stroke();

    // Reset line dash for other drawings
    ctx.setLineDash([]);

    // Draw nail centroid
    ctx.fillStyle = "#00ff88";
    ctx.beginPath();
    ctx.arc(scaledCentroid[0], scaledCentroid[1], 4, 0, 2 * Math.PI);
    ctx.fill();

    // Draw finger name and confidence
    const fingerName =
      FINGER_NAMES[FINGER_TIPS.indexOf(match.fingertipIndex)] || "Unknown";
    const label = `${match.handedness} ${fingerName} (${(
      match.matchConfidence * 100
    ).toFixed(0)}%)`;
    ctx.font = "bold 12px Arial";
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillRect(
      scaledFingertip[0] + 10,
      scaledFingertip[1] - 20,
      ctx.measureText(label).width + 8,
      18
    );
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, scaledFingertip[0] + 14, scaledFingertip[1] - 7);
  });
}
