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
import * as THREE from "three"; // Using THREE's Vector3 for convenience

// The interface remains the same, but the 'orientation' data will be more stable.
export interface NailFingerMatch {
  nailDetection: YoloDetection;
  fingertipIndex: number;
  fingertipPosition: [number, number];
  nailCentroid: [number, number];
  nailWidth: number;
  nailHeight: number;
  nailAngle: number;
  matchConfidence: number;
  handIndex: number;
  handedness: "Left" | "Right";
  orientation: {
    xAxis: [number, number, number];
    yAxis: [number, number, number];
    zAxis: [number, number, number];
  };
}

// MediaPipe finger landmark indices, defined for clarity
const FINGER_LANDMARKS = {
  THUMB: { TIP: 4, DIP: 3, PIP: 2 },
  INDEX: { TIP: 8, DIP: 7, PIP: 6 },
  MIDDLE: { TIP: 12, DIP: 11, PIP: 10 },
  RING: { TIP: 16, DIP: 15, PIP: 14 },
  PINKY: { TIP: 20, DIP: 19, PIP: 18 },
};

const FINGER_TIPS = [4, 8, 12, 16, 20];
const FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"];

function getFingerLandmarkIndices(tipIndex: number) {
  switch (tipIndex) {
    case 4:
      return FINGER_LANDMARKS.THUMB;
    case 8:
      return FINGER_LANDMARKS.INDEX;
    case 12:
      return FINGER_LANDMARKS.MIDDLE;
    case 16:
      return FINGER_LANDMARKS.RING;
    case 20:
      return FINGER_LANDMARKS.PINKY;
    default:
      return null;
  }
}

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
 * FIXED: This is the new core logic for calculating a stable 3D orientation basis.
 * It uses the hand's own geometry to avoid issues with world axes.
 */
function calculateOrientationBasis(
  hand: HandDetection,
  tipIndex: number
): NailFingerMatch["orientation"] | null {
  const lm = hand.landmarks;
  const indices = getFingerLandmarkIndices(tipIndex);
  if (!indices) return null;

  const { TIP, DIP, PIP } = indices;
  if ([TIP, DIP, PIP].some((i) => i >= lm.length)) return null;

  // Create vectors from the landmarks
  const p_tip = new THREE.Vector3(lm[TIP].x, lm[TIP].y, lm[TIP].z);
  const p_dip = new THREE.Vector3(lm[DIP].x, lm[DIP].y, lm[DIP].z);
  const p_pip = new THREE.Vector3(lm[PIP].x, lm[PIP].y, lm[PIP].z);

  // Define two vectors on the plane of the back of the finger
  const v1 = p_dip.clone().sub(p_pip); // Vector from PIP to DIP
  const v2 = p_tip.clone().sub(p_dip); // Vector from DIP to TIP

  // The cross product gives us the normal vector to the finger's surface (Y-axis)
  // This is the key to fixing the "upside down" problem.
  let yAxis = new THREE.Vector3().crossVectors(v1, v2).normalize();

  // The direction of the cross product depends on the "winding" of the points.
  // For MediaPipe's landmark ordering, a Left hand will have an inverted normal.
  // We correct this using the provided handedness info.
  if (hand.handedness === "Left") {
    yAxis.negate();
  }

  // The direction of the finger (Z-axis) can be taken from PIP to TIP
  let zAxis = p_tip.clone().sub(p_pip).normalize();

  // We create the third axis (X-axis) using another cross product to ensure orthogonality.
  let xAxis = new THREE.Vector3().crossVectors(yAxis, zAxis).normalize();

  // Re-calculate the Z-axis to make the basis perfectly orthogonal (Gram-Schmidt process)
  zAxis = new THREE.Vector3().crossVectors(xAxis, yAxis).normalize();

  return {
    xAxis: xAxis.toArray() as [number, number, number],
    yAxis: yAxis.toArray() as [number, number, number],
    zAxis: zAxis.toArray() as [number, number, number],
  };
}

function calculateNailDimensions(
  detection: YoloDetection,
  zAxis: [number, number, number]
): { width: number; height: number; angle: number } {
  const fingerDirection2D = new THREE.Vector2(zAxis[0], zAxis[1]).normalize();
  const angle = Math.atan2(fingerDirection2D.y, fingerDirection2D.x);
  const perpDirection2D = new THREE.Vector2(
    -fingerDirection2D.y,
    fingerDirection2D.x
  );

  if (detection.maskPolygon && detection.maskPolygon.length > 4) {
    const polygon = detection.maskPolygon;
    const centroid = calculateNailCentroid(detection);
    const centroidV2 = new THREE.Vector2(centroid[0], centroid[1]);

    let minLong = Infinity,
      maxLong = -Infinity;
    let minTrans = Infinity,
      maxTrans = -Infinity;

    polygon.forEach((p) => {
      const pointV2 = new THREE.Vector2(p[0], p[1]);
      const d = pointV2.sub(centroidV2);

      minLong = Math.min(minLong, d.dot(fingerDirection2D));
      maxLong = Math.max(maxLong, d.dot(fingerDirection2D));
      minTrans = Math.min(minTrans, d.dot(perpDirection2D));
      maxTrans = Math.max(maxTrans, d.dot(perpDirection2D));
    });

    return {
      height: maxLong - minLong, // Length along the finger
      width: maxTrans - minTrans, // Width across the finger
      angle,
    };
  } else {
    const [, , bboxWidth, bboxHeight] = detection.bbox;
    return {
      height: Math.max(bboxWidth, bboxHeight),
      width: Math.min(bboxWidth, bboxHeight),
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
  const maxDistance = Math.min(frameWidth, frameHeight) * 0.15;

  handDetections.forEach((hand, handIndex) => {
    FINGER_TIPS.forEach((tipIndex) => {
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
        // Use the new, robust orientation calculation
        const orientation = calculateOrientationBasis(hand, tipIndex);
        if (!orientation) return;

        const nailCentroid = calculateNailCentroid(bestMatch.detection);
        const nailDimensions = calculateNailDimensions(
          bestMatch.detection,
          orientation.zAxis
        );

        matches.push({
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
          orientation: orientation,
        });
      }
    });
  });

  // Deduplication to ensure each nail and finger is used only once
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

  return uniqueMatches;
}

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

    ctx.strokeStyle = "#00ff88";
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 4]);
    ctx.beginPath();
    ctx.moveTo(scaledCentroid[0], scaledCentroid[1]);
    ctx.lineTo(scaledFingertip[0], scaledFingertip[1]);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = "#00ff88";
    ctx.beginPath();
    ctx.arc(scaledCentroid[0], scaledCentroid[1], 4, 0, 2 * Math.PI);
    ctx.fill();

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
