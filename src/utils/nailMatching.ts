/**
 * Nail-Fingertip Matching and Orientation Analysis (REVISED)
 *
 * This module handles:
 * 1. Matching detected nails to specific fingertips using distance and confidence.
 * 2. Calculating a full 3D orientation basis (X, Y, Z axes) for each nail.
 * 3. Calculating nail dimensions (width, length) and 2D angle for drawing.
 * 4. Preparing robust data for the 3D model overlay.
 */

import { YoloDetection } from "./yolo";
import { HandDetection } from "./mediapipe";
import * as THREE from "three"; // Using THREE's Vector3 for convenience

export interface NailFingerMatch {
  nailDetection: YoloDetection;
  fingertipIndex: number;
  fingertipPosition: [number, number];
  nailCentroid: [number, number];
  nailWidth: number; // Across the finger
  nailHeight: number; // Along the finger
  nailAngle: number;
  matchConfidence: number;
  handIndex: number;
  handedness: "Left" | "Right";
  orientation: {
    // A right-handed coordinate system for the nail
    xAxis: [number, number, number]; // Points across the nail width
    yAxis: [number, number, number]; // Points out from the nail surface (normal)
    zAxis: [number, number, number]; // Points along the finger length
  };
}

// MediaPipe finger landmark indices, defined for clarity
const FINGER_LANDMARKS = {
  THUMB: { TIP: 4, DIP: 3, PIP: 2, MCP: 1 },
  INDEX: { TIP: 8, DIP: 7, PIP: 6, MCP: 5 },
  MIDDLE: { TIP: 12, DIP: 11, PIP: 10, MCP: 9 },
  RING: { TIP: 16, DIP: 15, PIP: 14, MCP: 13 },
  PINKY: { TIP: 20, DIP: 19, PIP: 18, MCP: 17 },
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
 * Calculate the centroid of a nail detection mask for accurate positioning.
 */
function calculateNailCentroid(detection: YoloDetection): [number, number] {
  if (detection.maskPolygon && detection.maskPolygon.length > 0) {
    let sumX = 0;
    let sumY = 0;
    for (const point of detection.maskPolygon) {
      sumX += point[0];
      sumY += point[1];
    }
    return [
      sumX / detection.maskPolygon.length,
      sumY / detection.maskPolygon.length,
    ];
  }
  // Fallback to the center of the bounding box
  const [x, y, width, height] = detection.bbox;
  return [x + width / 2, y + height / 2];
}

/**
 * FIXED: This is the core logic for calculating a stable 3D orientation basis.
 * It uses the hand's own geometry to create a robust coordinate system for the nail.
 */
function calculateOrientationBasis(
  hand: HandDetection,
  tipIndex: number
): NailFingerMatch["orientation"] | null {
  const lm = hand.landmarks;
  const indices = getFingerLandmarkIndices(tipIndex);
  if (!indices) return null;

  const { TIP, DIP, PIP } = indices;
  // Safety check: ensure all required landmark points exist.
  if ([TIP, DIP, PIP].some((i) => !lm[i])) return null;

  // Create THREE.Vector3 instances from landmark coordinates.
  const p_tip = new THREE.Vector3(lm[TIP].x, lm[TIP].y, lm[TIP].z);
  const p_dip = new THREE.Vector3(lm[DIP].x, lm[DIP].y, lm[DIP].z);
  const p_pip = new THREE.Vector3(lm[PIP].x, lm[PIP].y, lm[PIP].z);

  // --- Determine the 3D axes of the nail ---

  // 1. The z-axis (length) points along the finger's direction.
  //    We use the vector from the second joint (PIP) to the tip for a stable direction.
  const zAxis = p_tip.clone().sub(p_pip).normalize();

  // 2. The y-axis (normal) points "out" from the nail surface.
  //    We find it using the cross product of two vectors along the finger's surface.
  const v_pip_dip = p_dip.clone().sub(p_pip);
  const v_dip_tip = p_tip.clone().sub(p_dip);
  const yAxis = new THREE.Vector3()
    .crossVectors(v_pip_dip, v_dip_tip)
    .normalize();

  // MediaPipe's landmark winding order is consistent. For a left hand, this
  // results in a normal vector pointing "into" the finger. We must negate it
  // to ensure the normal always points "out", away from the nail bed.
  if (hand.handedness === "Left") {
    yAxis.negate();
  }

  // 3. The x-axis (width) is perpendicular to both the normal and length axes.
  //    We find it with another cross product to form a right-handed coordinate system.
  const xAxis = new THREE.Vector3().crossVectors(yAxis, zAxis).normalize();

  // To ensure the basis is perfectly orthogonal (in case the landmarks are not
  // perfectly co-planar), we re-calculate the z-axis from the new x and y axes.
  // This is a simplified Gram-Schmidt orthogonalization process.
  zAxis.crossVectors(xAxis, yAxis).normalize();

  return {
    xAxis: xAxis.toArray() as [number, number, number],
    yAxis: yAxis.toArray() as [number, number, number],
    zAxis: zAxis.toArray() as [number, number, number],
  };
}

/**
 * Calculates the nail's width and length by projecting its mask points onto the orientation axes.
 */
function calculateNailDimensions(
  detection: YoloDetection,
  orientation: NailFingerMatch["orientation"]
): { width: number; height: number; angle: number } {
  const zAxis2D = new THREE.Vector2(
    orientation.zAxis[0],
    orientation.zAxis[1]
  ).normalize();
  const xAxis2D = new THREE.Vector2(
    orientation.xAxis[0],
    orientation.xAxis[1]
  ).normalize();
  const angle = Math.atan2(zAxis2D.y, zAxis2D.x); // Angle for 2D drawing

  if (detection.maskPolygon && detection.maskPolygon.length > 4) {
    const polygon = detection.maskPolygon;
    const centroid = calculateNailCentroid(detection);
    const centroidV2 = new THREE.Vector2(centroid[0], centroid[1]);

    let minLength = Infinity,
      maxLength = -Infinity;
    let minWidth = Infinity,
      maxWidth = -Infinity;

    for (const p of polygon) {
      const pointV2 = new THREE.Vector2(p[0], p[1]);
      const d = pointV2.sub(centroidV2);

      // Project point onto the length and width axes
      minLength = Math.min(minLength, d.dot(zAxis2D));
      maxLength = Math.max(maxLength, d.dot(zAxis2D));
      minWidth = Math.min(minWidth, d.dot(xAxis2D));
      maxWidth = Math.max(maxWidth, d.dot(xAxis2D));
    }

    return {
      height: maxLength - minLength, // Total length along the finger
      width: maxWidth - minWidth, // Total width across the finger
      angle,
    };
  }

  // Fallback using the bounding box if mask isn't available
  const [, , bboxWidth, bboxHeight] = detection.bbox;
  return {
    height: Math.max(bboxWidth, bboxHeight), // Approximate length
    width: Math.min(bboxWidth, bboxHeight), // Approximate width
    angle,
  };
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
 * Main function to match nail detections to hand landmarks.
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

  const potentialMatches: (NailFingerMatch & { matchScore: number })[] = [];
  const maxDistance = Math.min(frameWidth, frameHeight) * 0.15; // Max search radius

  // Find all possible matches between nails and fingertips
  for (const detection of nailDetections) {
    const nailCentroid = calculateNailCentroid(detection);

    for (const [handIndex, hand] of handDetections.entries()) {
      for (const tipIndex of FINGER_TIPS) {
        if (!hand.landmarks[tipIndex]) continue;

        const fingertip = hand.landmarks[tipIndex];
        const fingertipPos: [number, number] = [
          fingertip.x * frameWidth,
          fingertip.y * frameHeight,
        ];

        const dist = distanceBetweenPoints(fingertipPos, nailCentroid);

        if (dist < maxDistance) {
          const orientation = calculateOrientationBasis(hand, tipIndex);
          if (!orientation) continue;

          const nailDimensions = calculateNailDimensions(
            detection,
            orientation
          );
          const distanceScore = 1 - dist / maxDistance;
          const matchScore = distanceScore * 0.7 + detection.score * 0.3;

          potentialMatches.push({
            nailDetection: detection,
            fingertipIndex: tipIndex,
            fingertipPosition: fingertipPos,
            nailCentroid,
            nailWidth: nailDimensions.width,
            nailHeight: nailDimensions.height,
            nailAngle: nailDimensions.angle,
            matchConfidence: detection.score, // Use raw detection confidence for display
            matchScore, // Internal score for finding the best match
            handIndex,
            handedness: hand.handedness as "Left" | "Right",
            orientation,
          });
        }
      }
    }
  }

  // Deduplication: Ensure each nail and each finger is used only once.
  // We sort by the match score so the most likely pairs are chosen first.
  potentialMatches.sort((a, b) => b.matchScore - a.matchScore);

  const finalMatches: NailFingerMatch[] = [];
  const usedNailIndices = new Set<number>();
  const usedFingerKeys = new Set<string>();

  for (const match of potentialMatches) {
    const nailIndex = nailDetections.indexOf(match.nailDetection);
    const fingerKey = `${match.handIndex}-${match.fingertipIndex}`;

    if (!usedNailIndices.has(nailIndex) && !usedFingerKeys.has(fingerKey)) {
      finalMatches.push(match);
      usedNailIndices.add(nailIndex);
      usedFingerKeys.add(fingerKey);
    }
  }

  return finalMatches;
}

/**
 * Draws debugging visuals for the nail-finger matches on a 2D canvas.
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

    // Draw connecting line
    ctx.strokeStyle = "#00ff88";
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 4]);
    ctx.beginPath();
    ctx.moveTo(scaledCentroid[0], scaledCentroid[1]);
    ctx.lineTo(scaledFingertip[0], scaledFingertip[1]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw dot on nail centroid
    ctx.fillStyle = "#00ff88";
    ctx.beginPath();
    ctx.arc(scaledCentroid[0], scaledCentroid[1], 4, 0, 2 * Math.PI);
    ctx.fill();

    // Draw label
    const fingerName =
      FINGER_NAMES[FINGER_TIPS.indexOf(match.fingertipIndex)] || "Unknown";
    const label = `${match.handedness} ${fingerName} (${(
      match.matchConfidence * 100
    ).toFixed(0)}%)`;
    ctx.font = "bold 12px Arial";
    const textWidth = ctx.measureText(label).width;

    ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
    ctx.fillRect(
      scaledFingertip[0] + 10,
      scaledFingertip[1] - 20,
      textWidth + 8,
      18
    );
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, scaledFingertip[0] + 14, scaledFingertip[1] - 7);
  });
}
