/**
 * Nail-Fingertip Matching and Orientation Analysis (REVISED & IMPROVED)
 *
 * This module handles:
 * 1. Matching detected nails to specific fingertips using distance and confidence.
 * 2. Calculating a full 3D orientation basis (X, Y, Z axes) for each nail.
 * 3. Calculating nail dimensions (width, length) and 2D angle for drawing.
 * 4. Preparing robust data for the 3D model overlay.
 */

import { YoloDetection } from "./yolo";
import { HandDetection } from "./mediapipe";
import * as THREE from "three"; // Using THREE's Vector3 for robust vector math

export interface NailFingerMatch {
  nailDetection: YoloDetection;
  fingertipIndex: number; // MediaPipe landmark index (e.g., 4, 8, 12, 16, 20)
  fingertipPosition: [number, number];
  nailCentroid: [number, number];
  nailWidth: number; // Across the finger
  nailHeight: number; // Along the finger
  nailAngle: number; // 2D rotation in radians for flat overlays
  matchConfidence: number;
  handIndex: number;
  handedness: "Left" | "Right";
  orientation: {
    // A right-handed coordinate system for the nail
    xAxis: [number, number, number]; // Points across the nail width (local X)
    yAxis: [number, number, number]; // Points out from the nail surface, the normal (local Y)
    zAxis: [number, number, number]; // Points along the finger length (local Z)
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

  const { TIP, DIP, PIP, MCP } = indices;
  // Use MCP for thumb's base to get a more stable Z-axis
  const basePointIndex = tipIndex === FINGER_LANDMARKS.THUMB.TIP ? MCP : PIP;

  // Safety check: ensure all required landmark points exist.
  if ([TIP, DIP, basePointIndex].some((i) => !lm[i])) return null;

  // Create THREE.Vector3 instances from landmark coordinates for vector operations.
  const p_tip = new THREE.Vector3(lm[TIP].x, lm[TIP].y, lm[TIP].z);
  const p_dip = new THREE.Vector3(lm[DIP].x, lm[DIP].y, lm[DIP].z);
  const p_base = new THREE.Vector3(
    lm[basePointIndex].x,
    lm[basePointIndex].y,
    lm[basePointIndex].z
  );

  // --- Determine the 3D axes of the nail ---

  // 1. The z-axis (length) points along the finger's direction.
  //    We use the vector from a base joint to the tip for a stable direction.
  const zAxis = new THREE.Vector3().subVectors(p_tip, p_base).normalize();

  // 2. The y-axis (normal) points "out" from the nail surface.
  //    We find it using the cross product of two vectors along the finger's surface plane.
  //    This creates a vector perpendicular to the plane of the finger tip.
  const v_base_dip = new THREE.Vector3().subVectors(p_dip, p_base);
  const v_dip_tip = new THREE.Vector3().subVectors(p_tip, p_dip);
  const yAxis = new THREE.Vector3()
    .crossVectors(v_base_dip, v_dip_tip)
    .normalize();

  // MediaPipe's landmark winding order is consistent. For a left hand, the raw
  // cross product points "into" the finger. We must negate it to ensure the
  // normal (yAxis) always points "out", away from the nail bed.
  if (hand.handedness === "Left") {
    yAxis.negate();
  }

  // 3. The x-axis (width) is perpendicular to both the normal (y) and length (z) axes.
  //    We find it with another cross product to form a right-handed coordinate system.
  //    Order (y, z) is important here for a right-handed system (X = Y x Z).
  const xAxis = new THREE.Vector3().crossVectors(yAxis, zAxis).normalize();

  // 4. (CRUCIAL) Gram-Schmidt Orthogonalization:
  //    To ensure the basis is perfectly orthogonal (no skew), we recalculate the z-axis
  //    from the new x and y axes. This corrects for any minor non-orthogonality from
  //    the landmark data.
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
  // Project 3D axes to 2D plane for dimension/angle calculation
  const zAxis2D = new THREE.Vector2(
    orientation.zAxis[0],
    orientation.zAxis[1]
  ).normalize();
  const xAxis2D = new THREE.Vector2(
    orientation.xAxis[0],
    orientation.xAxis[1]
  ).normalize();

  // The 2D angle of the nail is the angle of its length vector (z-axis).
  // This is used for simple 2D overlays.
  const angle = Math.atan2(zAxis2D.y, zAxis2D.x);

  if (detection.maskPolygon && detection.maskPolygon.length > 4) {
    const polygon = detection.maskPolygon;
    const centroid = calculateNailCentroid(detection);
    const centroidV2 = new THREE.Vector2(centroid[0], centroid[1]);

    let minLength = Infinity,
      maxLength = -Infinity;
    let minWidth = Infinity,
      maxWidth = -Infinity;

    // Project each point of the mask onto the nail's local axes to find the extents.
    for (const p of polygon) {
      const pointV2 = new THREE.Vector2(p[0], p[1]);
      const d = pointV2.sub(centroidV2);

      // Dot product projects the point onto the axis vector.
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

  // Fallback using the bounding box if mask isn't available.
  const [, , bboxWidth, bboxHeight] = detection.bbox;
  // Use the angle to better estimate which dimension is width vs. height.
  const cosAngle = Math.abs(Math.cos(angle));
  const sinAngle = Math.abs(Math.sin(angle));
  const height = bboxWidth * cosAngle + bboxHeight * sinAngle;
  const width = bboxWidth * sinAngle + bboxHeight * cosAngle;

  return {
    height,
    width,
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
          // Combine distance and detection confidence for a robust match score.
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
  const usedFingerKeys = new Set<string>(); // e.g., "0-8" for hand 0, index finger

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
    const labelX = scaledFingertip[0] + 10;
    const labelY = scaledFingertip[1] - 8;

    ctx.fillStyle = "rgba(0, 255, 136, 0.75)";
    ctx.fillRect(labelX - 4, labelY - 14, textWidth + 8, 20);
    ctx.fillStyle = "#000000";
    ctx.fillText(label, labelX, labelY);
  });
}
