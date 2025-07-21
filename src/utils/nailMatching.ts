/**
 * Nail-Fingertip Matching and Orientation Analysis
 *
 * This module handles:
 * 1. Matching detected nails to specific fingertips
 * 2. Calculating nail orientation and dimensions
 * 3. Preparing data for future 3D model overlay
 */

import { YoloDetection } from "./yolo";
import { HandDetection } from "./mediapipe";

export interface NailFingerMatch {
  nailDetection: YoloDetection;
  fingertipIndex: number; // MediaPipe finger index (4=thumb, 8=index, 12=middle, 16=ring, 20=pinky)
  fingertipPosition: [number, number];
  fingerDirection: [number, number]; // Unit vector pointing from finger base toward tip (nail longitudinal axis)
  nailCentroid: [number, number];
  nailWidth: number; // Length along finger direction (longitudinal)
  nailHeight: number; // Width perpendicular to finger direction (transverse)
  nailAngle: number; // Rotation angle in radians (aligned with finger direction)
  nailPerpendicularDirection: [number, number]; // Unit vector perpendicular to finger direction
  matchConfidence: number; // How confident we are in this match (0-1)
  handIndex: number; // Which hand this belongs to
  handedness: "Left" | "Right";
}

// MediaPipe finger landmark indices
const FINGER_TIPS = [4, 8, 12, 16, 20]; // Thumb, Index, Middle, Ring, Pinky
const FINGER_BASES = [2, 5, 9, 13, 17]; // Corresponding base points for direction calculation
const FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"];

/**
 * Calculate the centroid of a nail detection
 */
function calculateNailCentroid(detection: YoloDetection): [number, number] {
  if (detection.maskPolygon && detection.maskPolygon.length > 0) {
    // Use mask polygon for precise centroid
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
    // Fallback to bounding box center
    const [x, y, width, height] = detection.bbox;
    return [x + width / 2, y + height / 2];
  }
}

/**
 * Calculate nail dimensions and orientation aligned with finger direction
 * Returns dimensions where:
 * - width = length along finger direction (longitudinal)
 * - height = width perpendicular to finger direction (transverse)
 * - angle = finger direction angle
 */
function calculateNailDimensions(
  detection: YoloDetection,
  fingerDirection: [number, number]
): {
  width: number;
  height: number;
  angle: number;
  perpendicularDirection: [number, number];
} {
  // Always use finger direction as the primary orientation
  const angle = Math.atan2(fingerDirection[1], fingerDirection[0]);
  const perpendicularDirection: [number, number] = [
    -fingerDirection[1],
    fingerDirection[0],
  ];

  if (detection.maskPolygon && detection.maskPolygon.length > 4) {
    // For polygons, project points onto finger direction axes to get dimensions
    const polygon = detection.maskPolygon;

    // Calculate centroid
    const centroidX =
      polygon.reduce((sum, p) => sum + p[0], 0) / polygon.length;
    const centroidY =
      polygon.reduce((sum, p) => sum + p[1], 0) / polygon.length;

    // Project all points onto finger direction and perpendicular direction
    let minLongitudinal = Infinity,
      maxLongitudinal = -Infinity;
    let minTransverse = Infinity,
      maxTransverse = -Infinity;

    polygon.forEach((point) => {
      const dx = point[0] - centroidX;
      const dy = point[1] - centroidY;

      // Project onto finger direction (longitudinal axis)
      const longitudinal = dx * fingerDirection[0] + dy * fingerDirection[1];
      minLongitudinal = Math.min(minLongitudinal, longitudinal);
      maxLongitudinal = Math.max(maxLongitudinal, longitudinal);

      // Project onto perpendicular direction (transverse axis)
      const transverse =
        dx * perpendicularDirection[0] + dy * perpendicularDirection[1];
      minTransverse = Math.min(minTransverse, transverse);
      maxTransverse = Math.max(maxTransverse, transverse);
    });

    const longitudinalLength = maxLongitudinal - minLongitudinal;
    const transverseLength = maxTransverse - minTransverse;

    return {
      width: longitudinalLength, // Length along finger
      height: transverseLength, // Width across finger
      angle: angle,
      perpendicularDirection: perpendicularDirection,
    };
  } else {
    // Fallback to bounding box aligned with finger direction
    const [, , bboxWidth, bboxHeight] = detection.bbox;

    // Use the larger dimension as the longitudinal (along finger) dimension
    // This is a reasonable assumption for nail shapes
    const longitudinalLength = Math.max(bboxWidth, bboxHeight);
    const transverseLength = Math.min(bboxWidth, bboxHeight);

    return {
      width: longitudinalLength,
      height: transverseLength,
      angle: angle,
      perpendicularDirection: perpendicularDirection,
    };
  }
}

/**
 * Calculate distance between two points
 */
function distanceBetweenPoints(
  p1: [number, number],
  p2: [number, number]
): number {
  const dx = p1[0] - p2[0];
  const dy = p1[1] - p2[1];
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Calculate finger direction vector from base to tip
 */
function calculateFingerDirection(
  hand: HandDetection,
  fingerIndex: number
): [number, number] {
  const tipIndex = FINGER_TIPS[fingerIndex];
  const baseIndex = FINGER_BASES[fingerIndex];

  if (tipIndex >= hand.landmarks.length || baseIndex >= hand.landmarks.length) {
    return [0, 1]; // Default upward direction
  }

  const tip = hand.landmarks[tipIndex];
  const base = hand.landmarks[baseIndex];

  const dx = tip.x - base.x;
  const dy = tip.y - base.y;
  const length = Math.sqrt(dx * dx + dy * dy);

  if (length === 0) return [0, 1];

  return [dx / length, dy / length];
}

/**
 * Match nails to fingertips using improved proximity and geometric constraints
 */
export function matchNailsToFingertips(
  nailDetections: YoloDetection[],
  handDetections: HandDetection[],
  frameWidth: number,
  frameHeight: number
): NailFingerMatch[] {
  const matches: NailFingerMatch[] = [];

  if (nailDetections.length === 0 || handDetections.length === 0) {
    return matches;
  }

  // Improved distance threshold based on frame size
  const maxReasonableDistance = Math.min(frameWidth, frameHeight) * 0.12; // Reduced to 12% for better precision
  const minNailSize = frameWidth * frameHeight * 0.0001; // Minimum nail size to consider
  const maxNailSize = frameWidth * frameHeight * 0.02; // Maximum reasonable nail size

  console.log(
    `Matching parameters: maxDistance=${maxReasonableDistance.toFixed(
      1
    )}, frameSize=${frameWidth}x${frameHeight}`
  );

  // For each hand
  handDetections.forEach((hand, handIndex) => {
    console.log(
      `Processing ${hand.handedness} hand with ${hand.landmarks.length} landmarks`
    );

    // For each fingertip
    FINGER_TIPS.forEach((tipIndex, fingerIndex) => {
      if (tipIndex >= hand.landmarks.length) return;

      const fingertip = hand.landmarks[tipIndex];
      const fingertipPos: [number, number] = [
        fingertip.x * frameWidth,
        fingertip.y * frameHeight,
      ];

      // Find the best nail detection for this fingertip
      let bestMatch: {
        detection: YoloDetection;
        distance: number;
        confidence: number;
      } | null = null;

      nailDetections.forEach((detection, detectionIndex) => {
        const nailCentroid = calculateNailCentroid(detection);
        const dist = distanceBetweenPoints(fingertipPos, nailCentroid);

        // Improved size filtering
        const nailArea = detection.bbox[2] * detection.bbox[3];
        if (nailArea < minNailSize || nailArea > maxNailSize) {
          console.log(
            `Skipping nail ${detectionIndex}: size ${nailArea.toFixed(
              0
            )} outside range [${minNailSize.toFixed(0)}, ${maxNailSize.toFixed(
              0
            )}]`
          );
          return;
        }

        // Enhanced confidence calculation with multiple factors
        const distanceScore = Math.max(0, 1 - dist / maxReasonableDistance);

        // Size score: prefer nails that are reasonable size for the frame
        const idealNailArea = frameWidth * frameHeight * 0.002; // 0.2% of frame
        const sizeRatio = Math.min(
          nailArea / idealNailArea,
          idealNailArea / nailArea
        );
        const sizeScore = Math.max(0.1, Math.min(1, sizeRatio));

        // Detection confidence from YOLO
        const detectionScore = Math.min(1, detection.score * 1.2); // Boost slightly

        // Position score: nails should be near the fingertip
        const positionScore = dist < maxReasonableDistance ? 1 : 0;

        // Combined confidence with weighted factors
        const confidence =
          distanceScore * 0.4 +
          sizeScore * 0.25 +
          detectionScore * 0.25 +
          positionScore * 0.1;

        console.log(
          `Nail ${detectionIndex} -> ${hand.handedness} ${
            FINGER_NAMES[fingerIndex]
          }: dist=${dist.toFixed(1)}, conf=${confidence.toFixed(
            2
          )} (d=${distanceScore.toFixed(2)}, s=${sizeScore.toFixed(
            2
          )}, det=${detectionScore.toFixed(2)})`
        );

        // Higher confidence threshold for better matching
        if (dist < maxReasonableDistance && confidence > 0.4) {
          if (!bestMatch || confidence > bestMatch.confidence) {
            bestMatch = {
              detection,
              distance: dist,
              confidence,
            };
          }
        }
      });

      // If we found a good match, create the nail-finger match
      if (bestMatch) {
        const fingerDirection = calculateFingerDirection(hand, fingerIndex);
        const nailCentroid = calculateNailCentroid(
          (bestMatch as any).detection
        );
        const nailDimensions = calculateNailDimensions(
          (bestMatch as any).detection,
          fingerDirection
        );

        const match: NailFingerMatch = {
          nailDetection: (bestMatch as any).detection,
          fingertipIndex: tipIndex,
          fingertipPosition: fingertipPos,
          fingerDirection: fingerDirection,
          nailCentroid: nailCentroid,
          nailWidth: nailDimensions.width,
          nailHeight: nailDimensions.height,
          nailAngle: nailDimensions.angle,
          nailPerpendicularDirection: nailDimensions.perpendicularDirection,
          matchConfidence: (bestMatch as any).confidence,
          handIndex: handIndex,
          handedness: hand.handedness as "Left" | "Right",
        };

        matches.push(match);

        console.log(
          `✓ Matched nail to ${hand.handedness} ${
            FINGER_NAMES[fingerIndex]
          } finger: conf=${match.matchConfidence.toFixed(3)}, angle=${(
            (match.nailAngle * 180) /
            Math.PI
          ).toFixed(1)}°`
        );
      }
    });
  });

  // Improved duplicate removal: prefer higher confidence and closer matches
  const uniqueMatches: NailFingerMatch[] = [];
  const usedNails = new Set<YoloDetection>();
  const usedFingers = new Set<string>(); // Track hand+finger combinations

  // Sort by confidence first, then by distance
  matches
    .sort((a, b) => {
      const confDiff = b.matchConfidence - a.matchConfidence;
      if (Math.abs(confDiff) > 0.05) return confDiff; // Significant confidence difference
      return (
        a.fingertipPosition[0] +
        a.fingertipPosition[1] -
        (b.fingertipPosition[0] + b.fingertipPosition[1])
      ); // Tie-break by position
    })
    .forEach((match) => {
      const fingerKey = `${match.handIndex}-${match.fingertipIndex}`;

      if (!usedNails.has(match.nailDetection) && !usedFingers.has(fingerKey)) {
        uniqueMatches.push(match);
        usedNails.add(match.nailDetection);
        usedFingers.add(fingerKey);
      } else {
        console.log(`Skipping duplicate: nail or finger already matched`);
      }
    });

  console.log(
    `Successfully matched ${uniqueMatches.length} nails to fingertips (from ${matches.length} candidates)`
  );

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
  matches.forEach((match, index) => {
    const scaledCentroid: [number, number] = [
      match.nailCentroid[0] * scaleX,
      match.nailCentroid[1] * scaleY,
    ];
    const scaledFingertip: [number, number] = [
      match.fingertipPosition[0] * scaleX,
      match.fingertipPosition[1] * scaleY,
    ];

    // Draw connection line with better visibility
    ctx.strokeStyle = "#00ff88";
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 4]);
    ctx.beginPath();
    ctx.moveTo(scaledCentroid[0], scaledCentroid[1]);
    ctx.lineTo(scaledFingertip[0], scaledFingertip[1]);
    ctx.stroke();

    // Draw nail centroid as a larger, more visible dot
    ctx.fillStyle = "#00ff88";
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(scaledCentroid[0], scaledCentroid[1], 6, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();

    // Draw fingertip position for reference
    ctx.fillStyle = "#0088ff";
    ctx.strokeStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(scaledFingertip[0], scaledFingertip[1], 4, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();

    // Draw improved nail orientation indicators
    // Primary axis (longitudinal - along finger direction) in magenta
    const dirLength = Math.max(
      30,
      Math.min(match.nailWidth, match.nailHeight) *
        Math.max(scaleX, scaleY) *
        0.6
    );
    const dirX = Math.cos(match.nailAngle) * dirLength;
    const dirY = Math.sin(match.nailAngle) * dirLength;

    // Longitudinal axis (finger direction) - thicker magenta line
    ctx.strokeStyle = "#ff00ff";
    ctx.lineWidth = 4;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(scaledCentroid[0] - dirX, scaledCentroid[1] - dirY);
    ctx.lineTo(scaledCentroid[0] + dirX, scaledCentroid[1] + dirY);
    ctx.stroke();

    // Perpendicular axis (transverse - across finger) in cyan
    const perpDirX = match.nailPerpendicularDirection[0] * (dirLength * 0.7);
    const perpDirY = match.nailPerpendicularDirection[1] * (dirLength * 0.7);

    ctx.strokeStyle = "#00ffff";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(scaledCentroid[0] - perpDirX, scaledCentroid[1] - perpDirY);
    ctx.lineTo(scaledCentroid[0] + perpDirX, scaledCentroid[1] + perpDirY);
    ctx.stroke();

    // Add arrow heads to show longitudinal direction
    const arrowSize = 8;
    const arrowAngle = Math.PI / 6; // 30 degrees

    // Arrow head at the positive end of longitudinal axis
    const endX = scaledCentroid[0] + dirX;
    const endY = scaledCentroid[1] + dirY;

    ctx.strokeStyle = "#ff00ff";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(endX, endY);
    ctx.lineTo(
      endX - arrowSize * Math.cos(match.nailAngle - arrowAngle),
      endY - arrowSize * Math.sin(match.nailAngle - arrowAngle)
    );
    ctx.moveTo(endX, endY);
    ctx.lineTo(
      endX - arrowSize * Math.cos(match.nailAngle + arrowAngle),
      endY - arrowSize * Math.sin(match.nailAngle + arrowAngle)
    );
    ctx.stroke();

    // Draw finger direction reference line (from fingertip, orange dashed)
    const fingerDirLength = dirLength * 0.8;
    const fingerDirX = match.fingerDirection[0] * fingerDirLength;
    const fingerDirY = match.fingerDirection[1] * fingerDirLength;

    ctx.strokeStyle = "#ffaa00";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 3]);
    ctx.beginPath();
    ctx.moveTo(scaledFingertip[0], scaledFingertip[1]);
    ctx.lineTo(
      scaledFingertip[0] + fingerDirX,
      scaledFingertip[1] + fingerDirY
    );
    ctx.stroke();

    // Draw labels with orientation info
    const fingerName =
      FINGER_NAMES[FINGER_TIPS.indexOf(match.fingertipIndex)] || "Unknown";
    const label = `${match.handedness} ${fingerName}`;
    const confLabel = `${(match.matchConfidence * 100).toFixed(0)}%`;
    const angleLabel = `${((match.nailAngle * 180) / Math.PI).toFixed(0)}°`;
    const dimensionLabel = `L:${match.nailWidth.toFixed(
      0
    )} T:${match.nailHeight.toFixed(0)}`;

    ctx.font = "bold 12px Arial";
    const labelWidth =
      Math.max(
        ctx.measureText(label).width,
        ctx.measureText(confLabel).width,
        ctx.measureText(angleLabel).width,
        ctx.measureText(dimensionLabel).width
      ) + 8;

    // Background for better readability
    ctx.fillStyle = "rgba(0, 0, 0, 0.9)";
    ctx.fillRect(
      scaledCentroid[0] + 12,
      scaledCentroid[1] - 45,
      labelWidth,
      60
    );

    // Text labels with color coding
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, scaledCentroid[0] + 16, scaledCentroid[1] - 30);
    ctx.fillStyle = "#00ff88";
    ctx.fillText(confLabel, scaledCentroid[0] + 16, scaledCentroid[1] - 15);
    ctx.fillStyle = "#ff00ff";
    ctx.fillText(
      `Long: ${angleLabel}`,
      scaledCentroid[0] + 16,
      scaledCentroid[1]
    );
    ctx.fillStyle = "#00ffff";
    ctx.fillText(
      dimensionLabel,
      scaledCentroid[0] + 16,
      scaledCentroid[1] + 15
    );
  });
}
