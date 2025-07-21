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
  fingerDirection: [number, number]; // Unit vector pointing from finger base toward tip
  nailCentroid: [number, number];
  nailWidth: number;
  nailHeight: number;
  nailAngle: number; // Rotation angle in radians
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
 * Calculate nail dimensions from detection
 */
function calculateNailDimensions(detection: YoloDetection): {
  width: number;
  height: number;
  angle: number;
} {
  if (detection.maskPolygon && detection.maskPolygon.length > 4) {
    // For polygons, find the oriented bounding box
    const polygon = detection.maskPolygon;

    // Simple approach: find the longest and shortest distances between points
    let maxDistance = 0;
    let minDistance = Infinity;
    let primaryAxis: [number, number] = [1, 0];

    for (let i = 0; i < polygon.length; i++) {
      for (let j = i + 1; j < polygon.length; j++) {
        const dx = polygon[j][0] - polygon[i][0];
        const dy = polygon[j][1] - polygon[i][1];
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance > maxDistance) {
          maxDistance = distance;
          // Store the direction of the longest axis (likely nail length)
          primaryAxis = [dx / distance, dy / distance];
        }
        if (distance < minDistance && distance > 5) {
          // Avoid noise
          minDistance = distance;
        }
      }
    }

    // Calculate angle from primary axis
    const angle = Math.atan2(primaryAxis[1], primaryAxis[0]);

    return {
      width: Math.max(maxDistance * 0.6, minDistance), // Conservative estimate
      height: Math.min(maxDistance * 0.4, minDistance),
      angle: angle,
    };
  } else {
    // Fallback to bounding box
    const [, , width, height] = detection.bbox;
    return {
      width: Math.max(width, height), // Assume longer dimension is width
      height: Math.min(width, height),
      angle: width > height ? 0 : Math.PI / 2, // Guess orientation
    };
  }
}

/**
 * Calculate distance between two points
 */
function distance(p1: [number, number], p2: [number, number]): number {
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
        const dist = distance(fingertipPos, nailCentroid);

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
        const nailCentroid = calculateNailCentroid(bestMatch.detection);
        const nailDimensions = calculateNailDimensions(bestMatch.detection);

        const match: NailFingerMatch = {
          nailDetection: bestMatch.detection,
          fingertipIndex: tipIndex,
          fingertipPosition: fingertipPos,
          fingerDirection: fingerDirection,
          nailCentroid: nailCentroid,
          nailWidth: nailDimensions.width,
          nailHeight: nailDimensions.height,
          nailAngle: nailDimensions.angle,
          matchConfidence: bestMatch.confidence,
          handIndex: handIndex,
          handedness: hand.handedness as "Left" | "Right",
        };

        matches.push(match);

        console.log(
          `✓ Matched nail to ${hand.handedness} ${FINGER_NAMES[fingerIndex]} finger:`,
          {
            confidence: bestMatch.confidence.toFixed(3),
            distance: bestMatch.distance.toFixed(1),
            nailSize: `${nailDimensions.width.toFixed(
              1
            )}×${nailDimensions.height.toFixed(1)}`,
            angle: `${((nailDimensions.angle * 180) / Math.PI).toFixed(1)}°`,
          }
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
 * Draw nail-finger matches for debugging
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

    // Draw connection line
    ctx.strokeStyle = "#00ff00";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(scaledCentroid[0], scaledCentroid[1]);
    ctx.lineTo(scaledFingertip[0], scaledFingertip[1]);
    ctx.stroke();

    // Draw nail centroid
    ctx.fillStyle = "#00ff00";
    ctx.beginPath();
    ctx.arc(scaledCentroid[0], scaledCentroid[1], 4, 0, 2 * Math.PI);
    ctx.fill();

    // Draw nail orientation indicator
    const dirLength = 20;
    const dirX = Math.cos(match.nailAngle) * dirLength;
    const dirY = Math.sin(match.nailAngle) * dirLength;

    ctx.strokeStyle = "#ff00ff";
    ctx.lineWidth = 3;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(scaledCentroid[0] - dirX, scaledCentroid[1] - dirY);
    ctx.lineTo(scaledCentroid[0] + dirX, scaledCentroid[1] + dirY);
    ctx.stroke();

    // Draw confidence and finger info
    const fingerName =
      FINGER_NAMES[FINGER_TIPS.indexOf(match.fingertipIndex)] || "Unknown";
    const label = `${match.handedness} ${fingerName} (${(
      match.matchConfidence * 100
    ).toFixed(0)}%)`;

    ctx.font = "bold 11px Arial";
    ctx.fillStyle = "#000000";
    ctx.fillRect(
      scaledCentroid[0] + 8,
      scaledCentroid[1] - 25,
      ctx.measureText(label).width + 6,
      20
    );
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, scaledCentroid[0] + 11, scaledCentroid[1] - 10);
  });
}
