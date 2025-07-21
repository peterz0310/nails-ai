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
 * Calculate nail dimensions and orientation with finger direction constraint
 */
function calculateNailDimensions(
  detection: YoloDetection,
  fingerDirection?: [number, number]
): {
  width: number;
  height: number;
  angle: number;
} {
  if (detection.maskPolygon && detection.maskPolygon.length > 4) {
    // For polygons, use Principal Component Analysis for better orientation
    const polygon = detection.maskPolygon;

    // Calculate centroid
    const centroidX =
      polygon.reduce((sum, p) => sum + p[0], 0) / polygon.length;
    const centroidY =
      polygon.reduce((sum, p) => sum + p[1], 0) / polygon.length;

    // Calculate covariance matrix elements
    let cxx = 0,
      cxy = 0,
      cyy = 0;
    polygon.forEach((point) => {
      const dx = point[0] - centroidX;
      const dy = point[1] - centroidY;
      cxx += dx * dx;
      cxy += dx * dy;
      cyy += dy * dy;
    });

    // Normalize by number of points
    cxx /= polygon.length;
    cxy /= polygon.length;
    cyy /= polygon.length;

    // Find eigenvalues and eigenvectors
    const trace = cxx + cyy;
    const det = cxx * cyy - cxy * cxy;
    const lambda1 = (trace + Math.sqrt(trace * trace - 4 * det)) / 2;
    const lambda2 = (trace - Math.sqrt(trace * trace - 4 * det)) / 2;

    // Principal direction (eigenvector corresponding to larger eigenvalue)
    let primaryDirection: [number, number];
    if (Math.abs(cxy) > 1e-6) {
      primaryDirection = [lambda1 - cyy, cxy];
      const norm = Math.sqrt(
        primaryDirection[0] * primaryDirection[0] +
          primaryDirection[1] * primaryDirection[1]
      );
      primaryDirection = [
        primaryDirection[0] / norm,
        primaryDirection[1] / norm,
      ];
    } else {
      primaryDirection = cxx > cyy ? [1, 0] : [0, 1];
    }

    // If we have finger direction, constrain nail orientation
    let finalDirection = primaryDirection;
    if (fingerDirection) {
      // Calculate dot product to see alignment
      const dot =
        primaryDirection[0] * fingerDirection[0] +
        primaryDirection[1] * fingerDirection[1];

      // If the directions are more perpendicular than parallel, flip the nail direction
      if (Math.abs(dot) < 0.7) {
        // Allow some deviation but prefer alignment
        // Try the perpendicular direction
        const perpDirection: [number, number] = [
          -primaryDirection[1],
          primaryDirection[0],
        ];
        const perpDot =
          perpDirection[0] * fingerDirection[0] +
          perpDirection[1] * fingerDirection[1];

        if (Math.abs(perpDot) > Math.abs(dot)) {
          finalDirection = perpDirection;
        }
      }

      // Ensure direction points roughly in the same direction as finger
      const finalDot =
        finalDirection[0] * fingerDirection[0] +
        finalDirection[1] * fingerDirection[1];
      if (finalDot < 0) {
        finalDirection = [-finalDirection[0], -finalDirection[1]];
      }
    }

    const angle = Math.atan2(finalDirection[1], finalDirection[0]);

    // Calculate dimensions based on eigenvalues
    const width = Math.sqrt(lambda1) * 4; // Scale factor to get reasonable dimensions
    const height = Math.sqrt(lambda2) * 4;

    return {
      width: Math.max(width, height), // Ensure width is the longer dimension
      height: Math.min(width, height),
      angle: width > height ? angle : angle + Math.PI / 2,
    };
  } else {
    // Fallback to bounding box with finger direction constraint
    const [, , width, height] = detection.bbox;
    let angle = width > height ? 0 : Math.PI / 2;

    // If we have finger direction, align with it
    if (fingerDirection) {
      angle = Math.atan2(fingerDirection[1], fingerDirection[0]);
    }

    return {
      width: Math.max(width, height),
      height: Math.min(width, height),
      angle: angle,
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
      if (bestMatch !== null) {
        const fingerDirection = calculateFingerDirection(hand, fingerIndex);
        // @ts-ignore - TypeScript has trouble with the null check but this is safe
        const nailCentroid = calculateNailCentroid(bestMatch.detection);
        // @ts-ignore - TypeScript has trouble with the null check but this is safe
        const nailDimensions = calculateNailDimensions(
          bestMatch.detection,
          fingerDirection
        );

        const match: NailFingerMatch = {
          // @ts-ignore - TypeScript has trouble with the null check but this is safe
          nailDetection: bestMatch.detection,
          fingertipIndex: tipIndex,
          fingertipPosition: fingertipPos,
          fingerDirection: fingerDirection,
          nailCentroid: nailCentroid,
          nailWidth: nailDimensions.width,
          nailHeight: nailDimensions.height,
          nailAngle: nailDimensions.angle,
          // @ts-ignore - TypeScript has trouble with the null check but this is safe
          matchConfidence: bestMatch.confidence,
          handIndex: handIndex,
          handedness: hand.handedness as "Left" | "Right",
        };

        matches.push(match);

        console.log(
          `✓ Matched nail to ${hand.handedness} ${FINGER_NAMES[fingerIndex]} finger:`,
          {
            // @ts-ignore - TypeScript has trouble with the null check but this is safe
            confidence: bestMatch.confidence.toFixed(3),
            // @ts-ignore - TypeScript has trouble with the null check but this is safe
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

    // Draw improved nail orientation indicator
    const dirLength = Math.max(
      30,
      Math.min(match.nailWidth, match.nailHeight) *
        Math.max(scaleX, scaleY) *
        0.6
    );
    const dirX = Math.cos(match.nailAngle) * dirLength;
    const dirY = Math.sin(match.nailAngle) * dirLength;

    // Main orientation line (thicker, more visible)
    ctx.strokeStyle = "#ff00ff";
    ctx.lineWidth = 4;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(scaledCentroid[0] - dirX, scaledCentroid[1] - dirY);
    ctx.lineTo(scaledCentroid[0] + dirX, scaledCentroid[1] + dirY);
    ctx.stroke();

    // Add arrow heads to show direction
    const arrowSize = 8;
    const arrowAngle = Math.PI / 6; // 30 degrees

    // Arrow head at the positive end
    const endX = scaledCentroid[0] + dirX;
    const endY = scaledCentroid[1] + dirY;

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

    // Draw finger direction for comparison (thinner, different color)
    const fingerDirLength = dirLength * 0.8;
    const fingerDirX = match.fingerDirection[0] * fingerDirLength;
    const fingerDirY = match.fingerDirection[1] * fingerDirLength;

    ctx.strokeStyle = "#ffaa00";
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(scaledFingertip[0], scaledFingertip[1]);
    ctx.lineTo(
      scaledFingertip[0] + fingerDirX,
      scaledFingertip[1] + fingerDirY
    );
    ctx.stroke();

    // Draw confidence and finger info with better styling
    const fingerName =
      FINGER_NAMES[FINGER_TIPS.indexOf(match.fingertipIndex)] || "Unknown";
    const label = `${match.handedness} ${fingerName}`;
    const confLabel = `${(match.matchConfidence * 100).toFixed(0)}%`;
    const angleLabel = `${((match.nailAngle * 180) / Math.PI).toFixed(0)}°`;

    ctx.font = "bold 12px Arial";
    const labelWidth =
      Math.max(
        ctx.measureText(label).width,
        ctx.measureText(confLabel).width,
        ctx.measureText(angleLabel).width
      ) + 8;

    // Background for better readability
    ctx.fillStyle = "rgba(0, 0, 0, 0.8)";
    ctx.fillRect(
      scaledCentroid[0] + 12,
      scaledCentroid[1] - 35,
      labelWidth,
      45
    );

    // Text labels
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, scaledCentroid[0] + 16, scaledCentroid[1] - 20);
    ctx.fillStyle = "#00ff88";
    ctx.fillText(confLabel, scaledCentroid[0] + 16, scaledCentroid[1] - 5);
    ctx.fillStyle = "#ff00ff";
    ctx.fillText(angleLabel, scaledCentroid[0] + 16, scaledCentroid[1] + 10);
  });
}
