import * as tf from "@tensorflow/tfjs";

export interface YoloDetection {
  bbox: number[]; // [x, y, width, height]
  score: number;
  class: number;
  mask?: number[][]; // 2D mask array
  maskPolygon?: number[][]; // Array of [x, y] points for accurate boundary
}

export interface YoloOutput {
  detections: YoloDetection[];
  maskWidth: number;
  maskHeight: number;
}

/**
 * Process YOLOv8 segmentation model output
 * Expected output format from YOLOv8-seg:
 * - Output 0: [1, features, 8400] where features include [x,y,w,h,conf,mask_coeffs...]
 * - Output 1: [1, mask_channels, mask_height, mask_width] segmentation prototypes
 */
export function processYoloOutput(
  outputs: tf.Tensor[],
  inputWidth: number,
  inputHeight: number,
  confidenceThreshold: number = 0.25,
  nmsThreshold: number = 0.45
): YoloOutput {
  const detections: YoloDetection[] = [];

  if (outputs.length < 1) {
    console.log("No outputs from model");
    return { detections, maskWidth: 0, maskHeight: 0 };
  }

  try {
    const predictions = outputs[0];
    const maskPrototypes = outputs.length > 1 ? outputs[1] : null;

    console.log("Predictions shape:", predictions.shape);
    console.log("Predictions dtype:", predictions.dtype);

    if (maskPrototypes) {
      console.log("Mask prototypes shape:", maskPrototypes.shape);
      console.log("Mask prototypes dtype:", maskPrototypes.dtype);
    }

    // Handle different possible output shapes
    let processedPredictions = predictions;
    let numDetections: number;
    let numFeatures: number;

    if (predictions.shape.length === 3) {
      if (predictions.shape[1] === 8400 && predictions.shape[2] > 100) {
        // Shape is [1, 8400, features] - need to transpose
        processedPredictions = predictions.transpose([0, 2, 1]);
        console.log("Transposed to shape:", processedPredictions.shape);
      }
      numFeatures = processedPredictions.shape[1] || 0;
      numDetections = processedPredictions.shape[2] || 0;
    } else {
      console.error("Unexpected prediction shape:", predictions.shape);
      return { detections, maskWidth: 0, maskHeight: 0 };
    }

    if (numDetections === 0 || numFeatures === 0) {
      console.error("Invalid dimensions:", numFeatures, numDetections);
      return { detections, maskWidth: 0, maskHeight: 0 };
    }

    const predictionData = processedPredictions.dataSync();

    // Determine if this model has mask coefficients
    const hasMaskCoeffs = numFeatures > 5; // Basic YOLO has 5 features (x,y,w,h,conf), segmentation has more
    const numMaskCoeffs = hasMaskCoeffs ? Math.max(0, numFeatures - 5) : 0;

    console.log(
      `Processing: features=${numFeatures}, detections=${numDetections}, maskCoeffs=${numMaskCoeffs}`
    );
    console.log("First few prediction values:", predictionData.slice(0, 20));

    const validDetections: any[] = [];

    for (let i = 0; i < Math.min(numDetections, 8400); i++) {
      // Access data in column-major order: feature_index * numDetections + detection_index
      const centerX = predictionData[0 * numDetections + i]; // x_center
      const centerY = predictionData[1 * numDetections + i]; // y_center
      const width = predictionData[2 * numDetections + i]; // width
      const height = predictionData[3 * numDetections + i]; // height

      // For single class YOLO, confidence is typically at index 4
      const confidence = predictionData[4 * numDetections + i];

      if (confidence > confidenceThreshold) {
        // Log the raw normalized values for debugging
        console.log(`Raw detection ${i}:`, {
          centerX,
          centerY,
          width,
          height,
          confidence,
          inputDims: { inputWidth, inputHeight },
        });

        // YOLO outputs coordinates in pixels relative to 640x640 input
        // First normalize to 0-1, then scale to video dimensions
        const normalizedCenterX = centerX / 640;
        const normalizedCenterY = centerY / 640;
        const normalizedWidth = width / 640;
        const normalizedHeight = height / 640;

        // Convert to video pixel coordinates
        const pixelCenterX = normalizedCenterX * inputWidth;
        const pixelCenterY = normalizedCenterY * inputHeight;
        const pixelWidth = normalizedWidth * inputWidth;
        const pixelHeight = normalizedHeight * inputHeight;

        // Convert center coordinates to top-left coordinates
        const x = Math.max(0, pixelCenterX - pixelWidth / 2);
        const y = Math.max(0, pixelCenterY - pixelHeight / 2);

        console.log(`Converted detection ${i}:`, {
          normalized: [
            normalizedCenterX,
            normalizedCenterY,
            normalizedWidth,
            normalizedHeight,
          ],
          pixels: [pixelCenterX, pixelCenterY, pixelWidth, pixelHeight],
          bbox: [x, y, pixelWidth, pixelHeight],
        });

        // Extract mask coefficients if available
        let maskCoeffs: number[] = [];
        if (hasMaskCoeffs && numMaskCoeffs > 0) {
          for (let j = 0; j < numMaskCoeffs; j++) {
            const coeffIndex = (5 + j) * numDetections + i;
            if (coeffIndex < predictionData.length) {
              maskCoeffs.push(predictionData[coeffIndex]);
            }
          }

          // Log mask coefficients for debugging
          if (maskCoeffs.length > 0) {
            console.log(
              `Detection ${i} mask coefficients:`,
              maskCoeffs.slice(0, Math.min(5, maskCoeffs.length))
            );
          }
        }

        validDetections.push({
          bbox: [x, y, pixelWidth, pixelHeight],
          score: confidence,
          class: 0, // Nail class
          confidence: confidence,
          index: i,
          maskCoeffs: maskCoeffs,
        });
      }
    }

    console.log(
      `Found ${validDetections.length} valid detections above threshold ${confidenceThreshold}`
    );

    // Apply Non-Maximum Suppression
    const nmsDetections = applyNMS(validDetections, nmsThreshold);
    console.log(`After NMS: ${nmsDetections.length} detections`);

    // Process masks if available
    const processedDetections = processMasks(
      nmsDetections,
      maskPrototypes,
      inputWidth,
      inputHeight
    );

    // Clean up transposed tensor if we created one
    if (processedPredictions !== predictions) {
      processedPredictions.dispose();
    }

    return {
      detections: processedDetections,
      maskWidth: maskPrototypes ? maskPrototypes.shape[3] || 160 : 160,
      maskHeight: maskPrototypes ? maskPrototypes.shape[2] || 160 : 160,
    };
  } catch (error) {
    console.error("Error processing YOLO output:", error);
    return { detections, maskWidth: 0, maskHeight: 0 };
  }
}

/**
 * Process segmentation masks from mask prototypes and coefficients
 */
function processMasks(
  detections: any[],
  maskPrototypes: tf.Tensor | null,
  inputWidth: number,
  inputHeight: number
): YoloDetection[] {
  if (!maskPrototypes || detections.length === 0) {
    // Return detections without masks
    return detections.map((d) => ({
      bbox: d.bbox,
      score: d.score,
      class: d.class,
    }));
  }

  try {
    const processedDetections: YoloDetection[] = [];

    for (const detection of detections) {
      const { maskCoeffs } = detection;

      if (maskCoeffs && maskCoeffs.length > 0) {
        // Generate mask for this detection
        const mask = generateMaskFromCoeffs(
          maskCoeffs,
          maskPrototypes,
          detection.bbox,
          inputWidth,
          inputHeight
        );

        processedDetections.push({
          bbox: detection.bbox,
          score: detection.score,
          class: detection.class,
          mask: mask.mask,
          maskPolygon: mask.polygon,
        });
      } else {
        // No mask available, just return the detection
        processedDetections.push({
          bbox: detection.bbox,
          score: detection.score,
          class: detection.class,
        });
      }
    }

    return processedDetections;
  } catch (error) {
    console.error("Error processing masks:", error);
    // Return detections without masks in case of error
    return detections.map((d) => ({
      bbox: d.bbox,
      score: d.score,
      class: d.class,
    }));
  }
}

/**
 * Generate a mask from mask coefficients and prototypes
 */
function generateMaskFromCoeffs(
  coeffs: number[],
  prototypes: tf.Tensor | null,
  bbox: number[],
  inputWidth: number,
  inputHeight: number
): {
  mask: number[][];
  polygon: number[][];
  imageData: ImageData | null;
} {
  try {
    const [bboxX, bboxY, bboxWidth, bboxHeight] = bbox;

    // If we have mask coefficients and prototypes, use them for precise mask generation
    if (coeffs.length > 0 && prototypes) {
      console.log("Using actual mask coefficients for precise segmentation");
      return generateActualMask(
        coeffs,
        prototypes,
        bbox,
        inputWidth,
        inputHeight
      );
    }

    console.log("Falling back to shape-based mask generation");

    // Fallback: Create a more realistic nail-shaped mask
    const maskSize = 64; // Higher resolution for better detail
    const mask2D: number[][] = [];

    for (let y = 0; y < maskSize; y++) {
      const row: number[] = [];
      for (let x = 0; x < maskSize; x++) {
        // Create a nail-shaped mask (rounded rectangle)
        const normalizedX = x / maskSize;
        const normalizedY = y / maskSize;

        // Create rounded rectangle shape for nail
        const padding = 0.1;
        const cornerRadius = 0.3;

        let maskValue = 0;

        if (
          normalizedX >= padding &&
          normalizedX <= 1 - padding &&
          normalizedY >= padding &&
          normalizedY <= 1 - padding
        ) {
          // Calculate distance from edges for rounded corners
          const distFromLeft = normalizedX - padding;
          const distFromRight = 1 - padding - normalizedX;
          const distFromTop = normalizedY - padding;
          const distFromBottom = 1 - padding - normalizedY;

          const width = 1 - 2 * padding;
          const height = 1 - 2 * padding;

          // Check if we're in a corner region
          if (
            (distFromLeft < cornerRadius && distFromTop < cornerRadius) ||
            (distFromRight < cornerRadius && distFromTop < cornerRadius) ||
            (distFromLeft < cornerRadius && distFromBottom < cornerRadius) ||
            (distFromRight < cornerRadius && distFromBottom < cornerRadius)
          ) {
            // Calculate distance from nearest corner
            let cornerX, cornerY;
            if (distFromLeft < cornerRadius && distFromTop < cornerRadius) {
              cornerX = padding + cornerRadius;
              cornerY = padding + cornerRadius;
            } else if (
              distFromRight < cornerRadius &&
              distFromTop < cornerRadius
            ) {
              cornerX = 1 - padding - cornerRadius;
              cornerY = padding + cornerRadius;
            } else if (
              distFromLeft < cornerRadius &&
              distFromBottom < cornerRadius
            ) {
              cornerX = padding + cornerRadius;
              cornerY = 1 - padding - cornerRadius;
            } else {
              cornerX = 1 - padding - cornerRadius;
              cornerY = 1 - padding - cornerRadius;
            }

            const distFromCorner = Math.sqrt(
              Math.pow(normalizedX - cornerX, 2) +
                Math.pow(normalizedY - cornerY, 2)
            );

            if (distFromCorner <= cornerRadius) {
              maskValue = Math.max(
                0,
                1 - (distFromCorner / cornerRadius) * 0.3
              );
            }
          } else {
            // We're in the main body of the nail
            maskValue = 1;
          }
        }

        row.push(maskValue);
      }
      mask2D.push(row);
    }

    // Generate polygon from mask contours
    const polygon = extractPolygonFromMask(
      mask2D,
      bbox,
      inputWidth,
      inputHeight
    );

    return {
      mask: mask2D,
      polygon: polygon,
      imageData: null, // Don't pre-generate ImageData with fixed colors
    };
  } catch (error) {
    console.error("Error generating mask from coefficients:", error);
    return {
      mask: [],
      polygon: [],
      imageData: null,
    };
  }
}

/**
 * Generate actual mask using model coefficients and prototypes
 */
function generateActualMask(
  coeffs: number[],
  prototypes: tf.Tensor,
  bbox: number[],
  inputWidth: number,
  inputHeight: number
): {
  mask: number[][];
  polygon: number[][];
  imageData: ImageData | null;
} {
  try {
    if (!prototypes || coeffs.length === 0) {
      throw new Error("No prototypes or coefficients available");
    }

    const [bboxX, bboxY, bboxWidth, bboxHeight] = bbox;

    // Get mask dimensions from prototypes shape
    const maskHeight = prototypes.shape[2] || 160;
    const maskWidth = prototypes.shape[3] || 160;
    const numPrototypes = prototypes.shape[1] || coeffs.length;

    console.log(
      `Generating mask: prototypes shape: ${prototypes.shape}, coeffs length: ${coeffs.length}`
    );

    // Create coefficients tensor - limit to the number of available prototypes
    const coeffsTensor = tf.tensor1d(coeffs.slice(0, numPrototypes));

    // Remove batch dimension and get the prototypes tensor in correct shape
    // Shape should be [channels, height, width] e.g., [32, 160, 160]
    const prototypesReshaped = prototypes.squeeze([0]); // Remove batch dimension

    // Perform matrix multiplication: coeffs × prototypes
    // Using einsum with correct indices: c (channels) × chw (channels, height, width) -> hw (height, width)
    const maskTensor = tf.einsum("c,chw->hw", coeffsTensor, prototypesReshaped);

    // Apply sigmoid activation to get mask probabilities
    const sigmoidMask = tf.sigmoid(maskTensor);

    // Get the mask data
    const maskData = sigmoidMask.dataSync();

    // Convert to 2D array
    const mask2D: number[][] = [];
    for (let y = 0; y < maskHeight; y++) {
      const row: number[] = [];
      for (let x = 0; x < maskWidth; x++) {
        row.push(maskData[y * maskWidth + x]);
      }
      mask2D.push(row);
    }

    // Generate polygon from the precise mask
    const polygon = extractPolygonFromMask(
      mask2D,
      bbox,
      inputWidth,
      inputHeight,
      0.5 // Higher threshold for more precise boundaries
    );

    console.log(`Generated mask with ${polygon.length} polygon points`);

    // Clean up tensors
    coeffsTensor.dispose();
    prototypesReshaped.dispose();
    maskTensor.dispose();
    sigmoidMask.dispose();

    return {
      mask: mask2D,
      polygon: polygon,
      imageData: null,
    };
  } catch (error) {
    console.error("Error generating actual mask:", error);
    // Fallback to the improved shape-based approach
    return generateMaskFromCoeffs(
      [],
      null as any,
      bbox,
      inputWidth,
      inputHeight
    );
  }
}

/**
 * Extract polygon points from mask using contour detection
 */
function extractPolygonFromMask(
  mask: number[][],
  bbox: number[],
  inputWidth: number,
  inputHeight: number,
  threshold: number = 0.5
): number[][] {
  const [bboxX, bboxY, bboxWidth, bboxHeight] = bbox;
  const maskHeight = mask.length;
  const maskWidth = mask[0]?.length || 0;

  if (maskHeight === 0 || maskWidth === 0) return [];

  // Find all edge points using Marching Squares-like algorithm
  const edgePoints: number[][] = [];

  for (let y = 0; y < maskHeight - 1; y++) {
    for (let x = 0; x < maskWidth - 1; x++) {
      // Get the 2x2 cell values
      const topLeft = mask[y][x] > threshold ? 1 : 0;
      const topRight = mask[y][x + 1] > threshold ? 1 : 0;
      const bottomLeft = mask[y + 1][x] > threshold ? 1 : 0;
      const bottomRight = mask[y + 1][x + 1] > threshold ? 1 : 0;

      // Create a configuration index (0-15)
      const config =
        (topLeft << 3) | (topRight << 2) | (bottomRight << 1) | bottomLeft;

      // Add edge points based on configuration
      const cellPoints = getMarchingSquarePoints(
        x,
        y,
        config,
        maskWidth,
        maskHeight
      );
      edgePoints.push(...cellPoints);
    }
  }

  if (edgePoints.length === 0) {
    // Fallback: find outer boundary points
    for (let y = 0; y < maskHeight; y++) {
      for (let x = 0; x < maskWidth; x++) {
        if (mask[y][x] > threshold) {
          // Check if this is a boundary pixel
          const isBoundary =
            x === 0 ||
            x === maskWidth - 1 ||
            y === 0 ||
            y === maskHeight - 1 ||
            (mask[y - 1]?.[x] || 0) <= threshold ||
            (mask[y + 1]?.[x] || 0) <= threshold ||
            (mask[y][x - 1] || 0) <= threshold ||
            (mask[y][x + 1] || 0) <= threshold;

          if (isBoundary) {
            edgePoints.push([x, y]);
          }
        }
      }
    }
  }

  // Convert mask coordinates to image coordinates
  const imagePoints = edgePoints.map(([x, y]) => {
    const imageX = bboxX + (x / maskWidth) * bboxWidth;
    const imageY = bboxY + (y / maskHeight) * bboxHeight;
    return [imageX, imageY];
  });

  // Sort and simplify the polygon
  if (imagePoints.length > 0) {
    const sortedPoints = sortPointsForPolygon(imagePoints);
    return simplifyPolygon(sortedPoints, 2.0); // Tighter tolerance for more precision
  }

  return [];
}

/**
 * Get edge points for marching squares algorithm
 */
function getMarchingSquarePoints(
  x: number,
  y: number,
  config: number,
  maskWidth: number,
  maskHeight: number
): number[][] {
  const points: number[][] = [];

  // Marching squares lookup table for edge intersections
  // Each configuration defines which edges have intersections
  const edgeTable: number[][][] = [
    [], // 0: no intersections
    [[0, 3]], // 1: bottom-left
    [[1, 2]], // 2: bottom-right
    [[0, 2]], // 3: bottom edge
    [[2, 1]], // 4: top-right
    [
      [0, 3],
      [2, 1],
    ], // 5: diagonal
    [[3, 1]], // 6: right edge
    [[0, 1]], // 7: bottom-right corner
    [[3, 0]], // 8: top-left
    [[1, 3]], // 9: left edge
    [
      [3, 0],
      [1, 2],
    ], // 10: diagonal
    [[1, 2]], // 11: top-left corner
    [[2, 0]], // 12: top edge
    [[1, 0]], // 13: top-right corner
    [[3, 2]], // 14: top-left corner
    [], // 15: no intersections
  ];

  const edges = edgeTable[config] || [];

  for (const edge of edges) {
    if (edge.length >= 2) {
      const [start, end] = edge;
      // Calculate intersection points on cell edges
      // 0=top, 1=right, 2=bottom, 3=left
      const intersections = [
        [x + 0.5, y], // top edge
        [x + 1, y + 0.5], // right edge
        [x + 0.5, y + 1], // bottom edge
        [x, y + 0.5], // left edge
      ];

      if (start < intersections.length && end < intersections.length) {
        points.push(intersections[start], intersections[end]);
      }
    }
  }

  return points;
}

/**
 * Sort points to create a proper polygon outline
 */
function sortPointsForPolygon(points: number[][]): number[][] {
  if (points.length <= 2) return points;

  // Find the centroid
  const centroidX = points.reduce((sum, p) => sum + p[0], 0) / points.length;
  const centroidY = points.reduce((sum, p) => sum + p[1], 0) / points.length;

  // Sort points by angle from centroid
  return points.sort((a, b) => {
    const angleA = Math.atan2(a[1] - centroidY, a[0] - centroidX);
    const angleB = Math.atan2(b[1] - centroidY, b[0] - centroidX);
    return angleA - angleB;
  });
}

/**
 * Simplify polygon by removing points that are too close
 */
function simplifyPolygon(points: number[][], tolerance: number): number[][] {
  if (points.length <= 3) return points;

  const simplified: number[][] = [points[0]];

  for (let i = 1; i < points.length; i++) {
    const lastPoint = simplified[simplified.length - 1];
    const currentPoint = points[i];

    const distance = Math.sqrt(
      Math.pow(currentPoint[0] - lastPoint[0], 2) +
        Math.pow(currentPoint[1] - lastPoint[1], 2)
    );

    if (distance > tolerance) {
      simplified.push(currentPoint);
    }
  }

  // Ensure the polygon is closed and has enough points for a good shape
  if (simplified.length > 2) {
    const firstPoint = simplified[0];
    const lastPoint = simplified[simplified.length - 1];
    const closingDistance = Math.sqrt(
      Math.pow(lastPoint[0] - firstPoint[0], 2) +
        Math.pow(lastPoint[1] - firstPoint[1], 2)
    );

    // Only add closing point if it's not too close to the first point
    if (closingDistance > tolerance) {
      simplified.push(firstPoint);
    }
  }

  return simplified.length >= 3 ? simplified : points;
}

/**
 * Simple Non-Maximum Suppression implementation
 */
function applyNMS(detections: any[], iouThreshold: number): any[] {
  if (detections.length === 0) return [];

  // Sort by confidence (descending)
  detections.sort((a, b) => b.confidence - a.confidence);

  const keep: any[] = [];
  const suppress = new Set<number>();

  for (let i = 0; i < detections.length; i++) {
    if (suppress.has(i)) continue;

    keep.push(detections[i]);

    for (let j = i + 1; j < detections.length; j++) {
      if (suppress.has(j)) continue;

      const iou = calculateIoU(detections[i].bbox, detections[j].bbox);
      if (iou > iouThreshold) {
        suppress.add(j);
      }
    }
  }

  return keep;
}

/**
 * Calculate Intersection over Union (IoU) for two bounding boxes
 */
function calculateIoU(box1: number[], box2: number[]): number {
  const [x1, y1, w1, h1] = box1;
  const [x2, y2, w2, h2] = box2;

  const x1_max = x1 + w1;
  const y1_max = y1 + h1;
  const x2_max = x2 + w2;
  const y2_max = y2 + h2;

  const intersectionX1 = Math.max(x1, x2);
  const intersectionY1 = Math.max(y1, y2);
  const intersectionX2 = Math.min(x1_max, x2_max);
  const intersectionY2 = Math.min(y1_max, y2_max);

  const intersectionWidth = Math.max(0, intersectionX2 - intersectionX1);
  const intersectionHeight = Math.max(0, intersectionY2 - intersectionY1);
  const intersectionArea = intersectionWidth * intersectionHeight;

  const box1Area = w1 * h1;
  const box2Area = w2 * h2;
  const unionArea = box1Area + box2Area - intersectionArea;

  return unionArea > 0 ? intersectionArea / unionArea : 0;
}

/**
 * Preprocess image for YOLO model
 */
export function preprocessImageForYolo(video: HTMLVideoElement): tf.Tensor {
  return tf.tidy(() => {
    // Convert video to tensor
    const tensor = tf.browser.fromPixels(video);

    // Resize to 640x640 (YOLO input size)
    const resized = tf.image.resizeBilinear(tensor, [640, 640]);

    // Normalize to [0, 1]
    const normalized = resized.div(255.0);

    // Add batch dimension [1, 640, 640, 3]
    const batched = normalized.expandDims(0);

    return batched;
  });
}

/**
 * Apply color filter to detected nail areas
 */
export function applyNailColorFilter(
  canvas: HTMLCanvasElement,
  detections: YoloDetection[],
  color: { r: number; g: number; b: number; a: number }
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  // Save the current state
  ctx.save();

  detections.forEach((detection) => {
    // Always prefer polygon rendering when available for maximum precision
    if (detection.maskPolygon && detection.maskPolygon.length > 3) {
      // Use precise polygon outline with smooth curves
      applyPrecisePolygonColor(ctx, detection, color);
    } else if (detection.mask && detection.mask.length > 0) {
      // Use the mask data if polygon is not available
      applyMaskBasedColor(ctx, detection, color);
    } else {
      // Fallback to improved nail shape (should rarely be used now)
      applyNailShapeColor(ctx, detection, color);
    }
  });

  // Restore the previous state
  ctx.restore();
}

/**
 * Apply color using precise polygon with smooth rendering
 */
function applyPrecisePolygonColor(
  ctx: CanvasRenderingContext2D,
  detection: YoloDetection,
  color: { r: number; g: number; b: number; a: number }
): void {
  const polygon = detection.maskPolygon!;
  const [bboxX, bboxY, bboxWidth, bboxHeight] = detection.bbox;

  // Enable anti-aliasing for smoother edges
  ctx.imageSmoothingEnabled = true;
  ctx.globalCompositeOperation = "source-over";

  // Create the main nail color
  ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${color.a * 0.8})`;

  ctx.beginPath();
  if (polygon.length > 0) {
    ctx.moveTo(polygon[0][0], polygon[0][1]);

    // Use quadratic curves for smoother polygon edges
    for (let i = 1; i < polygon.length; i++) {
      const current = polygon[i];
      const next = polygon[(i + 1) % polygon.length];

      // Calculate control point for smooth curve
      const controlX = (current[0] + next[0]) / 2;
      const controlY = (current[1] + next[1]) / 2;

      if (i === polygon.length - 1) {
        // Close the path smoothly
        ctx.quadraticCurveTo(
          current[0],
          current[1],
          polygon[0][0],
          polygon[0][1]
        );
      } else {
        ctx.quadraticCurveTo(current[0], current[1], controlX, controlY);
      }
    }
  }

  ctx.closePath();
  ctx.fill();

  // Add subtle gradient highlight for realism
  const gradient = ctx.createRadialGradient(
    bboxX + bboxWidth * 0.3, // Offset highlight position
    bboxY + bboxHeight * 0.2,
    0,
    bboxX + bboxWidth / 2,
    bboxY + bboxHeight / 2,
    Math.max(bboxWidth, bboxHeight) * 0.7
  );

  gradient.addColorStop(0, `rgba(255, 255, 255, ${color.a * 0.25})`);
  gradient.addColorStop(0.4, `rgba(255, 255, 255, ${color.a * 0.1})`);
  gradient.addColorStop(1, `rgba(0, 0, 0, ${color.a * 0.05})`);

  ctx.fillStyle = gradient;
  ctx.fill();

  // Add a subtle shadow/depth effect
  ctx.strokeStyle = `rgba(0, 0, 0, ${color.a * 0.15})`;
  ctx.lineWidth = 0.5;
  ctx.stroke();
}

/**
 * Apply color using mask data for precise nail shape
 */
function applyMaskBasedColor(
  ctx: CanvasRenderingContext2D,
  detection: YoloDetection,
  color: { r: number; g: number; b: number; a: number }
): void {
  const [bboxX, bboxY, bboxWidth, bboxHeight] = detection.bbox;
  const mask = detection.mask!;
  const maskHeight = mask.length;
  const maskWidth = mask[0]?.length || 0;

  if (maskHeight === 0 || maskWidth === 0) return;

  // Use polygon approach to avoid gray background from ImageData
  if (detection.maskPolygon && detection.maskPolygon.length > 2) {
    ctx.globalCompositeOperation = "source-over";
    ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${
      color.a * 0.7
    })`;

    ctx.beginPath();
    ctx.moveTo(detection.maskPolygon[0][0], detection.maskPolygon[0][1]);

    for (let i = 1; i < detection.maskPolygon.length; i++) {
      ctx.lineTo(detection.maskPolygon[i][0], detection.maskPolygon[i][1]);
    }

    ctx.closePath();
    ctx.fill();

    // Add subtle highlight
    const gradient = ctx.createLinearGradient(
      bboxX,
      bboxY,
      bboxX,
      bboxY + bboxHeight
    );
    gradient.addColorStop(0, `rgba(255, 255, 255, ${color.a * 0.15})`);
    gradient.addColorStop(0.5, `rgba(255, 255, 255, ${color.a * 0.05})`);
    gradient.addColorStop(1, `rgba(0, 0, 0, ${color.a * 0.02})`);

    ctx.fillStyle = gradient;
    ctx.fill();
  } else {
    // Fallback to nail shape if no polygon available
    applyNailShapeColor(ctx, detection, color);
  }
}

/**
 * Apply color using nail-shaped overlay
 */
function applyNailShapeColor(
  ctx: CanvasRenderingContext2D,
  detection: YoloDetection,
  color: { r: number; g: number; b: number; a: number }
): void {
  const [x, y, width, height] = detection.bbox;

  // Create a more realistic nail shape
  const cornerRadius = Math.min(width, height) * 0.25;

  // Use source-over for natural blending
  ctx.globalCompositeOperation = "source-over";
  ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${color.a * 0.7})`;

  ctx.beginPath();

  // Create a rounded rectangle that looks more like a nail
  ctx.roundRect(x, y, width, height, [
    cornerRadius,
    cornerRadius, // top corners
    cornerRadius * 0.5,
    cornerRadius * 0.5, // bottom corners (less rounded)
  ]);

  ctx.fill();

  // Add a subtle highlight for more realistic appearance
  const gradient = ctx.createLinearGradient(x, y, x, y + height);
  gradient.addColorStop(0, `rgba(255, 255, 255, ${color.a * 0.2})`);
  gradient.addColorStop(0.3, `rgba(255, 255, 255, ${color.a * 0.1})`);
  gradient.addColorStop(1, `rgba(0, 0, 0, ${color.a * 0.05})`);

  ctx.fillStyle = gradient;
  ctx.fill();
}
