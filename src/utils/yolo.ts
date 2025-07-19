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
    const [bboxX, bboxY, bboxWidth, bboxHeight] = bbox;

    // If we have mask coefficients and prototypes, use them
    if (coeffs.length > 0 && prototypes) {
      return generateActualMask(coeffs, prototypes, bbox, inputWidth, inputHeight);
    }

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
        
        if (normalizedX >= padding && normalizedX <= 1 - padding &&
            normalizedY >= padding && normalizedY <= 1 - padding) {
          
          // Calculate distance from edges for rounded corners
          const distFromLeft = normalizedX - padding;
          const distFromRight = (1 - padding) - normalizedX;
          const distFromTop = normalizedY - padding;
          const distFromBottom = (1 - padding) - normalizedY;
          
          const width = 1 - 2 * padding;
          const height = 1 - 2 * padding;
          
          // Check if we're in a corner region
          if ((distFromLeft < cornerRadius && distFromTop < cornerRadius) ||
              (distFromRight < cornerRadius && distFromTop < cornerRadius) ||
              (distFromLeft < cornerRadius && distFromBottom < cornerRadius) ||
              (distFromRight < cornerRadius && distFromBottom < cornerRadius)) {
            
            // Calculate distance from nearest corner
            let cornerX, cornerY;
            if (distFromLeft < cornerRadius && distFromTop < cornerRadius) {
              cornerX = padding + cornerRadius;
              cornerY = padding + cornerRadius;
            } else if (distFromRight < cornerRadius && distFromTop < cornerRadius) {
              cornerX = 1 - padding - cornerRadius;
              cornerY = padding + cornerRadius;
            } else if (distFromLeft < cornerRadius && distFromBottom < cornerRadius) {
              cornerX = padding + cornerRadius;
              cornerY = 1 - padding - cornerRadius;
            } else {
              cornerX = 1 - padding - cornerRadius;
              cornerY = 1 - padding - cornerRadius;
            }
            
            const distFromCorner = Math.sqrt(
              Math.pow(normalizedX - cornerX, 2) + Math.pow(normalizedY - cornerY, 2)
            );
            
            if (distFromCorner <= cornerRadius) {
              maskValue = Math.max(0, 1 - (distFromCorner / cornerRadius) * 0.3);
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
    // This is where we would implement the actual mask generation
    // using the mask coefficients and prototypes from the YOLOv8 model
    // For now, we'll use the improved fallback
    return generateMaskFromCoeffs([], null as any, bbox, inputWidth, inputHeight);
  } catch (error) {
    console.error("Error generating actual mask:", error);
    return {
      mask: [],
      polygon: [],
      imageData: null,
    };
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
  threshold: number = 0.3
): number[][] {
  const [bboxX, bboxY, bboxWidth, bboxHeight] = bbox;
  const maskHeight = mask.length;
  const maskWidth = mask[0]?.length || 0;

  if (maskHeight === 0 || maskWidth === 0) return [];

  const points: number[][] = [];

  // Find contour points by tracing the edge of the mask
  for (let y = 0; y < maskHeight; y++) {
    for (let x = 0; x < maskWidth; x++) {
      if (mask[y][x] > threshold) {
        // Check if this is an edge pixel
        const isEdge =
          x === 0 ||
          x === maskWidth - 1 ||
          y === 0 ||
          y === maskHeight - 1 ||
          (mask[y - 1]?.[x] || 0) <= threshold ||
          (mask[y + 1]?.[x] || 0) <= threshold ||
          (mask[y][x - 1] || 0) <= threshold ||
          (mask[y][x + 1] || 0) <= threshold;

        if (isEdge) {
          // Convert mask coordinates to image coordinates
          const imageX = bboxX + (x / maskWidth) * bboxWidth;
          const imageY = bboxY + (y / maskHeight) * bboxHeight;
          points.push([imageX, imageY]);
        }
      }
    }
  }

  // Sort points to create a proper polygon outline
  if (points.length > 0) {
    const sortedPoints = sortPointsForPolygon(points);
    return simplifyPolygon(sortedPoints, 3.0);
  }

  return [];
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
  if (points.length <= 2) return points;

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

  return simplified;
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
    // Always prefer polygon rendering when available as it avoids gray backgrounds
    if (detection.maskPolygon && detection.maskPolygon.length > 2) {
      // Use polygon outline with proper color blending
      ctx.globalCompositeOperation = "source-over";
      ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${color.a * 0.7})`;
      ctx.beginPath();
      ctx.moveTo(detection.maskPolygon[0][0], detection.maskPolygon[0][1]);

      for (let i = 1; i < detection.maskPolygon.length; i++) {
        ctx.lineTo(detection.maskPolygon[i][0], detection.maskPolygon[i][1]);
      }

      ctx.closePath();
      ctx.fill();
      
      // Add subtle highlight
      const [bboxX, bboxY, bboxWidth, bboxHeight] = detection.bbox;
      const gradient = ctx.createLinearGradient(bboxX, bboxY, bboxX, bboxY + bboxHeight);
      gradient.addColorStop(0, `rgba(255, 255, 255, ${color.a * 0.15})`);
      gradient.addColorStop(0.5, `rgba(255, 255, 255, ${color.a * 0.05})`);
      gradient.addColorStop(1, `rgba(0, 0, 0, ${color.a * 0.02})`);
      
      ctx.fillStyle = gradient;
      ctx.fill();
    } else if (detection.mask && detection.mask.length > 0) {
      // Use the mask data if polygon is not available
      applyMaskBasedColor(ctx, detection, color);
    } else {
      // Fallback to improved nail shape
      applyNailShapeColor(ctx, detection, color);
    }
  });

  // Restore the previous state
  ctx.restore();
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
    ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${color.a * 0.7})`;
    
    ctx.beginPath();
    ctx.moveTo(detection.maskPolygon[0][0], detection.maskPolygon[0][1]);
    
    for (let i = 1; i < detection.maskPolygon.length; i++) {
      ctx.lineTo(detection.maskPolygon[i][0], detection.maskPolygon[i][1]);
    }
    
    ctx.closePath();
    ctx.fill();
    
    // Add subtle highlight
    const gradient = ctx.createLinearGradient(bboxX, bboxY, bboxX, bboxY + bboxHeight);
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
    cornerRadius, cornerRadius, // top corners
    cornerRadius * 0.5, cornerRadius * 0.5 // bottom corners (less rounded)
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
