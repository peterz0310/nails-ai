import * as tf from "@tensorflow/tfjs";

export interface YoloDetection {
  bbox: number[]; // [x, y, width, height]
  score: number;
  class: number;
  mask?: number[][];
}

export interface YoloOutput {
  detections: YoloDetection[];
  maskWidth: number;
  maskHeight: number;
}

/**
 * Process YOLOv8 segmentation model output
 * Expected output format from YOLOv8-seg:
 * - Output: [1, features, 8400] where features include [x,y,w,h,conf,mask_coeffs...]
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
    console.log("Predictions shape:", predictions.shape);
    console.log("Predictions dtype:", predictions.dtype);

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
    console.log(
      `Processing: features=${numFeatures}, detections=${numDetections}`
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

        validDetections.push({
          bbox: [x, y, pixelWidth, pixelHeight],
          score: confidence,
          class: 0, // Nail class
          confidence: confidence,
          index: i,
        });
      }
    }

    console.log(
      `Found ${validDetections.length} valid detections above threshold ${confidenceThreshold}`
    );

    // Apply Non-Maximum Suppression
    const nmsDetections = applyNMS(validDetections, nmsThreshold);
    console.log(`After NMS: ${nmsDetections.length} detections`);

    // Clean up transposed tensor if we created one
    if (processedPredictions !== predictions) {
      processedPredictions.dispose();
    }

    return {
      detections: nmsDetections.map((d) => ({
        bbox: d.bbox,
        score: d.score,
        class: d.class,
      })),
      maskWidth: 160,
      maskHeight: 160,
    };
  } catch (error) {
    console.error("Error processing YOLO output:", error);
    return { detections, maskWidth: 0, maskHeight: 0 };
  }
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
