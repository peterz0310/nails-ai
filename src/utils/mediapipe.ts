// MediaPipe types and utilities
export interface HandDetection {
  landmarks: Array<{ x: number; y: number; z: number }>;
  handedness: string; // "Left" or "Right"
  score: number;
}

export interface MediaPipeHandsResult {
  hands: HandDetection[];
}

// Simple type definitions to avoid import conflicts
interface MediaPipeResults {
  multiHandLandmarks?: Array<Array<{ x: number; y: number; z: number }>>;
  multiHandedness?: Array<{ label: string; score: number }>;
  image?: any;
}

interface MediaPipeHands {
  setOptions(options: any): void;
  onResults(callback: (results: MediaPipeResults) => void): void;
  send(inputs: { image: HTMLVideoElement }): Promise<void>;
  close(): void;
}

// Hand connection pairs for drawing skeleton
const HAND_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4], // Thumb
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8], // Index finger
  [0, 9],
  [9, 10],
  [10, 11],
  [11, 12], // Middle finger
  [0, 13],
  [13, 14],
  [14, 15],
  [15, 16], // Ring finger
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20], // Pinky
  [5, 9],
  [9, 13],
  [13, 17], // Palm connections
];

let handsModel: MediaPipeHands | null = null;

export const initializeMediaPipeHands = async (): Promise<MediaPipeHands> => {
  if (handsModel) {
    return handsModel;
  }

  try {
    // Dynamic import to avoid SSR issues
    const { Hands } = await import("@mediapipe/hands");

    const hands = new Hands({
      locateFile: (file: string) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
      },
    });

    hands.setOptions({
      maxNumHands: 2,
      modelComplexity: 0, // Reduced from 1 to 0 for better performance
      minDetectionConfidence: 0.7, // Increased to reduce false positives and processing
      minTrackingConfidence: 0.7, // Increased for better stability and less processing
    });

    handsModel = hands as MediaPipeHands;
    console.log("MediaPipe Hands model initialized successfully");
    return handsModel;
  } catch (error) {
    console.error("Failed to initialize MediaPipe Hands:", error);
    throw error;
  }
};

export const processMediaPipeResults = (
  results: MediaPipeResults
): MediaPipeHandsResult => {
  const hands: HandDetection[] = [];

  if (results.multiHandLandmarks && results.multiHandedness) {
    for (let i = 0; i < results.multiHandLandmarks.length; i++) {
      const landmarks = results.multiHandLandmarks[i];
      const handedness = results.multiHandedness[i];

      hands.push({
        landmarks: landmarks.map(
          (landmark: { x: number; y: number; z: number }) => ({
            x: landmark.x,
            y: landmark.y,
            z: landmark.z,
          })
        ),
        handedness: handedness.label,
        score: handedness.score,
      });
    }
  }

  return { hands };
};

export const drawHandDetections = (
  canvas: HTMLCanvasElement,
  hands: HandDetection[],
  options: {
    drawLandmarks: boolean;
    drawConnections: boolean;
    landmarkColor: string;
    connectionColor: string;
    landmarkSize: number;
    connectionWidth: number;
  } = {
    drawLandmarks: true,
    drawConnections: true,
    landmarkColor: "#00ff00",
    connectionColor: "#0000ff",
    landmarkSize: 4,
    connectionWidth: 2,
  }
): void => {
  const ctx = canvas.getContext("2d");
  if (!ctx || !hands || hands.length === 0) {
    console.log("No hands to draw or invalid canvas context");
    return;
  }

  console.log(
    `Drawing ${hands.length} hands on canvas ${canvas.width}x${canvas.height}`
  );

  hands.forEach((hand, handIndex) => {
    if (!hand.landmarks || hand.landmarks.length === 0) {
      console.warn(`Hand ${handIndex} has no landmarks`);
      return;
    }

    // Convert normalized coordinates to canvas coordinates
    const canvasLandmarks = hand.landmarks.map((landmark) => ({
      x: landmark.x * canvas.width,
      y: landmark.y * canvas.height,
    }));

    // Draw connections
    if (options.drawConnections) {
      ctx.strokeStyle = options.connectionColor;
      ctx.lineWidth = options.connectionWidth;

      HAND_CONNECTIONS.forEach(([start, end]) => {
        if (start < canvasLandmarks.length && end < canvasLandmarks.length) {
          const startPoint = canvasLandmarks[start];
          const endPoint = canvasLandmarks[end];

          ctx.beginPath();
          ctx.moveTo(startPoint.x, startPoint.y);
          ctx.lineTo(endPoint.x, endPoint.y);
          ctx.stroke();
        }
      });
    }

    // Draw landmarks
    if (options.drawLandmarks) {
      ctx.fillStyle = options.landmarkColor;

      canvasLandmarks.forEach((landmark) => {
        ctx.beginPath();
        ctx.arc(landmark.x, landmark.y, options.landmarkSize, 0, 2 * Math.PI);
        ctx.fill();
      });
    }

    // Draw hand label
    if (canvasLandmarks.length > 0) {
      const wristPoint = canvasLandmarks[0]; // Wrist is landmark 0
      ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
      ctx.fillRect(wristPoint.x - 20, wristPoint.y - 30, 80, 20);

      ctx.fillStyle = "white";
      ctx.font = "12px Arial";
      ctx.fillText(
        `${hand.handedness} ${(hand.score * 100).toFixed(0)}%`,
        wristPoint.x - 15,
        wristPoint.y - 15
      );
    }

    console.log(
      `Drew hand ${handIndex}: ${hand.handedness} with ${canvasLandmarks.length} landmarks`
    );
  });
};

export const getHandBoundingBox = (
  hand: HandDetection,
  imageWidth: number,
  imageHeight: number
): [number, number, number, number] => {
  if (hand.landmarks.length === 0) {
    return [0, 0, 0, 0];
  }

  let minX = hand.landmarks[0].x;
  let maxX = hand.landmarks[0].x;
  let minY = hand.landmarks[0].y;
  let maxY = hand.landmarks[0].y;

  hand.landmarks.forEach((landmark) => {
    minX = Math.min(minX, landmark.x);
    maxX = Math.max(maxX, landmark.x);
    minY = Math.min(minY, landmark.y);
    maxY = Math.max(maxY, landmark.y);
  });

  // Convert from normalized coordinates to pixel coordinates
  const x = minX * imageWidth;
  const y = minY * imageHeight;
  const width = (maxX - minX) * imageWidth;
  const height = (maxY - minY) * imageHeight;

  return [x, y, width, height];
};

export const disposeMediaPipeHands = (): void => {
  if (handsModel) {
    handsModel.close();
    handsModel = null;
  }
};
