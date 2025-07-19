"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import {
  processYoloOutput,
  preprocessImageForYolo,
  YoloDetection,
} from "../utils/yolo";

interface WebcamCaptureProps {
  onModelLoaded: (loaded: boolean) => void;
}

const WebcamCapture: React.FC<WebcamCaptureProps> = ({ onModelLoaded }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modelRef = useRef<tf.GraphModel | null>(null);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [detections, setDetections] = useState<YoloDetection[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  const animationRef = useRef<number | undefined>(undefined);
  const lastTimeRef = useRef<number>(0);
  const frameCountRef = useRef<number>(0);
  const resizeObserverRef = useRef<ResizeObserver | null>(null);
  const pendingInferenceRef = useRef<boolean>(false);
  const lastInferenceTimeRef = useRef<number>(0);
  const videoFrameRef = useRef<ImageData | null>(null);
  const syncedDetectionsRef = useRef<YoloDetection[]>([]);

  // Load the model
  const loadModel = useCallback(async () => {
    try {
      console.log("Loading nail segmentation model...");

      // Initialize TensorFlow.js backend
      await tf.ready();
      console.log("TensorFlow.js backend initialized");
      console.log("Available backends:", tf.getBackend());

      // Try to load the model from the model_web folder
      const modelUrl = "/model_web/model.json";
      const model = await tf.loadGraphModel(modelUrl);

      modelRef.current = model;
      onModelLoaded(true);

      console.log("Model loaded successfully");
      console.log(
        "Model inputs:",
        model.inputs.map((input) => ({
          name: input.name,
          shape: input.shape,
          dtype: input.dtype,
        }))
      );
      console.log(
        "Model outputs:",
        model.outputs.map((output) => ({
          name: output.name,
          shape: output.shape,
          dtype: output.dtype,
        }))
      );

      // Warm up the model with a dummy input
      console.log("Warming up model...");
      const dummyInput = tf.zeros([1, 640, 640, 3]);
      const warmupOutputs = (await model.executeAsync(
        dummyInput
      )) as tf.Tensor[];
      warmupOutputs.forEach((tensor) => tensor.dispose());
      dummyInput.dispose();
      console.log("Model warmup completed");
    } catch (error) {
      console.error("Error loading model:", error);
      setError(
        `Failed to load model: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
      onModelLoaded(false);
    }
  }, [onModelLoaded]);

  // Initialize webcam
  const startWebcam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 640 },
          facingMode: "user",
        },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsWebcamActive(true);
        setError(null);
        lastTimeRef.current = performance.now();
        frameCountRef.current = 0;
      }
    } catch (error) {
      console.error("Error accessing webcam:", error);
      setError("Failed to access webcam. Please check permissions.");
    }
  }, []);

  // Stop webcam
  const stopWebcam = useCallback(() => {
    if (videoRef.current?.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsWebcamActive(false);
    setDetections([]); // Clear detections when stopping
    syncedDetectionsRef.current = []; // Clear synced detections too
    pendingInferenceRef.current = false; // Reset pending state
    videoFrameRef.current = null; // Clear frame reference
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
  }, []);

  // Run inference
  const runInference = useCallback(async () => {
    if (
      !modelRef.current ||
      !videoRef.current ||
      isProcessing ||
      pendingInferenceRef.current
    )
      return;

    // Check if video dimensions are valid
    const videoWidth = videoRef.current.videoWidth;
    const videoHeight = videoRef.current.videoHeight;

    if (videoWidth === 0 || videoHeight === 0) {
      console.log("Video dimensions not ready:", videoWidth, "x", videoHeight);
      return;
    }

    // Throttle inference to avoid overwhelming the system
    const currentTime = performance.now();
    if (currentTime - lastInferenceTimeRef.current < 500) {
      // Reduced to 2 FPS max for inference
      return;
    }

    pendingInferenceRef.current = true;
    setIsProcessing(true);

    try {
      // Capture the current video frame for this inference
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      canvas.width = videoWidth;
      canvas.height = videoHeight;
      ctx.drawImage(videoRef.current, 0, 0, videoWidth, videoHeight);

      // Store the frame data for synchronized drawing
      const currentFrameData = ctx.getImageData(0, 0, videoWidth, videoHeight);
      videoFrameRef.current = currentFrameData;

      const preprocessed = preprocessImageForYolo(videoRef.current);

      // Run model inference
      const outputs = (await modelRef.current.executeAsync(
        preprocessed
      )) as tf.Tensor[];

      console.log(
        "Model outputs:",
        outputs.map((o) => ({ shape: o.shape, dtype: o.dtype }))
      );

      // Process YOLO output
      const result = processYoloOutput(
        outputs,
        videoWidth,
        videoHeight,
        0.5, // increased confidence threshold
        0.45 // NMS threshold
      );

      // Only update detections if this inference is still relevant
      if (pendingInferenceRef.current) {
        console.log(
          `Setting ${result.detections.length} detections from inference`
        );
        // Update both the state and the synced ref simultaneously
        setDetections(result.detections);
        syncedDetectionsRef.current = result.detections;
        lastInferenceTimeRef.current = currentTime;
      }

      // Clean up tensors
      preprocessed.dispose();
      outputs.forEach((tensor) => tensor.dispose());

      // Calculate FPS
      frameCountRef.current++;
      const fpsCurrentTime = performance.now();
      if (fpsCurrentTime - lastTimeRef.current >= 1000) {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
        lastTimeRef.current = fpsCurrentTime;
      }
    } catch (error) {
      console.error("Inference error:", error);
      setError(
        `Inference failed: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    } finally {
      setIsProcessing(false);
      pendingInferenceRef.current = false;
    }
  }, [isProcessing]);

  // Draw detections on canvas
  const drawDetections = useCallback(() => {
    if (!canvasRef.current || !videoRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const video = videoRef.current;
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;

    if (videoWidth === 0 || videoHeight === 0) return;

    // Get the displayed video element dimensions
    const displayWidth = video.clientWidth;
    const displayHeight = video.clientHeight;

    // Set canvas size to match the displayed video size exactly
    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
    }

    // Calculate how the video is actually displayed within the container
    // The video has object-fit: cover, so it might be cropped
    const videoAspectRatio = videoWidth / videoHeight;
    const displayAspectRatio = displayWidth / displayHeight;

    let actualVideoWidth, actualVideoHeight, offsetX, offsetY;

    if (videoAspectRatio > displayAspectRatio) {
      // Video is wider - it will be cropped horizontally
      actualVideoHeight = displayHeight;
      actualVideoWidth = displayHeight * videoAspectRatio;
      offsetX = (displayWidth - actualVideoWidth) / 2;
      offsetY = 0;
    } else {
      // Video is taller - it will be cropped vertically
      actualVideoWidth = displayWidth;
      actualVideoHeight = displayWidth / videoAspectRatio;
      offsetX = 0;
      offsetY = (displayHeight - actualVideoHeight) / 2;
    }

    // Calculate scaling factors based on actual displayed video area
    const scaleX = actualVideoWidth / videoWidth;
    const scaleY = actualVideoHeight / videoHeight;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Use synced detections to ensure frame/detection alignment
    const currentDetections = syncedDetectionsRef.current;

    // Debug: Always log detection count
    console.log(`Drawing ${currentDetections.length} synced detections`);

    // Debug: Log canvas and video info when detections are found
    if (currentDetections.length > 0) {
      console.log("Drawing synced detections:", {
        count: currentDetections.length,
        videoSize: { width: videoWidth, height: videoHeight },
        displaySize: { width: displayWidth, height: displayHeight },
        actualVideoSize: { width: actualVideoWidth, height: actualVideoHeight },
        offset: { x: offsetX, y: offsetY },
        canvasSize: { width: canvas.width, height: canvas.height },
        scales: { x: scaleX, y: scaleY },
      });
    }

    // Draw detections
    currentDetections.forEach((detection, index) => {
      const [x, y, width, height] = detection.bbox;

      // Scale coordinates to match displayed video size and add offset
      const scaledX = x * scaleX + offsetX;
      const scaledY = y * scaleY + offsetY;
      const scaledWidth = width * scaleX;
      const scaledHeight = height * scaleY;

      console.log(`Detection ${index}:`, {
        original: [x, y, width, height],
        scaled: [scaledX, scaledY, scaledWidth, scaledHeight],
        videoAspect: videoAspectRatio,
        displayAspect: displayAspectRatio,
        actualVideo: { width: actualVideoWidth, height: actualVideoHeight },
        offset: { x: offsetX, y: offsetY },
      });

      // Only draw if coordinates are reasonable
      if (scaledWidth <= 0 || scaledHeight <= 0) {
        console.warn(`Skipping detection ${index} - invalid dimensions`);
        return;
      }

      // Clamp coordinates to canvas bounds
      const clampedX = Math.max(0, Math.min(scaledX, canvas.width));
      const clampedY = Math.max(0, Math.min(scaledY, canvas.height));
      const clampedWidth = Math.min(scaledWidth, canvas.width - clampedX);
      const clampedHeight = Math.min(scaledHeight, canvas.height - clampedY);

      if (clampedWidth <= 0 || clampedHeight <= 0) {
        console.warn(
          `Skipping detection ${index} - out of bounds after clamping`
        );
        return;
      }

      // Draw bounding box with nail-themed styling
      ctx.strokeStyle = "#ff6b9d"; // Pink color for nails
      ctx.lineWidth = 3;
      ctx.setLineDash([]);
      ctx.strokeRect(clampedX, clampedY, clampedWidth, clampedHeight);

      // Draw filled background for label
      const label = `Nail ${(detection.score * 100).toFixed(1)}%`;
      ctx.font = "bold 14px Arial";
      const textMetrics = ctx.measureText(label);
      const textWidth = textMetrics.width;
      const textHeight = 20;

      // Ensure label stays within canvas bounds
      const labelX = Math.max(
        0,
        Math.min(clampedX, canvas.width - textWidth - 10)
      );
      const labelY = Math.max(textHeight + 2, clampedY);

      ctx.fillStyle = "#ff6b9d";
      ctx.fillRect(
        labelX,
        labelY - textHeight - 2,
        textWidth + 10,
        textHeight + 4
      );

      // Draw label text
      ctx.fillStyle = "white";
      ctx.fillText(label, labelX + 5, labelY - 5);

      // Draw corner indicators
      const cornerSize = Math.min(15, clampedWidth / 4, clampedHeight / 4);
      ctx.strokeStyle = "#ff6b9d";
      ctx.lineWidth = 2;

      // Top-left corner
      ctx.beginPath();
      ctx.moveTo(clampedX, clampedY + cornerSize);
      ctx.lineTo(clampedX, clampedY);
      ctx.lineTo(clampedX + cornerSize, clampedY);
      ctx.stroke();

      // Top-right corner
      ctx.beginPath();
      ctx.moveTo(clampedX + clampedWidth - cornerSize, clampedY);
      ctx.lineTo(clampedX + clampedWidth, clampedY);
      ctx.lineTo(clampedX + clampedWidth, clampedY + cornerSize);
      ctx.stroke();

      // Bottom-left corner
      ctx.beginPath();
      ctx.moveTo(clampedX, clampedY + clampedHeight - cornerSize);
      ctx.lineTo(clampedX, clampedY + clampedHeight);
      ctx.lineTo(clampedX + cornerSize, clampedY + clampedHeight);
      ctx.stroke();

      // Bottom-right corner
      ctx.beginPath();
      ctx.moveTo(
        clampedX + clampedWidth - cornerSize,
        clampedY + clampedHeight
      );
      ctx.lineTo(clampedX + clampedWidth, clampedY + clampedHeight);
      ctx.lineTo(
        clampedX + clampedWidth,
        clampedY + clampedHeight - cornerSize
      );
      ctx.stroke();
    });
  }, []);

  // Main processing loop
  const processFrame = useCallback(() => {
    if (isWebcamActive && videoRef.current && modelRef.current) {
      // Only draw when we have an inference to show
      if (
        syncedDetectionsRef.current.length > 0 ||
        !pendingInferenceRef.current
      ) {
        drawDetections();
      }

      // Run inference at a controlled rate, synchronized with drawing
      if (!pendingInferenceRef.current && !isProcessing) {
        runInference();
      }

      animationRef.current = requestAnimationFrame(processFrame);
    }
  }, [isWebcamActive, runInference, drawDetections, isProcessing]);

  // Effects
  useEffect(() => {
    loadModel();
  }, [loadModel]);

  // Set up resize observer to handle canvas resizing
  useEffect(() => {
    if (videoRef.current) {
      resizeObserverRef.current = new ResizeObserver(() => {
        // Trigger a redraw when video element is resized
        if (syncedDetectionsRef.current.length > 0) {
          drawDetections();
        }
      });
      resizeObserverRef.current.observe(videoRef.current);
    }

    return () => {
      if (resizeObserverRef.current) {
        resizeObserverRef.current.disconnect();
      }
    };
  }, [drawDetections]);

  useEffect(() => {
    if (isWebcamActive && modelRef.current) {
      processFrame();
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isWebcamActive, processFrame]);

  return (
    <div className="w-full max-w-4xl">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex flex-col items-center">
          {/* Single Video Feed with Overlaid Detections */}
          <div className="w-full max-w-2xl">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 justify-center">
              � Nail Detection Camera
              {fps > 0 && (
                <span className="text-sm bg-blue-100 text-blue-800 px-2 py-1 rounded">
                  {fps} FPS
                </span>
              )}
              {isProcessing && (
                <div className="animate-spin w-4 h-4 border-2 border-pink-500 border-t-transparent rounded-full"></div>
              )}
            </h3>
            <div className="relative bg-gray-900 rounded-lg overflow-hidden aspect-square">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
                onLoadedMetadata={() => {
                  console.log("Video metadata loaded:", {
                    width: videoRef.current?.videoWidth,
                    height: videoRef.current?.videoHeight,
                  });
                }}
                onCanPlay={() => {
                  console.log("Video can play:", {
                    width: videoRef.current?.videoWidth,
                    height: videoRef.current?.videoHeight,
                  });
                  // Wait a moment for video to be fully ready
                  setTimeout(() => {
                    if (
                      modelRef.current &&
                      videoRef.current &&
                      videoRef.current.videoWidth > 0
                    ) {
                      processFrame();
                    }
                  }, 100);
                }}
              />
              {/* Canvas overlay for detections */}
              <canvas
                ref={canvasRef}
                className="absolute top-0 left-0 w-full h-full pointer-events-none"
                style={{
                  zIndex: 10,
                }}
              />
              {!isWebcamActive && (
                <div className="absolute inset-0 flex items-center justify-center text-white">
                  <div className="text-center">
                    <div className="text-4xl mb-2">�</div>
                    <div>Camera not active</div>
                  </div>
                </div>
              )}
              {detections.length === 0 && isWebcamActive && !isProcessing && (
                <div className="absolute top-4 left-4 bg-black bg-opacity-50 text-white px-3 py-2 rounded-lg">
                  <div className="text-sm">
                    🔍 Place your hand in front of the camera
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="mt-6 flex flex-wrap gap-4 justify-center">
          <button
            onClick={isWebcamActive ? stopWebcam : startWebcam}
            disabled={!modelRef.current}
            className={`px-6 py-2 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
              isWebcamActive
                ? "bg-red-500 hover:bg-red-600 text-white"
                : "bg-green-500 hover:bg-green-600 text-white"
            }`}
          >
            {isWebcamActive ? "🛑 Stop Camera" : "📷 Start Camera"}
          </button>

          {isProcessing && (
            <div className="flex items-center gap-2 text-pink-600">
              <div className="animate-spin w-4 h-4 border-2 border-pink-600 border-t-transparent rounded-full"></div>
              <span>Analyzing...</span>
            </div>
          )}
        </div>

        {/* Stats */}
        {isWebcamActive && (
          <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div className="bg-pink-50 rounded-lg p-3">
              <div className="text-2xl font-bold text-pink-600">
                {detections.length}
              </div>
              <div className="text-sm text-gray-600">Nails Detected</div>
            </div>
            <div className="bg-blue-50 rounded-lg p-3">
              <div className="text-2xl font-bold text-blue-600">
                {detections.length > 0
                  ? (
                      (detections.reduce((acc, d) => acc + d.score, 0) /
                        detections.length) *
                      100
                    ).toFixed(1)
                  : 0}
                %
              </div>
              <div className="text-sm text-gray-600">Avg Confidence</div>
            </div>
            <div className="bg-purple-50 rounded-lg p-3">
              <div className="text-2xl font-bold text-purple-600">{fps}</div>
              <div className="text-sm text-gray-600">FPS</div>
            </div>
            <div className="bg-orange-50 rounded-lg p-3">
              <div className="text-2xl font-bold text-orange-600">YOLOv8</div>
              <div className="text-sm text-gray-600">Model</div>
            </div>
          </div>
        )}

        {/* Detection Details */}
        {detections.length > 0 && (
          <div className="mt-4">
            <h4 className="text-md font-semibold mb-2">Detection Details</h4>
            <div className="space-y-2 max-h-32 overflow-y-auto">
              {detections.map((detection, index) => (
                <div
                  key={index}
                  className="text-sm bg-gray-50 rounded p-2 flex justify-between"
                >
                  <span>Nail #{index + 1}</span>
                  <span className="font-medium text-pink-600">
                    {(detection.score * 100).toFixed(1)}% confidence
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-center gap-2 text-red-700">
              <span>⚠️</span>
              <span className="font-medium">Error:</span>
            </div>
            <div className="text-red-600 mt-1">{error}</div>
            <button
              onClick={() => {
                setError(null);
                if (!modelRef.current) {
                  loadModel();
                }
              }}
              className="mt-2 px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700"
            >
              Retry
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default WebcamCapture;
