"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import {
  processYoloOutput,
  preprocessImageForYolo,
  YoloDetection,
  applyNailColorFilter,
} from "../utils/yolo";
import {
  initializeMediaPipeHands,
  processMediaPipeResults,
  drawHandDetections,
  HandDetection,
  MediaPipeHandsResult,
  disposeMediaPipeHands,
} from "../utils/mediapipe";

interface WebcamCaptureProps {
  onModelLoaded: (loaded: boolean) => void;
}

type DetectionMode = "nails" | "hands" | "both";

const WebcamCapture: React.FC<WebcamCaptureProps> = ({ onModelLoaded }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modelRef = useRef<tf.GraphModel | null>(null);
  const handsModelRef = useRef<any>(null);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [detections, setDetections] = useState<YoloDetection[]>([]);
  const [handDetections, setHandDetections] = useState<HandDetection[]>([]);
  const [detectionMode, setDetectionMode] = useState<DetectionMode>("nails");
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  const animationRef = useRef<number | undefined>(undefined);
  const lastTimeRef = useRef<number>(0);
  const frameCountRef = useRef<number>(0);
  const resizeObserverRef = useRef<ResizeObserver | null>(null);
  const pendingInferenceRef = useRef<boolean>(false);
  const lastInferenceTimeRef = useRef<number>(0);
  const capturedFrameRef = useRef<HTMLCanvasElement | null>(null);
  const syncedDetectionsRef = useRef<YoloDetection[]>([]);
  const syncedHandDetectionsRef = useRef<HandDetection[]>([]);
  const [selectedColor, setSelectedColor] = useState({
    r: 255,
    g: 107,
    b: 157,
    a: 0.6,
  }); // Default pink
  const [showColorPicker, setShowColorPicker] = useState(false);
  const [enableColorFilter, setEnableColorFilter] = useState(false);

  // Load the models
  const loadModel = useCallback(async () => {
    try {
      console.log("Loading models...");

      // Initialize TensorFlow.js backend
      await tf.ready();
      console.log("TensorFlow.js backend initialized");
      console.log("Available backends:", tf.getBackend());

      let nailModelLoaded = false;
      let handsModelLoaded = false;

      // Load nail segmentation model
      try {
        const modelUrl = "/model_web/model.json";
        const model = await tf.loadGraphModel(modelUrl);
        modelRef.current = model;
        nailModelLoaded = true;

        console.log("Nail segmentation model loaded successfully");
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
        console.log("Warming up nail model...");
        const dummyInput = tf.zeros([1, 640, 640, 3]);
        const warmupOutputs = (await model.executeAsync(
          dummyInput
        )) as tf.Tensor[];
        warmupOutputs.forEach((tensor) => tensor.dispose());
        dummyInput.dispose();
        console.log("Nail model warmup completed");
      } catch (error) {
        console.error("Error loading nail segmentation model:", error);
      }

      // Load MediaPipe hands model
      try {
        const hands = await initializeMediaPipeHands();

        hands.onResults((results: any) => {
          const handsResult = processMediaPipeResults(results);
          console.log(`MediaPipe detected ${handsResult.hands.length} hands`);

          // Update hand detections in sync
          setHandDetections(handsResult.hands);
          syncedHandDetectionsRef.current = handsResult.hands;
        });

        handsModelRef.current = hands;
        handsModelLoaded = true;
        console.log("MediaPipe hands model loaded successfully");
      } catch (error) {
        console.error("Error loading MediaPipe hands model:", error);
      }

      // Report overall loading status
      onModelLoaded(nailModelLoaded || handsModelLoaded);

      if (!nailModelLoaded && !handsModelLoaded) {
        setError("Failed to load any models");
      } else if (!nailModelLoaded) {
        setError("Nail segmentation model failed to load");
      } else if (!handsModelLoaded) {
        setError("Hand detection model failed to load");
      }
    } catch (error) {
      console.error("Error during model loading:", error);
      setError(
        `Failed to load models: ${
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
    setHandDetections([]); // Clear hand detections
    syncedDetectionsRef.current = []; // Clear synced detections too
    syncedHandDetectionsRef.current = []; // Clear synced hand detections
    pendingInferenceRef.current = false; // Reset pending state
    capturedFrameRef.current = null; // Clear frame reference
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
  }, []);

  // Run inference
  const runInference = useCallback(async () => {
    if (
      !videoRef.current ||
      isProcessing ||
      pendingInferenceRef.current ||
      (!modelRef.current && !handsModelRef.current)
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

      // Store the captured frame for synchronized display
      capturedFrameRef.current = canvas;

      // Run nail segmentation if enabled and model is loaded
      if (
        (detectionMode === "nails" || detectionMode === "both") &&
        modelRef.current
      ) {
        const preprocessed = preprocessImageForYolo(videoRef.current);

        // Run model inference
        const outputs = (await modelRef.current.executeAsync(
          preprocessed
        )) as tf.Tensor[];

        console.log(
          "Nail model outputs:",
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
            `Setting ${result.detections.length} nail detections from inference`
          );
          // Update both the state and the synced ref simultaneously
          setDetections(result.detections);
          syncedDetectionsRef.current = result.detections;
        }

        // Clean up tensors
        preprocessed.dispose();
        outputs.forEach((tensor) => tensor.dispose());
      } else {
        // Clear nail detections if not in nail mode
        setDetections([]);
        syncedDetectionsRef.current = [];
      }

      // Run hand detection if enabled and model is loaded
      if (
        (detectionMode === "hands" || detectionMode === "both") &&
        handsModelRef.current
      ) {
        // MediaPipe hands expects the video element directly
        await handsModelRef.current.send({ image: videoRef.current });
        // Results are handled in the onResults callback set during model loading
      } else {
        // Clear hand detections if not in hands mode
        setHandDetections([]);
        syncedHandDetectionsRef.current = [];
      }

      lastInferenceTimeRef.current = currentTime;

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
  }, [isProcessing, detectionMode]);

  // Draw captured frame with detections on canvas
  const drawDetections = useCallback(() => {
    if (!canvasRef.current || !capturedFrameRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const capturedFrame = capturedFrameRef.current;
    const frameWidth = capturedFrame.width;
    const frameHeight = capturedFrame.height;

    if (frameWidth === 0 || frameHeight === 0) return;

    // Get the video element's display dimensions
    const video = videoRef.current;
    if (!video) return;

    const displayWidth = video.clientWidth;
    const displayHeight = video.clientHeight;

    // Set canvas size to match the displayed video size exactly
    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
    }

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw the captured frame (this ensures perfect sync between frame and detections)
    ctx.drawImage(capturedFrame, 0, 0, displayWidth, displayHeight);

    // Calculate scaling factors (simple since we're drawing the exact frame that was analyzed)
    const scaleX = displayWidth / frameWidth;
    const scaleY = displayHeight / frameHeight;

    // Use synced detections
    const currentDetections = syncedDetectionsRef.current;
    const currentHandDetections = syncedHandDetectionsRef.current;

    console.log(
      `Drawing ${currentDetections.length} nail detections and ${currentHandDetections.length} hand detections on captured frame`
    );

    // Apply color filter if enabled (only for nails)
    if (
      enableColorFilter &&
      currentDetections.length > 0 &&
      (detectionMode === "nails" || detectionMode === "both")
    ) {
      // Scale detections to display size
      const scaledDetections = currentDetections.map((detection) => ({
        ...detection,
        bbox: [
          detection.bbox[0] * scaleX,
          detection.bbox[1] * scaleY,
          detection.bbox[2] * scaleX,
          detection.bbox[3] * scaleY,
        ],
        maskPolygon: detection.maskPolygon?.map((point) => [
          point[0] * scaleX,
          point[1] * scaleY,
        ]),
      }));

      applyNailColorFilter(canvas, scaledDetections, selectedColor);
    }

    // Draw nail detection outlines and labels
    if (detectionMode === "nails" || detectionMode === "both") {
      currentDetections.forEach((detection, index) => {
        const [x, y, width, height] = detection.bbox;

        // Scale coordinates to match display size
        const scaledX = x * scaleX;
        const scaledY = y * scaleY;
        const scaledWidth = width * scaleX;
        const scaledHeight = height * scaleY;

        // Validate dimensions
        if (scaledWidth <= 0 || scaledHeight <= 0) {
          console.warn(`Skipping nail detection ${index} - invalid dimensions`);
          return;
        }

        // Clamp coordinates to canvas bounds
        const clampedX = Math.max(0, Math.min(scaledX, canvas.width));
        const clampedY = Math.max(0, Math.min(scaledY, canvas.height));
        const clampedWidth = Math.min(scaledWidth, canvas.width - clampedX);
        const clampedHeight = Math.min(scaledHeight, canvas.height - clampedY);

        if (clampedWidth <= 0 || clampedHeight <= 0) {
          console.warn(
            `Skipping nail detection ${index} - out of bounds after clamping`
          );
          return;
        }

        // Draw precise boundary if available
        if (detection.maskPolygon && detection.maskPolygon.length > 2) {
          // Draw precise mask polygon outline
          ctx.strokeStyle = "#ff6b9d";
          ctx.lineWidth = 2;
          ctx.setLineDash([]);

          ctx.beginPath();
          const firstPoint = detection.maskPolygon[0];
          ctx.moveTo(firstPoint[0] * scaleX, firstPoint[1] * scaleY);

          for (let i = 1; i < detection.maskPolygon.length; i++) {
            const point = detection.maskPolygon[i];
            ctx.lineTo(point[0] * scaleX, point[1] * scaleY);
          }

          ctx.closePath();
          ctx.stroke();

          // Optional: draw semi-transparent fill
          if (!enableColorFilter) {
            ctx.fillStyle = "rgba(255, 107, 157, 0.2)";
            ctx.fill();
          }
        } else {
          // Fallback to enhanced bounding box with nail-like shape
          const radius = Math.min(clampedWidth, clampedHeight) * 0.3;

          ctx.strokeStyle = "#ff6b9d";
          ctx.lineWidth = 3;
          ctx.setLineDash([]);

          ctx.beginPath();
          ctx.roundRect(
            clampedX,
            clampedY,
            clampedWidth,
            clampedHeight,
            radius
          );
          ctx.stroke();

          if (!enableColorFilter) {
            ctx.fillStyle = "rgba(255, 107, 157, 0.2)";
            ctx.fill();
          }
        }

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

        // Draw corner indicators for enhanced UI
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
    }

    // Draw hand detections
    if (detectionMode === "hands" || detectionMode === "both") {
      // Create a temporary canvas for hand drawing that matches our display canvas
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = displayWidth;
      tempCanvas.height = displayHeight;

      drawHandDetections(tempCanvas, currentHandDetections, {
        drawLandmarks: true,
        drawConnections: true,
        landmarkColor: "#00ff00",
        connectionColor: "#0000ff",
        landmarkSize: 4,
        connectionWidth: 2,
      });

      // Draw the hand detection canvas onto our main canvas
      ctx.drawImage(tempCanvas, 0, 0);
    }
  }, [enableColorFilter, selectedColor, detectionMode]);

  // Main processing loop
  const processFrame = useCallback(() => {
    if (
      isWebcamActive &&
      videoRef.current &&
      (modelRef.current || handsModelRef.current)
    ) {
      // Only draw when we have a captured frame with detections
      if (
        capturedFrameRef.current &&
        (syncedDetectionsRef.current.length >= 0 ||
          syncedHandDetectionsRef.current.length >= 0)
      ) {
        drawDetections();
      }

      // Run inference at a controlled rate - this will capture frames and sync detections
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

  // Close color picker when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        showColorPicker &&
        !(event.target as Element)?.closest(".color-picker-container")
      ) {
        setShowColorPicker(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [showColorPicker]);

  // Set up resize observer to handle canvas resizing
  useEffect(() => {
    if (videoRef.current) {
      resizeObserverRef.current = new ResizeObserver(() => {
        // Trigger a redraw when video element is resized
        if (
          syncedDetectionsRef.current.length > 0 ||
          syncedHandDetectionsRef.current.length > 0
        ) {
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
    if (isWebcamActive && (modelRef.current || handsModelRef.current)) {
      processFrame();
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isWebcamActive, processFrame]);

  // Cleanup effect for MediaPipe
  useEffect(() => {
    return () => {
      disposeMediaPipeHands();
    };
  }, []);

  return (
    <div className="w-full max-w-4xl">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex flex-col items-center">
          {/* Single Video Feed with Overlaid Detections */}
          <div className="w-full max-w-2xl">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 justify-center">
              {detectionMode === "nails"
                ? "💅"
                : detectionMode === "hands"
                ? "👋"
                : "💅👋"}
              {detectionMode === "nails"
                ? "Nail Detection"
                : detectionMode === "hands"
                ? "Hand Detection"
                : "Nail & Hand Detection"}{" "}
              Camera
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
              {/* Hidden video element - used only for capture */}
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="absolute top-0 left-0 w-full h-full object-cover opacity-0 pointer-events-none"
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
                      (modelRef.current || handsModelRef.current) &&
                      videoRef.current &&
                      videoRef.current.videoWidth > 0
                    ) {
                      processFrame();
                    }
                  }, 100);
                }}
              />
              {/* Canvas that shows captured frames with detections */}
              <canvas
                ref={canvasRef}
                className="w-full h-full object-cover"
                style={{
                  backgroundColor: "#1f2937", // gray-800 fallback
                }}
              />
              {!isWebcamActive && (
                <div className="absolute inset-0 flex items-center justify-center text-white">
                  <div className="text-center">
                    <div className="text-4xl mb-2">📹</div>
                    <div>Camera not active</div>
                  </div>
                </div>
              )}
              {!capturedFrameRef.current && isWebcamActive && !isProcessing && (
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
            disabled={!modelRef.current && !handsModelRef.current}
            className={`px-6 py-2 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
              isWebcamActive
                ? "bg-red-500 hover:bg-red-600 text-white"
                : "bg-green-500 hover:bg-green-600 text-white"
            }`}
          >
            {isWebcamActive ? "🛑 Stop Camera" : "📷 Start Camera"}
          </button>

          {/* Detection Mode Toggle */}
          <div className="flex rounded-lg border border-gray-300 overflow-hidden">
            {(["nails", "hands", "both"] as DetectionMode[]).map((mode) => (
              <button
                key={mode}
                onClick={() => setDetectionMode(mode)}
                disabled={
                  (mode === "nails" && !modelRef.current) ||
                  (mode === "hands" && !handsModelRef.current) ||
                  (mode === "both" &&
                    !modelRef.current &&
                    !handsModelRef.current)
                }
                className={`px-4 py-2 font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                  detectionMode === mode
                    ? mode === "nails"
                      ? "bg-pink-500 text-white"
                      : mode === "hands"
                      ? "bg-blue-500 text-white"
                      : "bg-purple-500 text-white"
                    : "bg-white text-gray-700 hover:bg-gray-50"
                }`}
              >
                {mode === "nails"
                  ? "💅 Nails"
                  : mode === "hands"
                  ? "👋 Hands"
                  : "💅👋 Both"}
              </button>
            ))}
          </div>

          {/* Color Filter Toggle */}
          <button
            onClick={() => setEnableColorFilter(!enableColorFilter)}
            disabled={
              !isWebcamActive ||
              detections.length === 0 ||
              detectionMode === "hands"
            }
            className={`px-6 py-2 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
              enableColorFilter
                ? "bg-purple-500 hover:bg-purple-600 text-white"
                : "bg-gray-500 hover:bg-gray-600 text-white"
            }`}
          >
            {enableColorFilter ? "🎨 Color Filter ON" : "🎨 Color Filter OFF"}
          </button>

          {/* Color Picker */}
          {enableColorFilter && (
            <div className="relative color-picker-container">
              <button
                onClick={() => setShowColorPicker(!showColorPicker)}
                className="px-4 py-2 rounded-lg font-medium bg-white border-2 border-gray-300 hover:border-gray-400 transition-colors flex items-center gap-2"
              >
                <div
                  className="w-6 h-6 rounded border-2 border-gray-300"
                  style={{
                    backgroundColor: `rgba(${selectedColor.r}, ${selectedColor.g}, ${selectedColor.b}, 1)`,
                  }}
                ></div>
                Pick Color
              </button>

              {showColorPicker && (
                <div className="absolute top-12 left-0 z-10 bg-white rounded-lg shadow-lg border p-4 min-w-[300px]">
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Nail Color
                      </label>
                      <div className="grid grid-cols-6 gap-2">
                        {/* Predefined nail colors */}
                        {[
                          { name: "Classic Red", r: 220, g: 20, b: 60 },
                          { name: "Pink", r: 255, g: 107, b: 157 },
                          { name: "Coral", r: 255, g: 127, b: 80 },
                          { name: "Purple", r: 138, g: 43, b: 226 },
                          { name: "Blue", r: 30, g: 144, b: 255 },
                          { name: "Green", r: 50, g: 205, b: 50 },
                          { name: "Gold", r: 255, g: 215, b: 0 },
                          { name: "Silver", r: 192, g: 192, b: 192 },
                          { name: "Black", r: 0, g: 0, b: 0 },
                          { name: "White", r: 255, g: 255, b: 255 },
                          { name: "Orange", r: 255, g: 165, b: 0 },
                          { name: "Turquoise", r: 64, g: 224, b: 208 },
                        ].map((color, index) => (
                          <button
                            key={index}
                            onClick={() =>
                              setSelectedColor({ ...color, a: selectedColor.a })
                            }
                            className={`w-8 h-8 rounded border-2 hover:scale-110 transition-transform ${
                              selectedColor.r === color.r &&
                              selectedColor.g === color.g &&
                              selectedColor.b === color.b
                                ? "border-gray-800"
                                : "border-gray-300"
                            }`}
                            style={{
                              backgroundColor: `rgb(${color.r}, ${color.g}, ${color.b})`,
                            }}
                            title={color.name}
                          />
                        ))}
                      </div>
                    </div>

                    {/* Opacity Slider */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Opacity: {Math.round(selectedColor.a * 100)}%
                      </label>
                      <input
                        type="range"
                        min="0.1"
                        max="1"
                        step="0.1"
                        value={selectedColor.a}
                        onChange={(e) =>
                          setSelectedColor({
                            ...selectedColor,
                            a: parseFloat(e.target.value),
                          })
                        }
                        className="w-full"
                      />
                    </div>

                    {/* Preview */}
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-gray-700">Preview:</span>
                      <div
                        className="w-16 h-8 rounded border"
                        style={{
                          backgroundColor: `rgba(${selectedColor.r}, ${selectedColor.g}, ${selectedColor.b}, ${selectedColor.a})`,
                        }}
                      ></div>
                    </div>

                    <button
                      onClick={() => setShowColorPicker(false)}
                      className="w-full px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm font-medium transition-colors"
                    >
                      Close
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {isProcessing && (
            <div className="flex items-center gap-2 text-pink-600">
              <div className="animate-spin w-4 h-4 border-2 border-pink-600 border-t-transparent rounded-full"></div>
              <span>Analyzing...</span>
            </div>
          )}
        </div>

        {/* Stats */}
        {isWebcamActive && (
          <div className="mt-4 grid grid-cols-2 md:grid-cols-6 gap-4 text-center">
            <div className="bg-pink-50 rounded-lg p-3">
              <div className="text-2xl font-bold text-pink-600">
                {detections.length}
              </div>
              <div className="text-sm text-gray-600">Nails Detected</div>
            </div>
            <div className="bg-blue-50 rounded-lg p-3">
              <div className="text-2xl font-bold text-blue-600">
                {handDetections.length}
              </div>
              <div className="text-sm text-gray-600">Hands Detected</div>
            </div>
            <div className="bg-green-50 rounded-lg p-3">
              <div className="text-2xl font-bold text-green-600">
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
              <div className="text-2xl font-bold text-orange-600">
                {detections.some((d) => d.maskPolygon) ? "✓" : "○"}
              </div>
              <div className="text-sm text-gray-600">Precise Bounds</div>
            </div>
            <div className="bg-indigo-50 rounded-lg p-3">
              <div className="text-2xl font-bold text-indigo-600">
                {detectionMode.toUpperCase()}
              </div>
              <div className="text-sm text-gray-600">Detection Mode</div>
            </div>
          </div>
        )}

        {/* Detection Details */}
        {(detections.length > 0 || handDetections.length > 0) && (
          <div className="mt-4">
            <h4 className="text-md font-semibold mb-2">Detection Details</h4>
            <div className="space-y-2 max-h-32 overflow-y-auto">
              {/* Nail Detections */}
              {detections.map((detection, index) => (
                <div
                  key={`nail-${index}`}
                  className="text-sm bg-pink-50 rounded p-2 flex justify-between items-center"
                >
                  <div className="flex items-center gap-2">
                    <span>💅 Nail #{index + 1}</span>
                    {detection.maskPolygon && (
                      <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded">
                        Precise
                      </span>
                    )}
                    {detection.mask && (
                      <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded">
                        Mask
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-pink-600">
                      {(detection.score * 100).toFixed(1)}%
                    </span>
                    {enableColorFilter && (
                      <div
                        className="w-4 h-4 rounded border"
                        style={{
                          backgroundColor: `rgba(${selectedColor.r}, ${selectedColor.g}, ${selectedColor.b}, 1)`,
                        }}
                      ></div>
                    )}
                  </div>
                </div>
              ))}

              {/* Hand Detections */}
              {handDetections.map((hand, index) => (
                <div
                  key={`hand-${index}`}
                  className="text-sm bg-blue-50 rounded p-2 flex justify-between items-center"
                >
                  <div className="flex items-center gap-2">
                    <span>
                      👋 {hand.handedness} Hand #{index + 1}
                    </span>
                    <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded">
                      {hand.landmarks.length} Landmarks
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-blue-600">
                      {(hand.score * 100).toFixed(1)}%
                    </span>
                  </div>
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
                if (!modelRef.current && !handsModelRef.current) {
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
