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
import {
  matchNailsToFingertips,
  drawNailFingerMatches,
  NailFingerMatch,
} from "../utils/nailMatching";

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
  const [nailFingerMatches, setNailFingerMatches] = useState<NailFingerMatch[]>(
    []
  );
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
  const syncedDetectionsRef = useRef<YoloDetection[]>([]);
  const syncedHandDetectionsRef = useRef<HandDetection[]>([]);
  const currentDetectionModeRef = useRef<DetectionMode>(detectionMode);
  const lastDrawTimeRef = useRef<number>(0);
  const lastNailInferenceRef = useRef<number>(0);
  const lastHandInferenceRef = useRef<number>(0);
  const [selectedColor, setSelectedColor] = useState({
    r: 255,
    g: 107,
    b: 157,
    a: 0.6,
  }); // Default pink
  const [showColorPicker, setShowColorPicker] = useState(false);
  const [enableColorFilter, setEnableColorFilter] = useState(false);

  // Detection mode change handler with cleanup
  const handleDetectionModeChange = useCallback(
    (newMode: DetectionMode) => {
      console.log(
        `Switching detection mode from ${detectionMode} to ${newMode}`
      );

      // Clear existing detections when switching modes
      if (newMode !== detectionMode) {
        if (newMode === "nails") {
          // Switching to nails-only: clear hands and matches immediately
          setHandDetections([]);
          syncedHandDetectionsRef.current = [];
          setNailFingerMatches([]);
          console.log("Cleared hand detections for nails-only mode");
        } else if (newMode === "hands") {
          // Switching to hands-only: clear nails and matches immediately
          setDetections([]);
          syncedDetectionsRef.current = [];
          setNailFingerMatches([]);
          console.log("Cleared nail detections for hands-only mode");
        } else if (newMode === "both") {
          // Switching to both: clear matches to recalculate, keep existing detections
          setNailFingerMatches([]);
          console.log("Cleared matches for both mode - will recalculate");
        }
      }

      setDetectionMode(newMode);
      currentDetectionModeRef.current = newMode;

      // Reset inference timing to ensure immediate processing with new mode
      lastNailInferenceRef.current = 0;
      lastHandInferenceRef.current = 0;
      pendingInferenceRef.current = false;

      // Force immediate canvas clearing and redraw with new mode
      if (canvasRef.current && videoRef.current) {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          // Clear the canvas
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          // Draw fresh video frame without any detections
          const video = videoRef.current;
          if (video.videoWidth > 0 && video.videoHeight > 0) {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          }
        }
      }

      // Reset drawing time to force immediate redraw
      lastDrawTimeRef.current = 0;

      console.log(`Mode switch complete. New mode: ${newMode}`);
    },
    [detectionMode]
  );

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

          // Use ref to get current detection mode (since callback is set once)
          const currentMode = currentDetectionModeRef.current;

          // Only process if we're still in a mode that uses hands
          if (currentMode === "hands" || currentMode === "both") {
            if (handsResult.hands.length > 0) {
              console.log(
                `MediaPipe detected ${handsResult.hands.length} hands in ${currentMode} mode`
              );

              // Update hand detections in sync
              setHandDetections(handsResult.hands);
              syncedHandDetectionsRef.current = handsResult.hands;

              // Calculate nail-finger matches if we have both models active and nail detections
              if (
                currentMode === "both" &&
                syncedDetectionsRef.current.length > 0
              ) {
                // Get current video dimensions
                const video = videoRef.current;
                if (video && video.videoWidth > 0 && video.videoHeight > 0) {
                  const matches = matchNailsToFingertips(
                    syncedDetectionsRef.current,
                    handsResult.hands,
                    video.videoWidth,
                    video.videoHeight
                  );
                  setNailFingerMatches(matches);
                  console.log(
                    `Updated nail-finger matches from hand detection: ${matches.length} matches found`
                  );
                }
              }
            } else {
              // More conservative clearing - only clear when no hands detected for current mode
              if (currentMode === "hands") {
                setHandDetections([]);
                syncedHandDetectionsRef.current = [];
                setNailFingerMatches([]); // Clear matches when hands disappear
              } else if (currentMode === "both") {
                // In both mode, clear hand detections but keep nail detections
                setHandDetections([]);
                syncedHandDetectionsRef.current = [];
                setNailFingerMatches([]); // Clear matches since hands are gone
                console.log(
                  "No hands detected in both mode - cleared hand detections and matches"
                );
              }
            }
          }
          // If we're in nails-only mode, completely ignore hand detection results
          // This prevents any interference with nails-only mode
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

  // Improved stop webcam function with better cleanup
  const stopWebcam = useCallback(() => {
    if (videoRef.current?.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsWebcamActive(false);
    setDetections([]); // Clear detections when stopping
    setHandDetections([]); // Clear hand detections
    setNailFingerMatches([]); // Clear nail-finger matches
    syncedDetectionsRef.current = []; // Clear synced detections too
    syncedHandDetectionsRef.current = []; // Clear synced hand detections
    pendingInferenceRef.current = false; // Reset pending state

    // Clean up timing references
    lastDrawTimeRef.current = 0;
    lastNailInferenceRef.current = 0;
    lastHandInferenceRef.current = 0;

    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = undefined;
    }

    // Force cleanup of any remaining tensors
    try {
      const tensorCount = tf.memory().numTensors;
      if (tensorCount > 10) {
        // Arbitrary threshold
        console.warn(
          `High tensor count detected: ${tensorCount}. Forcing cleanup...`
        );
        // Force garbage collection if available
        if (typeof window !== "undefined" && "gc" in window) {
          (window as any).gc();
        }
      }
    } catch (error) {
      console.warn("Error during tensor cleanup:", error);
    }
  }, []);

  // Optimized inference with better frame synchronization and scheduling
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
      return;
    }

    const currentTime = performance.now();

    // Improved timing strategy: separate intervals for different models
    let shouldRunNails = false;
    let shouldRunHands = false;

    if (detectionMode === "nails" && modelRef.current) {
      // Nail detection every 600ms for better responsiveness
      if (currentTime - lastNailInferenceRef.current > 600) {
        shouldRunNails = true;
        lastNailInferenceRef.current = currentTime;
      }
    } else if (detectionMode === "hands" && handsModelRef.current) {
      // Hand detection every 500ms for smooth tracking
      if (currentTime - lastHandInferenceRef.current > 500) {
        shouldRunHands = true;
        lastHandInferenceRef.current = currentTime;
      }
    } else if (detectionMode === "both") {
      // Staggered execution: nail detection every 600ms, hand detection every 800ms with 300ms offset
      if (
        modelRef.current &&
        currentTime - lastNailInferenceRef.current > 600
      ) {
        shouldRunNails = true;
        lastNailInferenceRef.current = currentTime;
      }
      if (
        handsModelRef.current &&
        currentTime - lastHandInferenceRef.current > 800 &&
        currentTime - lastNailInferenceRef.current > 300
      ) {
        shouldRunHands = true;
        lastHandInferenceRef.current = currentTime;
      }
    }

    if (!shouldRunNails && !shouldRunHands) {
      return;
    }

    pendingInferenceRef.current = true;
    setIsProcessing(true);

    try {
      // Run nail segmentation if needed
      if (shouldRunNails && modelRef.current) {
        const preprocessed = preprocessImageForYolo(videoRef.current!);

        try {
          const outputs = (await modelRef.current.executeAsync(
            preprocessed
          )) as tf.Tensor[];

          const result = processYoloOutput(
            outputs,
            videoWidth,
            videoHeight,
            0.4, // Lowered confidence threshold for better detection
            0.5 // Slightly higher NMS threshold for better deduplication
          );

          // Update nail detections only if we're still in the right mode
          if (
            currentDetectionModeRef.current === "nails" ||
            currentDetectionModeRef.current === "both"
          ) {
            setDetections(result.detections);
            syncedDetectionsRef.current = result.detections;

            // Calculate nail-finger matches if we have both models active
            if (
              currentDetectionModeRef.current === "both" &&
              syncedHandDetectionsRef.current.length > 0 &&
              result.detections.length > 0
            ) {
              const matches = matchNailsToFingertips(
                result.detections,
                syncedHandDetectionsRef.current,
                videoWidth,
                videoHeight
              );
              setNailFingerMatches(matches);
            }
          }

          // Clean up tensors explicitly
          outputs.forEach((tensor) => tensor.dispose());
        } finally {
          preprocessed.dispose();
        }
      }

      // Run hand detection if needed
      if (shouldRunHands && handsModelRef.current) {
        try {
          await handsModelRef.current.send({ image: videoRef.current });
        } catch (error) {
          console.error("Hand detection error:", error);
        }
      }

      // Calculate FPS less frequently
      frameCountRef.current++;
      const fpsCurrentTime = performance.now();
      if (fpsCurrentTime - lastTimeRef.current >= 2000) {
        setFps(Math.round(frameCountRef.current / 2));
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

  // Optimized drawing with better performance and frame synchronization
  const drawDetections = useCallback(() => {
    if (!canvasRef.current || !videoRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const video = videoRef.current;

    if (!ctx || video.videoWidth === 0 || video.videoHeight === 0) return;

    const displayWidth = video.clientWidth;
    const displayHeight = video.clientHeight;

    // Only resize canvas when dimensions actually change to prevent flashing
    if (
      Math.abs(canvas.width - displayWidth) > 1 ||
      Math.abs(canvas.height - displayHeight) > 1
    ) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
    }

    // Clear and draw the live video feed consistently across all modes
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, displayWidth, displayHeight);

    // Calculate scaling factors for overlay elements
    const scaleX = displayWidth / video.videoWidth;
    const scaleY = displayHeight / video.videoHeight;

    // Use synced detections and filter by current mode
    const currentDetections =
      detectionMode === "hands" ? [] : syncedDetectionsRef.current;
    const currentHandDetections =
      detectionMode === "nails" ? [] : syncedHandDetectionsRef.current;
    const currentMatches = detectionMode === "both" ? nailFingerMatches : [];

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

    // Draw nail detection outlines and labels with improved efficiency
    if (
      (detectionMode === "nails" || detectionMode === "both") &&
      currentDetections.length > 0
    ) {
      // Reduce excessive logging
      if (Math.random() < 0.02) {
        console.log(
          `Drawing ${currentDetections.length} nail detections in mode: ${detectionMode}`
        );
      }

      currentDetections.forEach((detection, index) => {
        const [x, y, width, height] = detection.bbox;

        // Scale coordinates to match display size
        const scaledX = x * scaleX;
        const scaledY = y * scaleY;
        const scaledWidth = width * scaleX;
        const scaledHeight = height * scaleY;

        // Validate dimensions
        if (scaledWidth <= 0 || scaledHeight <= 0) return;

        // Clamp coordinates to canvas bounds with improved logic
        const clampedX = Math.max(0, Math.min(scaledX, canvas.width - 1));
        const clampedY = Math.max(0, Math.min(scaledY, canvas.height - 1));
        const clampedWidth = Math.min(scaledWidth, canvas.width - clampedX);
        const clampedHeight = Math.min(scaledHeight, canvas.height - clampedY);

        if (clampedWidth <= 0 || clampedHeight <= 0) return;

        // Draw precise boundary if available
        if (detection.maskPolygon && detection.maskPolygon.length > 2) {
          // Draw precise mask polygon outline
          ctx.strokeStyle = "#ff6b9d";
          ctx.lineWidth = 2;
          ctx.setLineDash([]);

          ctx.beginPath();
          const firstPoint = detection.maskPolygon[0];
          const firstX = Math.max(
            0,
            Math.min(firstPoint[0] * scaleX, canvas.width)
          );
          const firstY = Math.max(
            0,
            Math.min(firstPoint[1] * scaleY, canvas.height)
          );
          ctx.moveTo(firstX, firstY);

          for (let i = 1; i < detection.maskPolygon.length; i++) {
            const point = detection.maskPolygon[i];
            const pointX = Math.max(
              0,
              Math.min(point[0] * scaleX, canvas.width)
            );
            const pointY = Math.max(
              0,
              Math.min(point[1] * scaleY, canvas.height)
            );
            ctx.lineTo(pointX, pointY);
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

        // Optimized label drawing
        const label = `${(detection.score * 100).toFixed(0)}%`;
        ctx.font = "bold 12px Arial";
        const textWidth = ctx.measureText(label).width;

        // Improved label positioning with bounds checking
        const labelX = Math.max(
          2,
          Math.min(clampedX, canvas.width - textWidth - 10)
        );
        const labelY = Math.max(18, Math.min(clampedY, canvas.height - 4));

        ctx.fillStyle = "#ff6b9d";
        ctx.fillRect(labelX - 2, labelY - 16, textWidth + 8, 18);

        ctx.fillStyle = "white";
        ctx.fillText(label, labelX + 2, labelY - 3);
      });
    }

    // Draw hand detections efficiently
    if (
      (detectionMode === "hands" || detectionMode === "both") &&
      currentHandDetections.length > 0
    ) {
      drawHandDetections(canvas, currentHandDetections, {
        drawLandmarks: true,
        drawConnections: true,
        landmarkColor: "#00ff88",
        connectionColor: "#0088ff",
        landmarkSize: 3,
        connectionWidth: 2,
      });
    }

    // Draw nail-finger matches if we have both types of detections
    if (detectionMode === "both" && currentMatches.length > 0) {
      drawNailFingerMatches(ctx, currentMatches, scaleX, scaleY);
    }
  }, [enableColorFilter, selectedColor, detectionMode, nailFingerMatches]);

  // Optimized main processing loop with better timing control
  const processFrame = useCallback(() => {
    if (
      isWebcamActive &&
      videoRef.current &&
      (modelRef.current || handsModelRef.current) &&
      videoRef.current.videoWidth > 0
    ) {
      const currentTime = performance.now();

      // Consistent frame rate for all modes to prevent inconsistency (60 FPS drawing)
      const drawInterval = 16; // ~60 FPS for smooth video
      const shouldDraw = currentTime - lastDrawTimeRef.current >= drawInterval;

      if (shouldDraw) {
        drawDetections();
        lastDrawTimeRef.current = currentTime;
      }

      // Optimized inference timing - different intervals for different modes
      if (!pendingInferenceRef.current && !isProcessing) {
        let shouldRunInference = false;

        if (detectionMode === "nails" && modelRef.current) {
          // Nail inference every 600ms for good responsiveness without overloading
          if (currentTime - lastNailInferenceRef.current >= 600) {
            shouldRunInference = true;
          }
        } else if (detectionMode === "hands" && handsModelRef.current) {
          // Hand inference every 500ms for smooth tracking
          if (currentTime - lastHandInferenceRef.current >= 500) {
            shouldRunInference = true;
          }
        } else if (detectionMode === "both") {
          // For both mode, check if either model is ready with proper staggering
          const nailReady =
            modelRef.current &&
            currentTime - lastNailInferenceRef.current >= 600;
          const handReady =
            handsModelRef.current &&
            currentTime - lastHandInferenceRef.current >= 800 &&
            currentTime - lastNailInferenceRef.current >= 300; // 300ms offset

          if (nailReady || handReady) {
            shouldRunInference = true;
          }
        }

        if (shouldRunInference) {
          runInference();
        }
      }

      animationRef.current = requestAnimationFrame(processFrame);
    }
  }, [
    isWebcamActive,
    runInference,
    drawDetections,
    isProcessing,
    detectionMode,
  ]);

  // Handle detection mode changes
  useEffect(() => {
    console.log(`Detection mode changed to: ${detectionMode}`);
    currentDetectionModeRef.current = detectionMode;

    // Force immediate redraw when mode changes by resetting draw timing
    lastDrawTimeRef.current = 0;

    // Force a redraw immediately to show the mode change
    if (isWebcamActive && canvasRef.current && videoRef.current) {
      setTimeout(() => {
        drawDetections();
      }, 50); // Small delay to ensure state has updated
    }
  }, [detectionMode, drawDetections, isWebcamActive]);

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
                ? "Nail Detection & Analysis"
                : detectionMode === "hands"
                ? "Hand Tracking & Analysis"
                : "Nail & Hand AI Analysis"}
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
              {isWebcamActive && (
                <div className="absolute top-4 left-4 bg-black bg-opacity-50 text-white px-3 py-2 rounded-lg">
                  <div className="text-sm">
                    🔍 Place your hand in front of the camera
                  </div>
                  {fps > 0 && (
                    <div className="text-xs mt-1 opacity-75">
                      Rendering: ~60 FPS | Inference: {fps} FPS
                    </div>
                  )}
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
                onClick={() => handleDetectionModeChange(mode)}
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
            <div className="bg-cyan-50 rounded-lg p-3">
              <div className="text-2xl font-bold text-cyan-600">
                {nailFingerMatches.length}
              </div>
              <div className="text-sm text-gray-600">Nail-Finger Matches</div>
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
            <div className="bg-red-50 rounded-lg p-3">
              <div className="text-2xl font-bold text-red-600">
                {tf.memory().numTensors}
              </div>
              <div className="text-sm text-gray-600">Memory (Tensors)</div>
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

              {/* Nail-Finger Matches */}
              {nailFingerMatches.map((match, index) => {
                const fingerNames = [
                  "Thumb",
                  "Index",
                  "Middle",
                  "Ring",
                  "Pinky",
                ];
                const fingerTips = [4, 8, 12, 16, 20];
                const fingerIndex = fingerTips.indexOf(match.fingertipIndex);
                const fingerName =
                  fingerIndex >= 0 ? fingerNames[fingerIndex] : "Unknown";

                return (
                  <div
                    key={`match-${index}`}
                    className="text-sm bg-cyan-50 rounded p-2 flex justify-between items-center"
                  >
                    <div className="flex items-center gap-2">
                      <span>
                        🎯 {match.handedness} {fingerName}
                      </span>
                      <span className="text-xs bg-cyan-100 text-cyan-700 px-2 py-1 rounded">
                        {match.nailWidth.toFixed(0)}×
                        {match.nailHeight.toFixed(0)}px
                      </span>
                      <span className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded">
                        {((match.nailAngle * 180) / Math.PI).toFixed(0)}°
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-cyan-600">
                        {(match.matchConfidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                );
              })}
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
