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
import {
  ThreeNailOverlay,
  ThreeNailOverlayConfig,
} from "../utils/threeNailOverlay";

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
  const detectionMode = "both" as const; // Always use both models
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
  const currentDetectionModeRef = useRef<DetectionMode>("both");
  const lastDrawTimeRef = useRef<number>(0);
  const lastNailInferenceRef = useRef<number>(0);
  const lastHandInferenceRef = useRef<number>(0);
  // Add frame synchronization tracking
  const capturedFrameRef = useRef<ImageData | null>(null);
  const frameTimestampRef = useRef<number>(0);
  const nailResultsRef = useRef<{
    detections: YoloDetection[];
    timestamp: number;
  } | null>(null);
  const handResultsRef = useRef<{
    hands: HandDetection[];
    timestamp: number;
  } | null>(null);
  const [selectedColor, setSelectedColor] = useState({
    r: 255,
    g: 107,
    b: 157,
    a: 0.6,
  }); // Default pink
  const [showColorPicker, setShowColorPicker] = useState(false);
  const [enableColorFilter, setEnableColorFilter] = useState(false);

  // Overlay visibility controls
  const [showNailLabels, setShowNailLabels] = useState(false);
  const [showNailOutlines, setShowNailOutlines] = useState(false);
  const [showConfidenceScores, setShowConfidenceScores] = useState(false);
  const [showNailFingerMatches, setShowNailFingerMatches] = useState(false);
  const [showHandLandmarks, setShowHandLandmarks] = useState(false);

  // 3D Overlay state and controls - always enabled
  const show3DOverlay = true; // Always show 3D overlay
  const [show3DWireframe, setShow3DWireframe] = useState(false); // Default to solid
  const nail3DOpacity = 0.8; // Fixed reasonable defaults
  const nail3DThickness = 6;
  const nail3DMetallic = 0.7;
  const nail3DRoughness = 0.3;
  const nail3DCurvature = 0.6;
  const enable3DRotation = true; // Always enabled
  const enable3DReflections = true;

  // Nail texture upload state
  const [uploadedTexture, setUploadedTexture] = useState<string | null>(null);
  const [textureOpacity, setTextureOpacity] = useState(1.0);
  const [showTextureControls, setShowTextureControls] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const threeOverlayRef = useRef<ThreeNailOverlay | null>(null);
  const threeContainerRef = useRef<HTMLDivElement>(null);

  // Track canvas dimensions for 3D overlay
  const [overlayCanvasDimensions, setOverlayCanvasDimensions] = useState({
    width: 0,
    height: 0,
  });

  // Add smoothing for nail orientations to reduce jitter
  const nailOrientationHistoryRef = useRef<Map<string, number[]>>(new Map());
  const smoothingWindowSize = 5; // Number of frames to average

  // Smooth nail orientations to reduce jitter
  const smoothNailOrientations = useCallback(
    (matches: NailFingerMatch[]): NailFingerMatch[] => {
      const history = nailOrientationHistoryRef.current;

      return matches.map((match) => {
        // Create a unique key for this nail (hand + finger)
        const key = `${match.handedness}_${match.fingertipIndex}`;

        // Get or create history for this nail
        if (!history.has(key)) {
          history.set(key, []);
        }

        const angles = history.get(key)!;

        // Add current angle to history
        angles.push(match.nailAngle);

        // Keep only the last N angles
        if (angles.length > smoothingWindowSize) {
          angles.shift();
        }

        // Calculate smoothed angle using circular mean for angles
        let sumSin = 0;
        let sumCos = 0;
        angles.forEach((angle) => {
          sumSin += Math.sin(angle);
          sumCos += Math.cos(angle);
        });

        const smoothedAngle = Math.atan2(
          sumSin / angles.length,
          sumCos / angles.length
        );

        // Return match with smoothed angle
        return {
          ...match,
          nailAngle: smoothedAngle,
        };
      });
    },
    [] // Removed smoothingWindowSize from dependencies as it's constant
  );

  // Handle image upload for nail texture
  const handleTextureUpload = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (file) {
        // Validate file type
        if (!file.type.startsWith("image/")) {
          alert("Please select a valid image file.");
          return;
        }

        // Validate file size (max 5MB)
        if (file.size > 5 * 1024 * 1024) {
          alert("Image file size must be less than 5MB.");
          return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
          const base64 = e.target?.result as string;
          setUploadedTexture(base64);
          console.log("Nail texture uploaded successfully");
        };
        reader.onerror = () => {
          console.error("Error reading uploaded file");
          alert("Error reading uploaded file. Please try again.");
        };
        reader.readAsDataURL(file);
      }
    },
    []
  );

  // Remove uploaded texture
  const removeTexture = useCallback(() => {
    setUploadedTexture(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    console.log("Nail texture removed");
  }, []);

  // Helper function to check if we have synchronized results for matching
  const trySyncMatchUpdate = useCallback(() => {
    const nailResults = nailResultsRef.current;
    const handResults = handResultsRef.current;

    // Only calculate matches if we have recent results from both models
    // and they're from a similar timeframe (within 500ms)
    if (
      nailResults &&
      handResults &&
      Math.abs(nailResults.timestamp - handResults.timestamp) < 500 &&
      nailResults.detections.length > 0 &&
      handResults.hands.length > 0
    ) {
      const video = videoRef.current;
      if (video && video.videoWidth > 0 && video.videoHeight > 0) {
        const matches = matchNailsToFingertips(
          nailResults.detections,
          handResults.hands,
          video.videoWidth,
          video.videoHeight
        );
        const smoothedMatches = smoothNailOrientations(matches);
        setNailFingerMatches(smoothedMatches);
        console.log(
          `Synchronized nail-finger matches: ${
            matches.length
          } matches from frames ${Math.abs(
            nailResults.timestamp - handResults.timestamp
          ).toFixed(0)}ms apart`
        );
      }
    }
  }, [smoothNailOrientations]);

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

              // Store hand results with timestamp for synchronized matching
              handResultsRef.current = {
                hands: handsResult.hands,
                timestamp: performance.now(),
              };

              // Try to update matches with synchronized data
              trySyncMatchUpdate();
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
                handResultsRef.current = null; // Clear stored hand results
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
  }, [onModelLoaded, smoothNailOrientations, trySyncMatchUpdate]);

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
    nailOrientationHistoryRef.current.clear(); // Clear smoothing history
    syncedDetectionsRef.current = []; // Clear synced detections too
    syncedHandDetectionsRef.current = []; // Clear synced hand detections
    nailResultsRef.current = null; // Clear stored nail results
    handResultsRef.current = null; // Clear stored hand results
    capturedFrameRef.current = null; // Clear captured frame
    frameTimestampRef.current = 0; // Reset frame timestamp
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

    // Synchronized timing strategy: run both models together but at a slower rate
    // for better frame synchronization and accuracy
    let shouldRunNails = false;
    let shouldRunHands = false;

    // Run both models together every 800ms for better synchronization
    const inferenceInterval = 800; // Slower but more synchronized
    const timeSinceLastInference = Math.min(
      currentTime - lastNailInferenceRef.current,
      currentTime - lastHandInferenceRef.current
    );

    if (timeSinceLastInference >= inferenceInterval) {
      if (modelRef.current) {
        shouldRunNails = true;
        lastNailInferenceRef.current = currentTime;
      }
      if (handsModelRef.current) {
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

            // Store nail results with timestamp for synchronized matching
            nailResultsRef.current = {
              detections: result.detections,
              timestamp: performance.now(),
            };

            // Try to update matches with synchronized data
            trySyncMatchUpdate();
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
  }, [isProcessing, smoothNailOrientations, trySyncMatchUpdate]);

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

    // Use synced detections - always show both
    const currentDetections = syncedDetectionsRef.current;
    const currentHandDetections = syncedHandDetectionsRef.current;
    const currentMatches = nailFingerMatches;

    // Apply color filter if enabled (only for nails)
    if (enableColorFilter && currentDetections.length > 0) {
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
      currentDetections.length > 0 &&
      (showNailOutlines || showNailLabels || showConfidenceScores)
    ) {
      // Reduce excessive logging
      if (Math.random() < 0.02) {
        console.log(`Drawing ${currentDetections.length} nail detections`);
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

        // Draw precise boundary if available and outlines are enabled
        if (
          showNailOutlines &&
          detection.maskPolygon &&
          detection.maskPolygon.length > 2
        ) {
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
        } else if (showNailOutlines) {
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

        // Draw confidence labels if enabled
        if (showNailLabels && showConfidenceScores) {
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
        }
      });
    }

    // Draw hand detections efficiently
    if (currentHandDetections.length > 0 && showHandLandmarks) {
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
    if (currentMatches.length > 0 && showNailFingerMatches) {
      drawNailFingerMatches(ctx, currentMatches, scaleX, scaleY);
    }

    // Update 3D overlay if enabled (separate from match visualization)
    if (show3DOverlay && threeOverlayRef.current) {
      // Update canvas size if needed
      if (
        canvas.width !== overlayCanvasDimensions.width ||
        canvas.height !== overlayCanvasDimensions.height
      ) {
        threeOverlayRef.current.resize(canvas.width, canvas.height);
        setOverlayCanvasDimensions({
          width: canvas.width,
          height: canvas.height,
        });
      }

      // Update 3D nail overlays with current matches
      threeOverlayRef.current.updateNailOverlays(
        currentMatches,
        scaleX,
        scaleY
      );
    }
  }, [
    enableColorFilter,
    selectedColor,
    nailFingerMatches,
    show3DOverlay,
    showConfidenceScores,
    showHandLandmarks,
    showNailFingerMatches,
    showNailLabels,
    showNailOutlines,
  ]);

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

      // Synchronized inference timing - both models together for better frame accuracy
      if (!pendingInferenceRef.current && !isProcessing) {
        // Use the same synchronized timing as in runInference
        const inferenceInterval = 800; // Slower but more synchronized
        const timeSinceLastInference = Math.min(
          currentTime - lastNailInferenceRef.current,
          currentTime - lastHandInferenceRef.current
        );

        if (timeSinceLastInference >= inferenceInterval) {
          runInference();
        }
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
      animationRef.current = requestAnimationFrame(processFrame);
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

  // 3D Overlay initialization and cleanup
  useEffect(() => {
    if (
      show3DOverlay &&
      threeContainerRef.current &&
      !threeOverlayRef.current
    ) {
      const container = threeContainerRef.current;
      const canvas = canvasRef.current;

      console.log("Attempting to initialize 3D overlay", {
        hasContainer: !!container,
        hasCanvas: !!canvas,
        canvasSize: canvas ? [canvas.width, canvas.height] : "N/A",
      });

      if (canvas && canvas.width > 0 && canvas.height > 0) {
        const config: ThreeNailOverlayConfig = {
          canvasWidth: canvas.width,
          canvasHeight: canvas.height,
          enableLighting: true,
          nailThickness: nail3DThickness,
          nailOpacity: nail3DOpacity,
          showWireframe: show3DWireframe,
          metallicIntensity: nail3DMetallic,
          roughness: nail3DRoughness,
          enable3DRotation: enable3DRotation,
          nailCurvature: nail3DCurvature,
          nailColor: { r: 255, g: 107, b: 157 }, // Default pink color
          // Texture settings
          nailTexture: uploadedTexture,
          textureOpacity: textureOpacity,
        };

        try {
          threeOverlayRef.current = new ThreeNailOverlay(container, config);
          setOverlayCanvasDimensions({
            width: canvas.width,
            height: canvas.height,
          });
          console.log("3D nail overlay initialized successfully");
        } catch (error) {
          console.error("Failed to initialize 3D overlay:", error);
          // 3D overlay is always enabled, so just log the error
        }
      } else {
        console.warn("Canvas not ready for 3D overlay initialization");
      }
    } else if (!show3DOverlay && threeOverlayRef.current) {
      threeOverlayRef.current.dispose();
      threeOverlayRef.current = null;
      console.log("3D nail overlay disposed");
    }
  }, [
    show3DOverlay,
    nail3DThickness,
    nail3DOpacity,
    show3DWireframe,
    nail3DMetallic,
    nail3DRoughness,
    enable3DReflections,
    enable3DRotation,
    nail3DCurvature,
  ]);

  // Helper function to update 3D overlay configuration
  const updateThreeOverlayConfig = useCallback(() => {
    if (threeOverlayRef.current && canvasRef.current) {
      const config: ThreeNailOverlayConfig = {
        canvasWidth: overlayCanvasDimensions.width || canvasRef.current.width,
        canvasHeight:
          overlayCanvasDimensions.height || canvasRef.current.height,
        enableLighting: true,
        nailThickness: nail3DThickness,
        nailOpacity: nail3DOpacity,
        showWireframe: show3DWireframe,
        metallicIntensity: nail3DMetallic,
        roughness: nail3DRoughness,
        enable3DRotation: enable3DRotation,
        nailCurvature: nail3DCurvature,
        nailColor: { r: 255, g: 107, b: 157 }, // Default pink color
        // Texture settings
        nailTexture: uploadedTexture,
        textureOpacity: textureOpacity,
      };
      threeOverlayRef.current.updateConfig(config);
    }
  }, [
    overlayCanvasDimensions,
    nail3DThickness,
    nail3DOpacity,
    show3DWireframe,
    nail3DMetallic,
    nail3DRoughness,
    enable3DRotation,
    nail3DCurvature,
    uploadedTexture,
    textureOpacity,
  ]);

  // Handle 3D overlay setting changes
  useEffect(() => {
    updateThreeOverlayConfig();
  }, [updateThreeOverlayConfig]);

  // Handle texture updates separately to avoid re-initializing the entire overlay
  useEffect(() => {
    if (threeOverlayRef.current && uploadedTexture !== undefined) {
      threeOverlayRef.current.setTexture(uploadedTexture);
    }
  }, [uploadedTexture]);

  return (
    <div className="w-full max-w-4xl">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex flex-col items-center">
          {/* Single Video Feed with Overlaid Detections */}
          <div className="w-full max-w-2xl">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 justify-center">
              💅👋 Nail & Hand AI Analysis
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
                      animationRef.current =
                        requestAnimationFrame(processFrame);
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
              {/* 3D Overlay Container */}
              <div
                ref={threeContainerRef}
                className="absolute inset-0 pointer-events-none"
                style={{
                  display: show3DOverlay ? "block" : "none",
                  zIndex: 10,
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

          {/* Color Filter Toggle */}
          <button
            onClick={() => setEnableColorFilter(!enableColorFilter)}
            disabled={!isWebcamActive || detections.length === 0}
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

          {/* Overlay Visibility Controls */}
          {isWebcamActive && (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-medium text-gray-700">Show:</span>

              <label className="flex items-center gap-1 text-sm">
                <input
                  type="checkbox"
                  checked={showNailOutlines}
                  onChange={(e) => setShowNailOutlines(e.target.checked)}
                  className="rounded text-pink-500 focus:ring-pink-500 focus:ring-1"
                />
                Nail Outlines
              </label>

              <label className="flex items-center gap-1 text-sm">
                <input
                  type="checkbox"
                  checked={showNailLabels && showConfidenceScores}
                  onChange={(e) => {
                    setShowNailLabels(e.target.checked);
                    setShowConfidenceScores(e.target.checked);
                  }}
                  className="rounded text-pink-500 focus:ring-pink-500 focus:ring-1"
                />
                Confidence Labels
              </label>

              <label className="flex items-center gap-1 text-sm">
                <input
                  type="checkbox"
                  checked={showHandLandmarks}
                  onChange={(e) => setShowHandLandmarks(e.target.checked)}
                  className="rounded text-blue-500 focus:ring-blue-500 focus:ring-1"
                />
                Hand Landmarks
              </label>

              <label className="flex items-center gap-1 text-sm">
                <input
                  type="checkbox"
                  checked={showNailFingerMatches}
                  onChange={(e) => setShowNailFingerMatches(e.target.checked)}
                  className="rounded text-cyan-500 focus:ring-cyan-500 focus:ring-1"
                />
                Match Arrows & Labels
              </label>
            </div>
          )}

          {/* 3D Overlay Controls */}
          <div className="flex flex-wrap items-center gap-2">
            {nailFingerMatches.length > 0 && (
              <span className="text-xs text-green-600 bg-green-100 px-2 py-1 rounded">
                {nailFingerMatches.length} nail
                {nailFingerMatches.length !== 1 ? "s" : ""} matched
              </span>
            )}

            {/* Wireframe Toggle */}
            <button
              onClick={() => {
                setShow3DWireframe(!show3DWireframe);
              }}
              className={`px-3 py-2 rounded-lg font-medium transition-colors ${
                show3DWireframe
                  ? "bg-white border-2 border-cyan-500 text-cyan-700"
                  : "bg-cyan-100 text-cyan-700 hover:bg-cyan-200"
              }`}
            >
              {show3DWireframe ? "📐 Wireframe" : "🎯 Solid"}
            </button>

            {/* Texture Controls Toggle */}
            <button
              onClick={() => {
                setShowTextureControls(!showTextureControls);
              }}
              className={`px-3 py-2 rounded-lg font-medium transition-colors ${
                showTextureControls
                  ? "bg-white border-2 border-purple-500 text-purple-700"
                  : "bg-purple-100 text-purple-700 hover:bg-purple-200"
              }`}
            >
              {uploadedTexture ? "🖼️ Texture ON" : "🖼️ Add Texture"}
            </button>
          </div>

          {/* Texture Upload Controls */}
          {showTextureControls && (
            <div className="mt-4 p-4 bg-purple-50 border border-purple-200 rounded-lg">
              <h3 className="text-sm font-bold text-purple-800 mb-3">
                🖼️ Nail Texture Controls
              </h3>

              <div className="space-y-4">
                {/* File Upload */}
                <div>
                  <label className="block text-sm font-medium text-purple-700 mb-2">
                    Upload Image Texture
                  </label>
                  <div className="flex items-center gap-2">
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*"
                      onChange={handleTextureUpload}
                      className="text-sm text-purple-700 file:mr-2 file:py-1 file:px-2 file:rounded file:border-0 file:text-sm file:font-medium file:bg-purple-100 file:text-purple-700 hover:file:bg-purple-200"
                    />
                    {uploadedTexture && (
                      <button
                        onClick={removeTexture}
                        className="px-2 py-1 bg-red-100 text-red-700 rounded text-sm hover:bg-red-200"
                      >
                        Remove
                      </button>
                    )}
                  </div>
                  <div className="text-xs text-purple-600 mt-1">
                    Supported formats: JPG, PNG, GIF. Max size: 5MB
                  </div>
                </div>

                {/* Texture Opacity */}
                {uploadedTexture && (
                  <div>
                    <label className="block text-sm font-medium text-purple-700 mb-2">
                      Texture Opacity: {Math.round(textureOpacity * 100)}%
                    </label>
                    <input
                      type="range"
                      min="0.1"
                      max="1"
                      step="0.1"
                      value={textureOpacity}
                      onChange={(e) =>
                        setTextureOpacity(parseFloat(e.target.value))
                      }
                      className="w-full"
                    />
                  </div>
                )}

                {/* Preview */}
                {uploadedTexture && (
                  <div>
                    <label className="block text-sm font-medium text-purple-700 mb-2">
                      Texture Preview:
                    </label>
                    <div className="w-20 h-20 border border-purple-300 rounded overflow-hidden">
                      <img
                        src={uploadedTexture}
                        alt="Nail texture preview"
                        className="w-full h-full object-cover"
                      />
                    </div>
                  </div>
                )}
              </div>
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
