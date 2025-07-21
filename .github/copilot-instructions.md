# Nails AI - Real-time Nail Segmentation

This is a Next.js application that performs real-time nail detection and segmentation using a custom YOLOv8 model with TensorFlow.js.

## Architecture Overview

**Core Components:**

- `src/app/page.tsx` - Main UI with model status indicator and webcam feed
- `src/components/WebcamCapture.tsx` - Heart of the app: handles webcam, model inference, and real-time visualization
- `src/utils/yolo.ts` - YOLOv8 processing pipeline: preprocessing, postprocessing, NMS, mask generation
- `public/model_web/` - Custom-trained YOLOv8s-seg model (12 binary shards + metadata)

**Data Flow:**

1. Webcam video → preprocessing (640x640, normalized) → TensorFlow.js inference
2. Model outputs: predictions tensor + mask prototypes → YOLO processing → detections with masks
3. Frame synchronization: captured frame + detections → canvas rendering with overlays

## Critical Development Patterns

### Inference Performance Management

- **Frame throttling**: Inference limited to ~2 FPS (`500ms` intervals) to prevent overwhelming
- **Memory management**: All tensors disposed immediately after use with `tf.tidy()` and manual cleanup
- **Async coordination**: `pendingInferenceRef` prevents concurrent inference calls
- **Frame synchronization**: `capturedFrameRef` + `syncedDetectionsRef` ensure visual consistency

### State Management Architecture

```tsx
// Dual state pattern for real-time sync
const [detections, setDetections] = useState<YoloDetection[]>([]);
const syncedDetectionsRef = useRef<YoloDetection[]>([]);
// Update both simultaneously for consistency
setDetections(result.detections);
syncedDetectionsRef.current = result.detections;
```

### YOLO Processing Pipeline

- **Input**: YOLOv8-seg expects `[1, 640, 640, 3]` normalized tensors
- **Output format**: `outputs[0]` = predictions `[1, features, 8400]`, `outputs[1]` = mask prototypes `[1, channels, 160, 160]`
- **Processing**: Confidence filtering (0.5) → NMS (0.45) → mask generation → polygon extraction
- **Coordinate system**: All processing in original video dimensions, scaled for display

### Model Loading & Warmup

```tsx
const modelUrl = "/model_web/model.json";
const model = await tf.loadGraphModel(modelUrl);
// Essential: warm up with dummy inference
const dummyInput = tf.zeros([1, 640, 640, 3]);
const warmupOutputs = await model.executeAsync(dummyInput);
```

## Essential Development Commands

```bash
npm run dev        # Development with Turbopack
npm run build      # Production build
npm run lint       # ESLint with Next.js config
```

## Configuration Specifics

**Next.js webpack config** (`next.config.ts`):

- `.bin` files ignored with `ignore-loader` (prevents processing model weights)
- Turbopack enabled for dev server performance

**TensorFlow.js setup**:

- WebGL backend required for performance (`@tensorflow/tfjs-backend-webgl`)
- Model files served from `public/model_web/` (must be accessible via HTTP)

**TypeScript config**: ES2017 target for TensorFlow.js compatibility

## Debugging & Performance

**Key logging points:**

- Model loading: Input/output shapes, warmup completion
- Inference: `"Model outputs:"` with tensor shapes/dtypes
- YOLO processing: Detection counts before/after NMS
- Frame sync: `"Setting X detections from inference"`

**Performance monitoring:**

- FPS counter updates every 1000ms
- Memory usage via browser DevTools (watch for tensor leaks)
- Inference timing in console logs

**Common issues:**

- Black video: Check webcam permissions and browser compatibility
- Memory leaks: Ensure all `tf.Tensor` objects are disposed
- Poor detection: Verify lighting, hand positioning, model confidence thresholds
- Frame desync: Check `pendingInferenceRef` and synchronization logic

## UI/Visualization Patterns

**Canvas rendering approach:**

- Video frame captured → processed with detections → drawn to display canvas
- Dual rendering: color filtering (when enabled) + outline overlays
- Coordinate scaling: `scaleX/scaleY` for video → canvas dimension mapping
- Bounds clamping: All drawn elements constrained to canvas boundaries

**Detection visualization:**

- Precise polygon outlines when mask available
- Rounded rectangle fallback for bounding boxes only
- Color picker integration with real-time preview
- Confidence score labels with background fills

## Notes:

- Assume the dev server is already running.
