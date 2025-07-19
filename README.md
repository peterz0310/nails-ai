# 🔬 Nails AI - Real-time Nail Segmentation

An intelligent web application that uses AI-powered computer vision to detect and segment fingernails in real-time through your webcam. Built with YOLOv8 segmentation model and powered by Next.js and TensorFlow.js.

## ✨ Features

- **Real-time Nail Detection**: Advanced YOLOv8 segmentation model for accurate nail detection
- **Live Webcam Integration**: Real-time processing of webcam feed
- **Custom Color Highlighting**: Customizable nail highlighting with color picker
- **Performance Optimized**: Efficient inference with TensorFlow.js WebGL backend
- **Modern UI**: Clean, responsive interface built with Tailwind CSS
- **Model Status Indicator**: Visual feedback for model loading and processing status

## 🚀 Getting Started

### Prerequisites

- Node.js 18+
- Modern web browser with webcam access
- WebGL support for optimal performance

### Installation

1. Clone the repository:

```bash
git clone https://github.com/peterz0310/nails-ai.git
cd nails-ai
```

2. Install dependencies:

```bash
npm install
# or
yarn install
# or
pnpm install
```

3. Run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

### First Use

1. Allow webcam permissions when prompted
2. Wait for the AI model to load (indicated by green status)
3. Position your hand in front of the camera
4. Watch as the AI detects and highlights your nails in real-time!

## 🛠️ Technology Stack

- **Framework**: Next.js 15.4.2 with App Router
- **AI/ML**: TensorFlow.js with YOLOv8 segmentation model
- **Styling**: Tailwind CSS 4
- **Language**: TypeScript
- **Computer Vision**: Custom YOLO utilities for preprocessing and postprocessing

## 📁 Project Structure

```
src/
├── app/
│   ├── page.tsx          # Main application page
│   ├── layout.tsx        # App layout
│   └── globals.css       # Global styles
├── components/
│   └── WebcamCapture.tsx # Webcam capture and AI inference component
└── utils/
    └── yolo.ts           # YOLO model utilities and processing functions

public/
└── model_web/            # Pre-trained YOLOv8 segmentation model
    ├── model.json        # Model architecture
    ├── metadata.yaml     # Model metadata
    └── group1-shard*.bin # Model weights (12 shards)
```

## 🎯 How It Works

1. **Model Loading**: The app loads a custom-trained YOLOv8 segmentation model specifically trained for nail detection
2. **Image Preprocessing**: Webcam frames are preprocessed to 640x640 resolution for optimal model performance
3. **AI Inference**: The model processes frames and outputs bounding boxes and segmentation masks
4. **Postprocessing**: Detections are filtered using confidence thresholds and Non-Maximum Suppression (NMS)
5. **Visualization**: Detected nails are highlighted with customizable colors and overlays

## ⚙️ Configuration

### Model Parameters

- **Input Size**: 640x640 pixels
- **Confidence Threshold**: 0.5 (adjustable)
- **NMS Threshold**: 0.45
- **Inference Rate**: ~2 FPS (optimized for performance)

### Performance Tips

- Ensure good lighting for better detection accuracy
- Keep hands steady for consistent tracking
- Use a modern browser with WebGL support for optimal performance

## 🔧 Development

### Building for Production

```bash
npm run build
npm start
```

### Linting

```bash
npm run lint
```

## 📝 Model Information

The application uses a custom YOLOv8s-seg model trained specifically for nail segmentation:

- **Model Type**: YOLOv8s-seg (segmentation)
- **Training Data**: Custom nail dataset
- **License**: AGPL-3.0 (Ultralytics)
- **Version**: 8.3.168

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the segmentation model
- [TensorFlow.js](https://www.tensorflow.org/js) for browser-based ML inference
- [Next.js](https://nextjs.org/) for the React framework
