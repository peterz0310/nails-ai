"use client";

import { useState } from "react";
import WebcamCapture from "../components/WebcamCapture";

export default function Home() {
  const [isModelLoaded, setIsModelLoaded] = useState(false);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-6xl mx-auto">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            🔬 Nail AI Segmentation
          </h1>
          <p className="text-gray-600 mb-4">
            Real-time nail detection and highlighting using YOLOv8 segmentation
          </p>
          <div className="flex items-center justify-center gap-2">
            <div
              className={`w-3 h-3 rounded-full ${
                isModelLoaded ? "bg-green-500" : "bg-red-500"
              }`}
            ></div>
            <span className="text-sm text-gray-600">
              Model Status: {isModelLoaded ? "Loaded" : "Loading..."}
            </span>
          </div>
        </header>

        <main className="flex flex-col items-center">
          <WebcamCapture onModelLoaded={setIsModelLoaded} />
        </main>
      </div>
    </div>
  );
}
