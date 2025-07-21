/**
 * Three.js 3D Nail Overlay System
 *
 * This module provides 3D nail overlay functionality using Three.js to render
 * rectangular prism "fake nails" on top of detected and matched real nails.
 * The system handles:
 * 1. Creating and managing Three.js scene, camera, and renderer
 * 2. Positioning 3D nail prisms based on nail detection and orientation data
 * 3. Proper lighting and shading for realistic appearance
 * 4. Smooth transitions and updates for real-time tracking
 */

import * as THREE from "three";
import { NailFingerMatch } from "./nailMatching";

export interface ThreeNailOverlayConfig {
  canvasWidth: number;
  canvasHeight: number;
  videoWidth: number;
  videoHeight: number;
  enableLighting: boolean;
  nailThickness: number; // Depth of the nail prism (Z-axis)
  nailOpacity: number;
  showWireframe: boolean;
}

export class ThreeNailOverlay {
  private scene: THREE.Scene;
  private camera: THREE.OrthographicCamera;
  private renderer: THREE.WebGLRenderer;
  private nailMeshes: Map<string, THREE.Mesh> = new Map();
  private directionalLight!: THREE.DirectionalLight;
  private ambientLight!: THREE.AmbientLight;
  private config: ThreeNailOverlayConfig;
  private containerElement: HTMLElement;
  private isInitialized = false;

  // Material for the nail prisms
  private nailMaterial!: THREE.MeshPhongMaterial;
  private wireframeMaterial!: THREE.MeshBasicMaterial;

  constructor(containerElement: HTMLElement, config: ThreeNailOverlayConfig) {
    this.containerElement = containerElement;
    this.config = { ...config };

    // Initialize Three.js scene
    this.scene = new THREE.Scene();
    this.scene.background = null; // Transparent background

    // Create orthographic camera for 2D overlay projection
    const { canvasWidth, canvasHeight } = config;
    this.camera = new THREE.OrthographicCamera(
      -canvasWidth / 2,
      canvasWidth / 2, // left, right (centered)
      canvasHeight / 2,
      -canvasHeight / 2, // top, bottom (Y-axis flipped to match canvas)
      -1000,
      1000 // near, far
    );
    this.camera.position.set(0, 0, 100);

    // Create WebGL renderer
    this.renderer = new THREE.WebGLRenderer({
      alpha: true,
      antialias: true,
      preserveDrawingBuffer: true,
    });
    this.renderer.setSize(canvasWidth, canvasHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    // Style the Three.js canvas to overlay perfectly
    this.renderer.domElement.style.position = "absolute";
    this.renderer.domElement.style.top = "0";
    this.renderer.domElement.style.left = "0";
    this.renderer.domElement.style.width = "100%";
    this.renderer.domElement.style.height = "100%";
    this.renderer.domElement.style.pointerEvents = "none";
    this.renderer.domElement.style.zIndex = "5";

    // Create materials
    this.nailMaterial = new THREE.MeshPhongMaterial({
      color: 0xff69b4, // Pink color for fake nails
      transparent: true,
      opacity: config.nailOpacity,
      shininess: 100,
      specular: 0x444444,
    });

    this.wireframeMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      wireframe: true,
      transparent: true,
      opacity: 0.8,
    });

    // Set up lighting
    this.setupLighting();

    // Append renderer to container
    this.containerElement.appendChild(this.renderer.domElement);
    this.isInitialized = true;

    console.log("Three.js nail overlay initialized");
  }

  private setupLighting(): void {
    // Ambient light for general illumination
    this.ambientLight = new THREE.AmbientLight(0x404040, 0.4);
    this.scene.add(this.ambientLight);

    // Directional light for shadows and highlights
    this.directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    this.directionalLight.position.set(100, 100, 200);
    this.directionalLight.castShadow = true;

    // Configure shadow properties
    this.directionalLight.shadow.mapSize.width = 2048;
    this.directionalLight.shadow.mapSize.height = 2048;
    this.directionalLight.shadow.camera.near = 0.5;
    this.directionalLight.shadow.camera.far = 500;
    this.directionalLight.shadow.camera.left = -200;
    this.directionalLight.shadow.camera.right = 200;
    this.directionalLight.shadow.camera.top = 200;
    this.directionalLight.shadow.camera.bottom = -200;

    this.scene.add(this.directionalLight);
  }

  /**
   * Update the overlay with new nail matches
   */
  public updateNailOverlays(
    matches: NailFingerMatch[],
    scaleX: number,
    scaleY: number
  ): void {
    if (!this.isInitialized) return;

    // Clear existing meshes that are no longer matched
    const currentKeys = new Set(
      matches.map((match) => this.createNailKey(match))
    );

    this.nailMeshes.forEach((mesh, key) => {
      if (!currentKeys.has(key)) {
        this.scene.remove(mesh);
        this.nailMeshes.delete(key);
      }
    });

    // Update or create meshes for current matches
    matches.forEach((match) => {
      this.updateNailMesh(match, scaleX, scaleY);
    });

    // Render the scene
    this.render();
  }

  private createNailKey(match: NailFingerMatch): string {
    return `${match.handedness}_${match.fingertipIndex}`;
  }

  private updateNailMesh(
    match: NailFingerMatch,
    scaleX: number,
    scaleY: number
  ): void {
    const key = this.createNailKey(match);

    // Scale nail dimensions to canvas coordinates
    const scaledWidth = match.nailWidth * scaleX;
    const scaledHeight = match.nailHeight * scaleY;
    const thickness = this.config.nailThickness;

    // Create geometry for rectangular prism with proper dimensions
    // Use the nail's actual dimensions but ensure minimum size for visibility
    const minSize = 12;
    let finalWidth = Math.max(scaledWidth, minSize);
    let finalHeight = Math.max(scaledHeight, minSize * 0.6);

    // Ensure the longer dimension is used as the nail length (along finger direction)
    // The nail width in the data represents the length along the finger
    const geometry = new THREE.BoxGeometry(finalWidth, finalHeight, thickness);

    let mesh = this.nailMeshes.get(key);

    if (!mesh) {
      // Create new mesh
      const material = this.config.showWireframe
        ? this.wireframeMaterial
        : this.nailMaterial;
      mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = true;
      mesh.receiveShadow = true;

      this.scene.add(mesh);
      this.nailMeshes.set(key, mesh);

      console.log(`Created new nail mesh for ${key}`);
    } else {
      // Update existing mesh geometry
      mesh.geometry.dispose();
      mesh.geometry = geometry;
    }

    // Position the mesh at the nail centroid (convert to Three.js coordinate system)
    const scaledCentroid = [
      match.nailCentroid[0] * scaleX,
      match.nailCentroid[1] * scaleY,
    ];

    // Convert from canvas coordinates to Three.js coordinates
    // Canvas: (0,0) at top-left, X right, Y down
    // Three.js: (0,0) at center, X right, Y up
    const threeX = scaledCentroid[0] - this.config.canvasWidth / 2;
    const threeY = -(scaledCentroid[1] - this.config.canvasHeight / 2); // Flip Y and center

    mesh.position.set(
      threeX,
      threeY,
      thickness / 2 // Slightly above the surface
    );

    // Apply nail rotation - rotate around Z axis to match nail orientation
    // The nail angle is already in the correct coordinate system from nailMatching
    mesh.rotation.z = match.nailAngle;

    // Debug logging (remove in production)
    if (Math.random() < 0.1) {
      console.log(`3D Nail ${key}:`, {
        canvas: scaledCentroid,
        three: [threeX, threeY],
        angle: ((match.nailAngle * 180) / Math.PI).toFixed(1) + "°",
        size: [finalWidth.toFixed(1), finalHeight.toFixed(1)],
        canvasSize: [this.config.canvasWidth, this.config.canvasHeight],
      });
    }

    // Add some dynamic rotation for visual interest
    // Temporarily disabled for debugging - uncomment when positioning is correct
    // const time = Date.now() * 0.001;
    // mesh.rotation.x = Math.sin(time * 0.5) * 0.1; // Subtle X tilt
    // mesh.rotation.y = Math.cos(time * 0.3) * 0.05; // Subtle Y tilt
    mesh.rotation.x = 0;
    mesh.rotation.y = 0;

    // Update material color based on confidence
    if (
      !this.config.showWireframe &&
      mesh.material instanceof THREE.MeshPhongMaterial
    ) {
      const confidence = match.matchConfidence;
      const hue = confidence * 0.3; // 0 (red) to 0.3 (green)
      const color = new THREE.Color().setHSL(hue, 0.8, 0.6);
      mesh.material.color = color;
    }
  }

  /**
   * Toggle between solid and wireframe display
   */
  public setWireframeMode(enabled: boolean): void {
    this.config.showWireframe = enabled;

    this.nailMeshes.forEach((mesh) => {
      mesh.material = enabled ? this.wireframeMaterial : this.nailMaterial;
    });

    this.render();
  }

  /**
   * Update overlay opacity
   */
  public setOpacity(opacity: number): void {
    this.config.nailOpacity = opacity;
    this.nailMaterial.opacity = opacity;
    this.wireframeMaterial.opacity = opacity * 0.8;
    this.render();
  }

  /**
   * Update nail thickness
   */
  public setNailThickness(thickness: number): void {
    this.config.nailThickness = thickness;
    // Note: Existing meshes will update their thickness on next updateNailOverlays call
  }

  /**
   * Resize the overlay to match new canvas dimensions
   */
  public resize(width: number, height: number): void {
    this.config.canvasWidth = width;
    this.config.canvasHeight = height;

    // Update camera to maintain centered coordinate system
    this.camera.left = -width / 2;
    this.camera.right = width / 2;
    this.camera.top = height / 2;
    this.camera.bottom = -height / 2;
    this.camera.updateProjectionMatrix();

    this.renderer.setSize(width, height);
    this.render();
  }

  /**
   * Render the Three.js scene
   */
  public render(): void {
    if (!this.isInitialized) return;
    this.renderer.render(this.scene, this.camera);
  }

  /**
   * Clean up Three.js resources
   */
  public dispose(): void {
    if (!this.isInitialized) return;

    // Dispose of all meshes and geometries
    this.nailMeshes.forEach((mesh) => {
      mesh.geometry.dispose();
      if (Array.isArray(mesh.material)) {
        mesh.material.forEach((material) => material.dispose());
      } else {
        mesh.material.dispose();
      }
      this.scene.remove(mesh);
    });
    this.nailMeshes.clear();

    // Dispose of materials
    this.nailMaterial.dispose();
    this.wireframeMaterial.dispose();

    // Remove renderer from DOM
    if (this.containerElement.contains(this.renderer.domElement)) {
      this.containerElement.removeChild(this.renderer.domElement);
    }

    // Dispose of renderer
    this.renderer.dispose();

    this.isInitialized = false;
    console.log("Three.js nail overlay disposed");
  }

  /**
   * Set the base color of the nail material
   */
  public setNailColor(color: number): void {
    this.nailMaterial.color.setHex(color);
    this.render();
  }

  /**
   * Enable or disable lighting effects
   */
  public setLightingEnabled(enabled: boolean): void {
    this.config.enableLighting = enabled;
    this.ambientLight.visible = enabled;
    this.directionalLight.visible = enabled;
    this.render();
  }

  /**
   * Get current configuration
   */
  public getConfig(): ThreeNailOverlayConfig {
    return { ...this.config };
  }
}
