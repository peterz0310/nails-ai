/**
 * Three.js 3D Nail Overlay System
 *
 * This module provides 3D nail overlay functionality using Three.js to render
 * realistic 3D nail prisms on top of detected and matched real nails.
 *
 * Features:
 * - Real-time 3D nail rendering with proper positioning and rotation
 * - Multiple material types (standard, metallic, wireframe)
 * - Dynamic lighting with shadows and reflections
 * - Configurable nail properties (thickness, opacity, curvature)
 * - Smooth coordinate system mapping from video to 3D space
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
  metallicIntensity: number; // 0-1, how metallic the nails appear
  roughness: number; // 0-1, surface roughness (0 = mirror, 1 = rough)
  enableReflections: boolean;
  enable3DRotation: boolean; // Enable subtle 3D rotation for visual interest
  nailCurvature: number; // 0-1, how curved the nail surface is
}

export class ThreeNailOverlay {
  private scene: THREE.Scene;
  private camera: THREE.OrthographicCamera;
  private renderer: THREE.WebGLRenderer;
  private nailMeshes: Map<string, THREE.Mesh> = new Map();
  private directionalLight!: THREE.DirectionalLight;
  private ambientLight!: THREE.AmbientLight;
  private pointLight!: THREE.PointLight;
  private config: ThreeNailOverlayConfig;
  private containerElement: HTMLElement;
  private isInitialized = false;

  // Materials for the nail prisms
  private nailMaterial!: THREE.MeshStandardMaterial;
  private wireframeMaterial!: THREE.MeshBasicMaterial;
  private metallicMaterial!: THREE.MeshPhysicalMaterial;

  constructor(containerElement: HTMLElement, config: ThreeNailOverlayConfig) {
    this.containerElement = containerElement;
    this.config = { ...config };

    // Initialize Three.js scene
    this.scene = new THREE.Scene();
    this.scene.background = null; // Transparent background

    // Create perspective camera for proper 3D visualization
    const { canvasWidth, canvasHeight } = config;
    this.camera = new THREE.OrthographicCamera(
      -canvasWidth / 2,
      canvasWidth / 2, // left, right (centered)
      canvasHeight / 2,
      -canvasHeight / 2, // top, bottom (Y-axis flipped to match canvas)
      0.1, // near - closer for better 3D depth
      2000 // far - increased range
    );
    this.camera.position.set(0, 0, 300); // Move camera further back
    this.camera.lookAt(0, 0, 0); // Ensure camera looks at center

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
    this.renderer.domElement.style.zIndex = "15"; // Higher than container's zIndex: 10

    // Create materials
    this.nailMaterial = new THREE.MeshStandardMaterial({
      color: 0xff69b4, // Pink color for fake nails
      transparent: true,
      opacity: config.nailOpacity,
      metalness: config.metallicIntensity || 0.3,
      roughness: config.roughness || 0.4,
    });

    this.metallicMaterial = new THREE.MeshPhysicalMaterial({
      color: 0xf0c0ff, // Slightly lighter metallic pink
      transparent: true,
      opacity: config.nailOpacity,
      metalness: config.metallicIntensity || 0.8,
      roughness: config.roughness || 0.2,
      clearcoat: 1.0,
      clearcoatRoughness: 0.1,
      reflectivity: config.enableReflections ? 0.9 : 0.3,
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

    console.log("3D nail overlay initialized successfully");
  }

  private setupLighting(): void {
    // Ambient light for general illumination
    this.ambientLight = new THREE.AmbientLight(0x404040, 0.3);
    this.scene.add(this.ambientLight);

    // Main directional light for primary shadows and highlights
    this.directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
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

    // Add a point light for enhanced metallic reflections
    this.pointLight = new THREE.PointLight(0xffffff, 0.8, 400);
    this.pointLight.position.set(-50, 50, 150);
    this.pointLight.castShadow = true;
    this.scene.add(this.pointLight);

    // Add rim lighting for better 3D appearance
    const rimLight = new THREE.DirectionalLight(0x8888ff, 0.4);
    rimLight.position.set(-100, -50, 100);
    this.scene.add(rimLight);
  }

  /**
   * Update the overlay with new nail matches
   */
  public updateNailOverlays(
    matches: NailFingerMatch[],
    scaleX: number,
    scaleY: number
  ): void {
    if (!this.isInitialized) {
      console.warn("3D overlay not initialized, skipping update");
      return;
    }

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

    // Always render the scene (test cube or nail meshes)
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

    // Create curved nail geometry for more realistic appearance
    let geometry: THREE.BufferGeometry;

    if (this.config.nailCurvature > 0) {
      // Create a slightly curved nail using cylinder geometry
      const segments = 8;
      geometry = new THREE.CylinderGeometry(
        finalWidth / 2, // radiusTop
        finalWidth / 2, // radiusBottom
        thickness, // height
        segments, // radialSegments
        1, // heightSegments
        false, // openEnded
        0, // thetaStart
        Math.PI * (0.8 + this.config.nailCurvature * 0.4) // thetaLength - partial cylinder for curve
      );
      // Rotate to lay flat
      geometry.rotateZ(Math.PI / 2);
      geometry.scale(1, finalHeight / finalWidth, 1);
    } else {
      // Standard rectangular prism
      geometry = new THREE.BoxGeometry(finalWidth, finalHeight, thickness);
    }

    let mesh = this.nailMeshes.get(key);

    if (!mesh) {
      // Create new mesh - choose material based on config
      let material: THREE.Material;
      if (this.config.showWireframe) {
        material = this.wireframeMaterial;
      } else if (this.config.metallicIntensity > 0.6) {
        material = this.metallicMaterial;
      } else {
        material = this.nailMaterial;
      }

      mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = true;
      mesh.receiveShadow = true;

      this.scene.add(mesh);
      this.nailMeshes.set(key, mesh);
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
      thickness // Position at thickness level, not half
    );

    // Apply nail rotation - rotate around Z axis to match nail orientation
    // The nail angle is already in the correct coordinate system from nailMatching
    mesh.rotation.z = match.nailAngle;

    // Add dynamic 3D rotation if enabled
    if (this.config.enable3DRotation) {
      const time = Date.now() * 0.001;
      mesh.rotation.x = Math.sin(time * 0.5 + key.charCodeAt(0)) * 0.15; // Subtle X tilt
      mesh.rotation.y = Math.cos(time * 0.3 + key.charCodeAt(1)) * 0.1; // Subtle Y tilt
    } else {
      mesh.rotation.x = 0.1; // Slight permanent tilt to show 3D
      mesh.rotation.y = 0.05;
    }

    // Update material color based on confidence
    if (!this.config.showWireframe) {
      const confidence = match.matchConfidence;
      const hue = confidence * 0.3; // 0 (red) to 0.3 (green)
      const color = new THREE.Color().setHSL(hue, 0.8, 0.6);

      if (
        mesh.material instanceof THREE.MeshStandardMaterial ||
        mesh.material instanceof THREE.MeshPhysicalMaterial
      ) {
        mesh.material.color = color;
      }
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
    this.metallicMaterial.dispose();
    this.wireframeMaterial.dispose();

    // Remove renderer from DOM
    if (this.containerElement.contains(this.renderer.domElement)) {
      this.containerElement.removeChild(this.renderer.domElement);
    }

    // Dispose of renderer
    this.renderer.dispose();

    this.isInitialized = false;
    console.log("3D nail overlay disposed successfully");
  }

  /**
   * Set the base color of the nail material
   */
  public setNailColor(color: number): void {
    this.nailMaterial.color.setHex(color);
    this.metallicMaterial.color.setHex(color);
    this.render();
  }

  /**
   * Set metallic intensity (0-1)
   */
  public setMetallicIntensity(intensity: number): void {
    this.config.metallicIntensity = Math.max(0, Math.min(1, intensity));
    this.nailMaterial.metalness = this.config.metallicIntensity;
    this.metallicMaterial.metalness = Math.max(
      0.6,
      this.config.metallicIntensity
    );

    // Switch materials based on metallic intensity
    this.nailMeshes.forEach((mesh) => {
      if (!this.config.showWireframe) {
        mesh.material =
          this.config.metallicIntensity > 0.6
            ? this.metallicMaterial
            : this.nailMaterial;
      }
    });

    this.render();
  }

  /**
   * Set surface roughness (0-1)
   */
  public setRoughness(roughness: number): void {
    this.config.roughness = Math.max(0, Math.min(1, roughness));
    this.nailMaterial.roughness = this.config.roughness;
    this.metallicMaterial.roughness = this.config.roughness * 0.5; // Keep metallic smoother
    this.render();
  }

  /**
   * Enable/disable reflections
   */
  public setReflectionsEnabled(enabled: boolean): void {
    this.config.enableReflections = enabled;
    this.metallicMaterial.reflectivity = enabled ? 0.9 : 0.3;
    this.render();
  }

  /**
   * Enable/disable 3D rotation animation
   */
  public set3DRotationEnabled(enabled: boolean): void {
    this.config.enable3DRotation = enabled;
    if (!enabled) {
      // Reset rotations to flat
      this.nailMeshes.forEach((mesh) => {
        mesh.rotation.x = 0;
        mesh.rotation.y = 0;
      });
    }
    this.render();
  }

  /**
   * Set nail curvature (0-1)
   */
  public setNailCurvature(curvature: number): void {
    this.config.nailCurvature = Math.max(0, Math.min(1, curvature));
    // Note: Existing meshes will update their curvature on next updateNailOverlays call
  }

  /**
   * Enable or disable lighting effects
   */
  public setLightingEnabled(enabled: boolean): void {
    this.config.enableLighting = enabled;
    this.ambientLight.visible = enabled;
    this.directionalLight.visible = enabled;
    this.pointLight.visible = enabled;
    this.render();
  }

  /**
   * Get current configuration
   */
  public getConfig(): ThreeNailOverlayConfig {
    return { ...this.config };
  }
}
