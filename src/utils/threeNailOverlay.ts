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
  nailThickness: number; // Depth of the nail prism (Z-axis) - for press-on nails, should be thin (2-4px)
  nailOpacity: number;
  showWireframe: boolean;
  metallicIntensity: number; // 0-1, how metallic the nails appear
  roughness: number; // 0-1, surface roughness (0 = mirror, 1 = rough)
  enableReflections: boolean;
  enable3DRotation: boolean; // Enable subtle 3D rotation for visual interest
  nailCurvature: number; // 0-1, how curved the nail surface is (0 = flat rectangular, 1 = curved/cylindrical)
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
      color: 0xffb6d9, // Soft pink color for press-on nails
      transparent: true,
      opacity: config.nailOpacity,
      metalness: config.metallicIntensity || 0.4, // Slightly more metallic for nail polish effect
      roughness: config.roughness || 0.3, // Smoother for nail polish shine
    });

    this.metallicMaterial = new THREE.MeshPhysicalMaterial({
      color: 0xffc0e7, // Slightly lighter metallic pink
      transparent: true,
      opacity: config.nailOpacity,
      metalness: config.metallicIntensity || 0.8,
      roughness: config.roughness || 0.15, // Very smooth for high-gloss nail effect
      clearcoat: 1.0,
      clearcoatRoughness: 0.05, // Very smooth clearcoat for nail polish effect
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

    // For press-on nails, we want them to be more prominent and nail-shaped
    // Make them larger and enforce a more realistic nail aspect ratio
    const nailScaleFactor = 1.8; // Make nails 80% larger for better visibility
    const minNailLength = 20; // Minimum length along finger direction
    const minNailWidth = 12; // Minimum width across finger direction

    // IMPORTANT: In nailMatching.ts:
    // - match.nailWidth = length along finger direction (longitudinal)
    // - match.nailHeight = width perpendicular to finger direction (transverse)
    // So we use nailWidth as the length and nailHeight as the width
    let nailLength = Math.max(scaledWidth * nailScaleFactor, minNailLength); // Along finger
    let nailWidth = Math.max(scaledHeight * nailScaleFactor, minNailWidth); // Across finger

    // Enforce press-on nail proportions: length should be 1.5-2x the width
    const idealRatio = 1.7; // Length to width ratio for realistic nail shape
    if (nailLength / nailWidth < idealRatio) {
      nailLength = nailWidth * idealRatio;
    }

    // Make the nail thickness proportional to size but keep it thin like a real nail
    const thickness = Math.max(2, Math.min(4, nailWidth * 0.15)); // Create nail-shaped geometry - use a rounded rectangle that's more nail-like
    let geometry: THREE.BufferGeometry;

    if (this.config.nailCurvature > 0) {
      // Create a nail-shaped geometry with subtle curvature
      // Use a cylinder geometry but make it oval and nail-proportioned
      const segments = 12; // More segments for smoother curves
      geometry = new THREE.CylinderGeometry(
        nailWidth / 2, // radiusTop
        nailWidth / 2.2, // radiusBottom - slightly tapered for nail shape
        thickness, // height (thickness)
        segments, // radialSegments
        1, // heightSegments
        false, // openEnded
        0, // thetaStart
        Math.PI * 2 // Full circle
      );

      // Scale to nail proportions and rotate to lay flat
      geometry.rotateX(Math.PI / 2); // Rotate to lie flat in XY plane
      geometry.scale(nailLength / nailWidth, 1, 1); // Stretch to nail length along X-axis
    } else {
      // Create a rectangular nail shape with length along X-axis
      // This ensures that when we rotate by nailAngle, the length aligns with finger direction
      geometry = new THREE.BoxGeometry(nailLength, nailWidth, thickness);
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
      thickness / 2 // Position at half thickness to center the nail
    );

    // Apply comprehensive 3D nail rotation for realistic positioning
    this.applyNailRotation(mesh, match, key, thickness);

    // Update material color based on confidence - use more nail-appropriate colors
    if (!this.config.showWireframe) {
      const confidence = match.matchConfidence;
      // Use nail polish colors: from red (low confidence) to pink (high confidence)
      const hue = 0.9 + confidence * 0.1; // 0.9 (magenta) to 1.0 (red) - nail polish range
      const saturation = 0.6 + confidence * 0.3; // More saturated for higher confidence
      const lightness = 0.4 + confidence * 0.2; // Brighter for higher confidence
      const color = new THREE.Color().setHSL(hue, saturation, lightness);

      if (
        mesh.material instanceof THREE.MeshStandardMaterial ||
        mesh.material instanceof THREE.MeshPhysicalMaterial
      ) {
        mesh.material.color = color;
      }
    }
  }

  /**
   * Create a rounded box geometry for more realistic nail shapes
   */
  private createRoundedBoxGeometry(
    width: number,
    height: number,
    depth: number,
    radius: number
  ): THREE.BufferGeometry {
    // Clamp radius to prevent issues
    const maxRadius = Math.min(width, height, depth) / 2;
    const clampedRadius = Math.min(radius, maxRadius);

    // For simplicity, create a regular box geometry
    // In a more advanced implementation, you could create actual rounded corners
    // using custom buffer geometry or imported rounded box geometry
    const geometry = new THREE.BoxGeometry(width, height, depth);

    // TODO: For future enhancement, implement actual rounded corners
    // This would involve creating custom buffer geometry with rounded edges
    // For now, the regular box with proper nail proportions will work well

    return geometry;
  }

  /**
   * Apply comprehensive 3D nail rotation for realistic positioning
   * Handles Z-axis alignment, natural finger curvature, and 3D depth rotation
   */
  private applyNailRotation(
    mesh: THREE.Mesh,
    match: NailFingerMatch,
    key: string,
    thickness: number
  ): void {
    // Reset all rotations to start fresh
    mesh.rotation.set(0, 0, 0);

    // 1. Primary Z-axis rotation: Align nail with finger direction
    // The nail angle from nailMatching represents the finger direction in canvas coordinates
    // Since we flip Y coordinates when converting to Three.js, we need to negate the angle
    // to maintain the correct orientation relative to the flipped coordinate system
    const primaryRotationZ = -match.nailAngle;
    mesh.rotation.z = primaryRotationZ;

    // 2. Natural nail curvature based on finger type and position
    const fingerCurvature = this.calculateFingerCurvature(match);

    // 3. Apply X-axis rotation for natural nail tilt toward fingertip
    // Real nails are not completely flat - they curve slightly toward the fingertip
    let tiltX = fingerCurvature.longitudinalTilt;

    // 4. Apply Y-axis rotation for lateral nail curvature
    // Nails curve slightly along their width, especially on curved fingers
    let tiltY = fingerCurvature.lateralTilt;

    // 5. Add dynamic 3D rotation if enabled for visual interest
    if (this.config.enable3DRotation) {
      const time = Date.now() * 0.001;
      // Use finger-specific seeds for consistent but varied animation
      const fingerSeed =
        match.fingertipIndex * 7 + (match.handedness === "Left" ? 0 : 31);

      // Subtle breathing/floating animation - much more natural than previous version
      const breatheX = Math.sin(time * 0.3 + fingerSeed * 0.1) * 0.02; // Very subtle longitudinal sway
      const breatheY = Math.cos(time * 0.4 + fingerSeed * 0.15) * 0.015; // Gentle lateral motion
      const breatheZ = Math.sin(time * 0.25 + fingerSeed * 0.2) * 0.01; // Minor twist variation

      tiltX += breatheX;
      tiltY += breatheY;
      mesh.rotation.z += breatheZ; // Add to primary rotation
    }

    // Apply the calculated rotations
    mesh.rotation.x = tiltX;
    mesh.rotation.y = tiltY;

    // 6. Apply proper rotation order for realistic nail positioning
    // Set the rotation order to ZYX so Z-axis (finger alignment) is applied first,
    // then Y-axis (lateral curvature), then X-axis (longitudinal tilt)
    mesh.rotation.order = "ZYX";

    // Debug: Log rotation information occasionally
    if (Math.random() < 0.05) {
      // Only log occasionally to avoid spam
      console.log(`3D Nail rotation for ${key}:`, {
        fingerType: this.getFingerName(match.fingertipIndex),
        handedness: match.handedness,
        primaryAngle: (primaryRotationZ * 180) / Math.PI,
        longitudinalTilt: (tiltX * 180) / Math.PI,
        lateralTilt: (tiltY * 180) / Math.PI,
        fingerDirection: match.fingerDirection,
        curvature: fingerCurvature,
      });
    }
  }

  /**
   * Calculate natural finger curvature based on finger type and position
   */
  private calculateFingerCurvature(match: NailFingerMatch): {
    longitudinalTilt: number;
    lateralTilt: number;
  } {
    const fingerType = match.fingertipIndex;
    const handedness = match.handedness;

    // Base curvature values for different finger types (in radians)
    let longitudinalTilt = 0;
    let lateralTilt = 0;

    // Different fingers have different natural nail orientations
    switch (fingerType) {
      case 4: // Thumb
        // Thumbs typically angle more toward the palm and have more lateral curvature
        longitudinalTilt = 0.08; // ~4.6 degrees toward palm
        lateralTilt = handedness === "Left" ? -0.06 : 0.06; // Curve toward other fingers
        break;

      case 8: // Index finger
        // Index fingers are relatively straight but tilt slightly toward middle finger
        longitudinalTilt = 0.04; // ~2.3 degrees
        lateralTilt = handedness === "Left" ? 0.02 : -0.02; // Slight inward curve
        break;

      case 12: // Middle finger
        // Middle fingers are most straight, minimal curvature
        longitudinalTilt = 0.02; // ~1.1 degrees
        lateralTilt = 0; // Straight lateral alignment
        break;

      case 16: // Ring finger
        // Ring fingers curve slightly toward pinky
        longitudinalTilt = 0.03; // ~1.7 degrees
        lateralTilt = handedness === "Left" ? 0.025 : -0.025; // Slight outward curve
        break;

      case 20: // Pinky
        // Pinkies have more pronounced curvature and angle
        longitudinalTilt = 0.06; // ~3.4 degrees
        lateralTilt = handedness === "Left" ? 0.04 : -0.04; // More pronounced outward curve
        break;

      default:
        // Default minimal curvature
        longitudinalTilt = 0.02;
        lateralTilt = 0;
    }

    // Apply additional hand orientation adjustments based on viewing angle
    // When hands are viewed from different angles, the natural curvature appears different
    const fingerDirection = match.fingerDirection;
    const handAngle = Math.atan2(fingerDirection[1], fingerDirection[0]);

    // Adjust curvature based on overall hand orientation
    // This makes nails look more natural when hands are rotated
    const handOrientationFactor = Math.cos(handAngle) * 0.02; // Small adjustment
    longitudinalTilt += handOrientationFactor;

    // Apply confidence-based scaling - less confident matches get less extreme rotations
    const confidenceScale = Math.min(1.0, Math.max(0.3, match.matchConfidence));
    longitudinalTilt *= confidenceScale;
    lateralTilt *= confidenceScale;

    // Apply global curvature setting
    const curvatureMultiplier = this.config.nailCurvature;
    longitudinalTilt *= curvatureMultiplier;
    lateralTilt *= curvatureMultiplier;

    return {
      longitudinalTilt,
      lateralTilt,
    };
  }

  /**
   * Get human-readable finger name for debugging
   */
  private getFingerName(fingertipIndex: number): string {
    const names: Record<number, string> = {
      4: "Thumb",
      8: "Index",
      12: "Middle",
      16: "Ring",
      20: "Pinky",
    };
    return names[fingertipIndex] || "Unknown";
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
   * Update nail thickness - note this affects new nails, existing ones update on next overlay update
   */
  public setNailThickness(thickness: number): void {
    this.config.nailThickness = Math.max(1, Math.min(8, thickness)); // Clamp for press-on nail realism
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
    // Note: We don't reset rotations here because static natural curvature should remain
    // The 3D animation will simply stop adding dynamic movement on the next update cycle
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
   * Force refresh of all nail rotations - useful when rotation settings change
   */
  public refreshNailRotations(): void {
    // This will be called automatically on the next updateNailOverlays call
    // since rotations are recalculated every frame
    this.render();
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
