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

  // New enhanced features
  enableDynamicLighting?: boolean; // Enable animated lighting effects
  enableEnvironmentMapping?: boolean; // Enable realistic reflections
  nailVariety?: boolean; // Enable different materials per finger
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
  private animationId: number | null = null;

  // Materials for the nail prisms
  private nailMaterial!: THREE.MeshStandardMaterial;
  private wireframeMaterial!: THREE.MeshBasicMaterial;
  private metallicMaterial!: THREE.MeshPhysicalMaterial;

  // Additional specialized materials for variety
  private glossyMaterial!: THREE.MeshPhysicalMaterial;
  private matteMaterial!: THREE.MeshStandardMaterial;
  private holographicMaterial!: THREE.MeshPhysicalMaterial;

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
      metalness: config.metallicIntensity || 0.4,
      roughness: config.roughness || 0.3,
      envMapIntensity: 1.5, // Enhanced environment mapping
    });

    this.metallicMaterial = new THREE.MeshPhysicalMaterial({
      color: 0xffc0e7, // Slightly lighter metallic pink
      transparent: true,
      opacity: config.nailOpacity,
      metalness: config.metallicIntensity || 0.8,
      roughness: config.roughness || 0.15,
      clearcoat: 1.0,
      clearcoatRoughness: 0.05,
      reflectivity: config.enableReflections ? 0.9 : 0.3,
      envMapIntensity: 2.0, // Strong environment reflections
      sheen: 0.5, // Add subtle sheen for nail polish effect
      sheenRoughness: 0.2,
    });

    this.wireframeMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      wireframe: true,
      transparent: true,
      opacity: 0.8,
    });

    // Create additional specialized materials
    this.createSpecializedMaterials();

    // Set up lighting
    this.setupLighting();

    // Append renderer to container
    this.containerElement.appendChild(this.renderer.domElement);
    this.isInitialized = true;

    console.log("3D nail overlay initialized successfully");
  }

  private setupLighting(): void {
    // Enhanced ambient light for better base illumination
    this.ambientLight = new THREE.AmbientLight(0x404040, 0.4);
    this.scene.add(this.ambientLight);

    // Main directional light with improved positioning for nail highlights
    this.directionalLight = new THREE.DirectionalLight(0xffffff, 1.2);
    this.directionalLight.position.set(150, 150, 300);
    this.directionalLight.castShadow = true;

    // Enhanced shadow properties for better definition
    this.directionalLight.shadow.mapSize.width = 4096;
    this.directionalLight.shadow.mapSize.height = 4096;
    this.directionalLight.shadow.camera.near = 0.5;
    this.directionalLight.shadow.camera.far = 800;
    this.directionalLight.shadow.camera.left = -300;
    this.directionalLight.shadow.camera.right = 300;
    this.directionalLight.shadow.camera.top = 300;
    this.directionalLight.shadow.camera.bottom = -300;
    this.directionalLight.shadow.bias = -0.0001;

    this.scene.add(this.directionalLight);

    // Enhanced point light for metallic reflections and highlights
    this.pointLight = new THREE.PointLight(0xffffff, 1.0, 500);
    this.pointLight.position.set(-80, 80, 200);
    this.pointLight.castShadow = true;
    this.pointLight.shadow.mapSize.width = 2048;
    this.pointLight.shadow.mapSize.height = 2048;
    this.scene.add(this.pointLight);

    // Add multiple rim lights for better 3D definition
    const rimLight1 = new THREE.DirectionalLight(0x8888ff, 0.6);
    rimLight1.position.set(-150, -80, 150);
    this.scene.add(rimLight1);

    const rimLight2 = new THREE.DirectionalLight(0xff8888, 0.4);
    rimLight2.position.set(150, -80, 150);
    this.scene.add(rimLight2);

    // Add a fill light to reduce harsh shadows
    const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
    fillLight.position.set(0, -100, 100);
    this.scene.add(fillLight);

    // Add dynamic lighting that follows the camera perspective
    this.setupDynamicLighting();
  }

  /**
   * Setup dynamic lighting that enhances the 3D effect
   */
  private setupDynamicLighting(): void {
    // Create a light that simulates screen reflection
    const screenLight = new THREE.PointLight(0xffffff, 0.5, 400);
    screenLight.position.set(0, 0, 250);
    this.scene.add(screenLight);

    // Add subtle colored accent lights for visual interest
    const accentLight1 = new THREE.PointLight(0xff99dd, 0.3, 300);
    accentLight1.position.set(100, 100, 100);
    this.scene.add(accentLight1);

    const accentLight2 = new THREE.PointLight(0x99ddff, 0.3, 300);
    accentLight2.position.set(-100, -100, 100);
    this.scene.add(accentLight2);

    // Create environment map for realistic reflections
    this.createEnvironmentMap();
  }

  /**
   * Create a simple environment map for realistic reflections
   */
  private createEnvironmentMap(): void {
    // Create a simple gradient environment map
    const pmremGenerator = new THREE.PMREMGenerator(this.renderer);

    // Create a simple sky-like environment
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);

    // Add some geometry to create interesting reflections
    const geometry = new THREE.SphereGeometry(500, 32, 32);
    const material = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      side: THREE.BackSide,
    });
    const sphere = new THREE.Mesh(geometry, material);
    scene.add(sphere);

    const envMap = pmremGenerator.fromScene(scene).texture;

    // Apply environment map to materials
    this.nailMaterial.envMap = envMap;
    this.metallicMaterial.envMap = envMap;
    this.glossyMaterial.envMap = envMap;
    this.holographicMaterial.envMap = envMap;

    pmremGenerator.dispose();
  }

  /**
   * Create additional specialized materials for enhanced nail variety
   */
  private createSpecializedMaterials(): void {
    // High-gloss material for super shiny nails
    this.glossyMaterial = new THREE.MeshPhysicalMaterial({
      color: 0xffd6e8,
      transparent: true,
      opacity: this.config.nailOpacity,
      metalness: 0.1,
      roughness: 0.05,
      clearcoat: 1.0,
      clearcoatRoughness: 0.01,
      reflectivity: 1.0,
      envMapIntensity: 3.0,
      sheen: 1.0,
      sheenRoughness: 0.1,
    });

    // Matte material for subtle, natural look
    this.matteMaterial = new THREE.MeshStandardMaterial({
      color: 0xf5c6d6,
      transparent: true,
      opacity: this.config.nailOpacity,
      metalness: 0.0,
      roughness: 0.8,
      envMapIntensity: 0.3,
    });

    // Holographic/iridescent material for special effects
    this.holographicMaterial = new THREE.MeshPhysicalMaterial({
      color: 0xe6b3ff,
      transparent: true,
      opacity: this.config.nailOpacity,
      metalness: 0.9,
      roughness: 0.1,
      clearcoat: 1.0,
      clearcoatRoughness: 0.02,
      reflectivity: 1.0,
      envMapIntensity: 5.0,
      sheen: 1.0,
      sheenRoughness: 0.05,
      iridescence: 1.0,
      iridescenceIOR: 2.0,
      iridescenceThicknessRange: [100, 800],
    });
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

    // Start animation loop if 3D rotation is enabled
    if (this.config.enable3DRotation && !this.animationId) {
      this.startAnimationLoop();
    } else if (!this.config.enable3DRotation && this.animationId) {
      this.stopAnimationLoop();
    }
  }

  /**
   * Start the animation loop for dynamic effects
   */
  private startAnimationLoop(): void {
    if (this.animationId) return;

    const animate = () => {
      if (!this.isInitialized || !this.config.enable3DRotation) {
        this.animationId = null;
        return;
      }

      // Animate lighting for dynamic effect
      this.animateLighting();

      // Re-render the scene
      this.render();

      this.animationId = requestAnimationFrame(animate);
    };

    this.animationId = requestAnimationFrame(animate);
  }

  /**
   * Stop the animation loop
   */
  private stopAnimationLoop(): void {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  /**
   * Animate lighting for dynamic visual effects
   */
  private animateLighting(): void {
    const time = Date.now() * 0.001;

    // Subtle breathing effect on ambient light
    this.ambientLight.intensity = 0.4 + Math.sin(time * 0.5) * 0.1;

    // Gentle movement of the main point light
    this.pointLight.position.x = -80 + Math.sin(time * 0.3) * 20;
    this.pointLight.position.y = 80 + Math.cos(time * 0.4) * 15;

    // Subtle color temperature variation
    const colorTemp = 0.5 + Math.sin(time * 0.2) * 0.1;
    this.directionalLight.color.setHSL(0, 0, 1 - colorTemp * 0.1);
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

    // FIXED: Account for foreshortening when nails are viewed at steep angles
    // When fingers point down (claw position), nails appear smaller due to perspective
    const [dx, dy, dz] = match.fingerDirection;
    const viewAngle = Math.abs(Math.atan2(dy, Math.sqrt(dx * dx + dz * dz)));
    const foreshortening = Math.cos(viewAngle);

    // Apply perspective correction to prevent nails from becoming too small
    const perspectiveScale = Math.max(0.7, foreshortening + 0.3); // Minimum 70% of original size
    nailLength *= perspectiveScale;
    nailWidth *= perspectiveScale;

    // Make the nail thickness proportional to size but keep it thin like a real nail
    const thickness = Math.max(2, Math.min(6, nailWidth * 0.2));

    // Create anatomically accurate nail-shaped geometry
    let geometry: THREE.BufferGeometry;

    if (this.config.nailCurvature > 0.1) {
      // Create a realistic nail shape with proper curvature
      geometry = this.createAnatomicalNailGeometry(
        nailLength,
        nailWidth,
        thickness
      );
    } else {
      // Create a more refined rounded rectangle for flat nails
      geometry = this.createRoundedNailGeometry(
        nailLength,
        nailWidth,
        thickness
      );
    }

    let mesh = this.nailMeshes.get(key);

    if (!mesh) {
      // Create new mesh - choose material based on config and finger characteristics
      let material: THREE.Material;
      if (this.config.showWireframe) {
        material = this.wireframeMaterial;
      } else {
        material = this.selectMaterialForFinger(match);
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

      // Update material if not in wireframe mode
      if (!this.config.showWireframe) {
        mesh.material = this.selectMaterialForFinger(match);
      }
    }

    // Position the mesh at the nail centroid with improved depth positioning
    const scaledCentroid = [
      match.nailCentroid[0] * scaleX,
      match.nailCentroid[1] * scaleY,
    ];

    // Convert from canvas coordinates to Three.js coordinates
    // Canvas: (0,0) at top-left, X right, Y down
    // Three.js: (0,0) at center, X right, Y up
    const threeX = scaledCentroid[0] - this.config.canvasWidth / 2;
    const threeY = -(scaledCentroid[1] - this.config.canvasHeight / 2); // Flip Y and center

    // Calculate a more realistic Z position based on finger type and perspective
    const baseZ = thickness / 2;
    const fingerTypeZ = this.calculateFingerDepth(match);
    const perspectiveZ = this.calculatePerspectiveDepth(match, scaleX, scaleY);

    // Add extra depth based on rotation to make nails appear to "sit" on fingertips
    const rotationDepthOffset = this.calculateRotationDepthOffset(match);

    mesh.position.set(
      threeX,
      threeY,
      baseZ + fingerTypeZ + perspectiveZ + rotationDepthOffset
    );

    // Apply comprehensive 3D nail rotation for realistic positioning
    this.applyNailRotation(mesh, match, key, thickness);

    // Update material color based on confidence and finger characteristics
    this.updateMaterialColor(mesh, match);
  }

  /**
   * Select the most appropriate material for a specific finger based on characteristics
   */
  private selectMaterialForFinger(match: NailFingerMatch): THREE.Material {
    const confidence = match.matchConfidence;
    const fingerType = match.fingertipIndex;

    // High confidence fingers get special materials
    if (confidence > 0.8) {
      // Thumb and index finger often have more prominent/glossy nails
      if (fingerType === 4 || fingerType === 8) {
        return this.config.metallicIntensity > 0.7
          ? this.holographicMaterial
          : this.glossyMaterial;
      }
      return this.config.metallicIntensity > 0.6
        ? this.metallicMaterial
        : this.glossyMaterial;
    }

    // Medium confidence gets standard materials
    if (confidence > 0.5) {
      return this.config.metallicIntensity > 0.6
        ? this.metallicMaterial
        : this.nailMaterial;
    }

    // Lower confidence gets matte material for subtlety
    return this.matteMaterial;
  }

  /**
   * Update material color based on confidence and finger characteristics
   */
  private updateMaterialColor(mesh: THREE.Mesh, match: NailFingerMatch): void {
    if (this.config.showWireframe) return;

    const confidence = match.matchConfidence;
    const fingerType = match.fingertipIndex;

    // Create finger-specific color variations
    let baseHue = 0.9; // Default magenta base

    // Different fingers get slightly different base colors for variety
    switch (fingerType) {
      case 4: // Thumb - slightly more red
        baseHue = 0.95;
        break;
      case 8: // Index - standard pink
        baseHue = 0.9;
        break;
      case 12: // Middle - slightly more purple
        baseHue = 0.85;
        break;
      case 16: // Ring - warmer pink
        baseHue = 0.92;
        break;
      case 20: // Pinky - cooler pink
        baseHue = 0.88;
        break;
    }

    // Adjust hue based on confidence
    const hue = baseHue + confidence * 0.05;
    const saturation = 0.5 + confidence * 0.4; // More saturated for higher confidence
    const lightness = 0.35 + confidence * 0.25; // Brighter for higher confidence

    const color = new THREE.Color().setHSL(hue, saturation, lightness);

    // Apply color to the material
    if (
      mesh.material instanceof THREE.MeshStandardMaterial ||
      mesh.material instanceof THREE.MeshPhysicalMaterial
    ) {
      mesh.material.color = color;
    }
  }

  /**
   * Create anatomically accurate nail geometry with natural curvature
   */
  private createAnatomicalNailGeometry(
    length: number,
    width: number,
    thickness: number
  ): THREE.BufferGeometry {
    // Create a nail shape that mimics the natural curve of human nails
    // This uses a combination of sphere and cylinder geometries

    const segments = 16; // High detail for smooth curves
    const geometry = new THREE.SphereGeometry(1, segments, segments / 2);

    // Scale to nail proportions
    geometry.scale(length / 2, width / 2, thickness / 4);

    // Flatten the top to create a nail-like surface
    const positions = geometry.attributes.position;
    for (let i = 0; i < positions.count; i++) {
      const z = positions.getZ(i);
      if (z > 0) {
        // Flatten the top surface while maintaining slight curvature
        positions.setZ(i, Math.min(z, thickness / 6));
      }
    }

    positions.needsUpdate = true;
    geometry.computeVertexNormals();

    return geometry;
  }

  /**
   * Create a refined rounded nail geometry for more realistic appearance
   */
  private createRoundedNailGeometry(
    length: number,
    width: number,
    thickness: number
  ): THREE.BufferGeometry {
    // Create a rounded rectangle that looks more like a real nail
    const segments = 12;

    // Start with a cylinder and modify it
    const geometry = new THREE.CylinderGeometry(
      width / 2,
      width / 2.1, // Slight taper
      thickness,
      segments,
      1,
      false
    );

    // Rotate to lay flat and scale to nail proportions
    geometry.rotateX(Math.PI / 2);
    geometry.scale(length / width, 1, 1);

    // Add subtle roundedness to the corners
    const positions = geometry.attributes.position;
    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i);
      const y = positions.getY(i);

      // Round the corners slightly
      if (Math.abs(x) > length * 0.35) {
        const factor = 0.95;
        positions.setY(i, y * factor);
      }
    }

    positions.needsUpdate = true;
    geometry.computeVertexNormals();

    return geometry;
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
   * Simplified rotation application with better visual results
   */
  private applyNailRotation(
    mesh: THREE.Mesh,
    match: NailFingerMatch,
    key: string,
    thickness: number
  ): void {
    // Reset all rotations to start fresh
    mesh.rotation.set(0, 0, 0);

    // Calculate the 3D orientation with improved algorithms
    const rotation = this.calculate3DNailOrientation(match);

    // Apply rotations in the proper order for realistic nail positioning
    mesh.rotation.x = rotation.tiltX; // Forward/backward tilt (the key improvement!)
    mesh.rotation.y = rotation.tiltY; // Left/right tilt
    mesh.rotation.z = rotation.tiltZ; // In-plane rotation (finger direction alignment)

    // Set rotation order to XYZ for most natural nail orientation
    mesh.rotation.order = "XYZ";

    // Add subtle dynamic 3D rotation if enabled for visual interest
    if (this.config.enable3DRotation) {
      const time = Date.now() * 0.001;
      const fingerSeed =
        match.fingertipIndex * 7 + (match.handedness === "Left" ? 0 : 31);

      // Subtle breathing effects that enhance the 3D appearance
      const breatheX = Math.sin(time * 0.3 + fingerSeed * 0.1) * 0.08; // Gentle forward/back motion
      const breatheY = Math.cos(time * 0.4 + fingerSeed * 0.15) * 0.05; // Subtle side-to-side
      const breatheZ = Math.sin(time * 0.25 + fingerSeed * 0.2) * 0.03; // Minimal rotation

      mesh.rotation.x += breatheX;
      mesh.rotation.y += breatheY;
      mesh.rotation.z += breatheZ;
    }

    // Debug output for troubleshooting (reduced frequency)
    if (Math.random() < 0.02) {
      console.log(`🎯 Applied Rotation for ${key}:`, {
        fingerType: this.getFingerName(match.fingertipIndex),
        tiltX_deg: (mesh.rotation.x * 180) / Math.PI,
        tiltY_deg: (mesh.rotation.y * 180) / Math.PI,
        tiltZ_deg: (mesh.rotation.z * 180) / Math.PI,
        confidence: match.matchConfidence,
      });
    }
  }

  /**
   * Calculate complete 3D nail orientation based on finger position and anatomy
   * Simplified and more reliable approach
   */
  private calculate3DNailOrientation(match: NailFingerMatch): {
    tiltX: number; // Forward/backward tilt (pitch)
    tiltY: number; // Left/right tilt (roll)
    tiltZ: number; // In-plane rotation (yaw)
  } {
    const confidence = match.matchConfidence;

    // Calculate each rotation component independently for better control
    let tiltZ = this.calculateImprovedFingerAlignment(match);
    let tiltX = this.calculateForwardBackwardTilt(match);
    let tiltY = this.calculateLeftRightTilt(match);

    // Apply confidence scaling - less confident matches get more conservative rotations
    const confidenceScale = Math.pow(
      Math.min(1.0, Math.max(0.3, confidence)),
      0.5
    );
    tiltX *= confidenceScale;
    tiltY *= confidenceScale;
    // Don't scale tiltZ as much since finger direction is usually reliable

    // Apply global curvature multiplier for user control
    const curvatureMultiplier = Math.max(0.1, this.config.nailCurvature);
    tiltX *= curvatureMultiplier;
    tiltY *= curvatureMultiplier;

    // Debug the final calculations occasionally
    if (Math.random() < 0.05) {
      console.log(
        `🎯 Final 3D Orientation for ${this.getFingerName(
          match.fingertipIndex
        )}:`,
        {
          fingerType: this.getFingerName(match.fingertipIndex),
          handedness: match.handedness,
          confidence: confidence,
          confidenceScale: confidenceScale,
          tiltX_degrees: (tiltX * 180) / Math.PI,
          tiltY_degrees: (tiltY * 180) / Math.PI,
          tiltZ_degrees: (tiltZ * 180) / Math.PI,
        }
      );
    }

    return { tiltX, tiltY, tiltZ };
  }

  /**
   * Calculate improved finger alignment accounting for coordinate system issues
   * Fixed to handle all finger orientations properly
   */
  private calculateImprovedFingerAlignment(match: NailFingerMatch): number {
    // Now we have true 3D finger direction!
    const [dx, dy, dz] = match.fingerDirection;

    // FIXED: Use the actual 3D finger direction for more accurate alignment
    // Project the 3D direction onto the screen plane, considering all orientations
    let fingerAngle = Math.atan2(dy, dx);

    // CRITICAL FIX: Don't flip Y coordinate here - let the natural direction be preserved
    // The previous -fingerAngle was causing issues with downward-pointing fingers

    // Add confidence-based adjustments for low-confidence matches
    const confidence = match.matchConfidence;
    if (confidence < 0.6) {
      // For very low confidence, blend with anatomical expectations
      const anatomicalAngle = this.calculateAnatomicalExpectedAngle(match);
      const blendFactor = (0.6 - confidence) * 2.0; // Stronger blending for lower confidence
      fingerAngle =
        fingerAngle * (1 - blendFactor) + anatomicalAngle * blendFactor;
    }

    // Debug the improved calculation
    if (Math.random() < 0.05) {
      console.log(
        `🧭 Fixed Finger Alignment for ${this.getFingerName(
          match.fingertipIndex
        )}:`,
        {
          fingerDirection3D: match.fingerDirection,
          fingerAngle_degrees: (fingerAngle * 180) / Math.PI,
          confidence: confidence,
          dy_component: dy, // This should show negative values for downward fingers
        }
      );
    }

    return fingerAngle;
  }
  /**
   * Calculate anatomically expected angle for finger alignment
   * Simplified to focus on basic anatomical positions
   */
  private calculateAnatomicalExpectedAngle(match: NailFingerMatch): number {
    const fingerType = match.fingertipIndex;
    const handedness = match.handedness;

    // Simplified anatomical expectations based on typical finger orientations
    switch (fingerType) {
      case 4: // Thumb
        return handedness === "Left" ? Math.PI * 0.25 : -Math.PI * 0.25; // ±45°
      case 8: // Index finger
        return Math.PI * 0.5; // 90° (typically pointing up)
      case 12: // Middle finger
        return Math.PI * 0.5; // 90° (typically pointing up)
      case 16: // Ring finger
        return Math.PI * 0.5; // 90° (typically pointing up)
      case 20: // Pinky
        return handedness === "Left" ? Math.PI * 0.75 : -Math.PI * 0.75; // ±135°
      default:
        return Math.PI * 0.5; // Default to pointing up
    }
  }

  /**
   * Calculate forward/backward tilt based on finger anatomy and 3D direction
   * Fixed to handle extreme angles better and prevent over-rotation
   */
  private calculateForwardBackwardTilt(match: NailFingerMatch): number {
    const fingerType = match.fingertipIndex;

    // Extract the Z component from the 3D finger direction
    const [dx, dy, dz] = match.fingerDirection;

    // FIXED: More conservative depth tilt calculation
    // Use atan instead of atan2 for more stable results
    const xyMagnitude = Math.sqrt(dx * dx + dy * dy);
    let depthTilt = 0;

    if (xyMagnitude > 0.01) {
      // Avoid division by very small numbers
      depthTilt = Math.atan(dz / xyMagnitude);
    }

    // FIXED: Less aggressive scaling to prevent extreme rotations
    depthTilt *= 1.5; // Reduced from 3.0 to 1.5 for more natural appearance

    // Anatomical base tilts - nails naturally tilt based on finger curvature
    let anatomicalTilt = 0;
    switch (fingerType) {
      case 4: // Thumb - significant natural curve
        anatomicalTilt = 0.3; // Reduced from 0.4 (~17 degrees)
        break;
      case 8: // Index finger - moderate curve
        anatomicalTilt = 0.2; // Reduced from 0.25 (~11 degrees)
        break;
      case 12: // Middle finger - slight curve
        anatomicalTilt = 0.15; // Kept same (~9 degrees)
        break;
      case 16: // Ring finger - moderate curve
        anatomicalTilt = 0.18; // Slightly reduced (~10 degrees)
        break;
      case 20: // Pinky - significant curve
        anatomicalTilt = 0.25; // Reduced from 0.35 (~14 degrees)
        break;
    }

    // Combine depth-based tilt with anatomical expectations
    const totalTilt = anatomicalTilt + depthTilt;

    // FIXED: More restrictive clamping to prevent extreme rotations
    const clampedTilt = Math.max(
      -Math.PI / 4,
      Math.min(Math.PI / 3, totalTilt)
    ); // -45° to +60°

    // Debug the improved calculation
    if (Math.random() < 0.05) {
      console.log(
        `🎯 Fixed Depth Tilt for ${this.getFingerName(match.fingertipIndex)}:`,
        {
          fingerDirection3D: match.fingerDirection,
          dz_component: dz,
          xyMagnitude: xyMagnitude,
          depthTilt_degrees: (depthTilt * 180) / Math.PI,
          anatomicalTilt_degrees: (anatomicalTilt * 180) / Math.PI,
          totalTilt_degrees: (totalTilt * 180) / Math.PI,
          clampedTilt_degrees: (clampedTilt * 180) / Math.PI,
        }
      );
    }

    return clampedTilt;
  }

  /**
   * Calculate depth offset based on finger type for more realistic positioning
   * Simplified to focus on basic finger anatomy
   */
  private calculateFingerDepth(match: NailFingerMatch): number {
    const fingerType = match.fingertipIndex;

    // Different fingers naturally appear at different depths
    switch (fingerType) {
      case 4: // Thumb
        return 6; // Thumbs stick out moderately
      case 8: // Index finger
        return 4; // Index is prominent
      case 12: // Middle finger
        return 5; // Middle finger is longest
      case 16: // Ring finger
        return 3; // Ring finger is slightly recessed
      case 20: // Pinky
        return 2; // Pinky is shortest
      default:
        return 3;
    }
  }

  /**
   * Calculate left/right tilt based on 3D finger direction and hand anatomy
   * Fixed to handle extreme angles better
   */
  private calculateLeftRightTilt(match: NailFingerMatch): number {
    const fingerType = match.fingertipIndex;
    const handedness = match.handedness;
    const [dx, dy, dz] = match.fingerDirection;

    // FIXED: More stable lateral tilt calculation
    // Use the 3D finger direction but be more conservative
    const yzMagnitude = Math.sqrt(dy * dy + dz * dz);
    let lateralTilt = 0;

    if (yzMagnitude > 0.01) {
      // Avoid division by very small numbers
      lateralTilt = Math.atan(dx / yzMagnitude);
    }

    // FIXED: Less aggressive scaling for more natural appearance
    lateralTilt *= 1.0; // Reduced from 1.5 to 1.0

    // Add anatomical corrections based on finger type and handedness
    let anatomicalCorrection = 0;
    switch (fingerType) {
      case 4: // Thumb - significant lateral angle
        anatomicalCorrection = handedness === "Left" ? -0.1 : 0.1; // Reduced from ±0.15
        break;
      case 8: // Index finger - slight lateral angle
        anatomicalCorrection = handedness === "Left" ? 0.03 : -0.03; // Reduced from ±0.05
        break;
      case 12: // Middle finger - most neutral
        anatomicalCorrection = 0;
        break;
      case 16: // Ring finger - slight lateral angle
        anatomicalCorrection = handedness === "Left" ? 0.03 : -0.03; // Reduced from ±0.05
        break;
      case 20: // Pinky - moderate lateral angle
        anatomicalCorrection = handedness === "Left" ? 0.07 : -0.07; // Reduced from ±0.1
        break;
    }

    const totalTilt = lateralTilt + anatomicalCorrection;

    // FIXED: More conservative clamping
    return Math.max(-Math.PI / 6, Math.min(Math.PI / 6, totalTilt)); // Max ±30 degrees (reduced from ±45)
  }

  /**
   * Calculate perspective-based depth for more realistic 3D appearance
   */
  private calculatePerspectiveDepth(
    match: NailFingerMatch,
    scaleX: number,
    scaleY: number
  ): number {
    // Calculate distance from center for perspective effect
    const centerX = this.config.canvasWidth / 2;
    const centerY = this.config.canvasHeight / 2;

    const scaledCentroid = [
      match.nailCentroid[0] * scaleX,
      match.nailCentroid[1] * scaleY,
    ];

    const distanceFromCenter = Math.sqrt(
      Math.pow(scaledCentroid[0] - centerX, 2) +
        Math.pow(scaledCentroid[1] - centerY, 2)
    );

    // Normalize distance and apply subtle perspective depth
    const maxDistance = Math.sqrt(centerX * centerX + centerY * centerY);
    const normalizedDistance = distanceFromCenter / maxDistance;

    // Fingers further from center appear slightly more recessed
    return -normalizedDistance * 5;
  }

  /**
   * Calculate additional depth offset based on nail rotation for more realistic positioning
   */
  private calculateRotationDepthOffset(match: NailFingerMatch): number {
    // Calculate how much the nail should be offset in Z based on its tilt
    // This makes nails appear to properly "sit" on the fingertips

    const fingerType = match.fingertipIndex;

    // More tilted fingers (like thumb and pinky) should have more depth variation
    let rotationFactor = 0;

    switch (fingerType) {
      case 4: // Thumb - significant rotation offset
        rotationFactor = 8;
        break;
      case 8: // Index finger
        rotationFactor = 4;
        break;
      case 12: // Middle finger - most neutral
        rotationFactor = 2;
        break;
      case 16: // Ring finger
        rotationFactor = 5;
        break;
      case 20: // Pinky - significant rotation offset
        rotationFactor = 7;
        break;
    }

    // Calculate distance from center for additional perspective effect
    const centerX = this.config.canvasWidth / 2;
    const centerY = this.config.canvasHeight / 2;
    const nailX = match.nailCentroid[0];
    const nailY = match.nailCentroid[1];

    const distanceFromCenter = Math.sqrt(
      Math.pow(nailX - centerX, 2) + Math.pow(nailY - centerY, 2)
    );
    const maxDistance = Math.sqrt(centerX * centerX + centerY * centerY);
    const normalizedDistance = distanceFromCenter / maxDistance;

    // Combine rotation factor with distance for realistic depth
    return rotationFactor * (1 + normalizedDistance * 0.5);
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

    // Stop animation loop
    this.stopAnimationLoop();

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
    this.glossyMaterial.dispose();
    this.matteMaterial.dispose();
    this.holographicMaterial.dispose();

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

    if (enabled) {
      this.startAnimationLoop();
    } else {
      this.stopAnimationLoop();
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

  /**
   * Get performance and rendering information
   */
  public getPerformanceInfo(): {
    triangles: number;
    geometries: number;
    textures: number;
    materials: number;
    meshCount: number;
  } {
    const renderer = this.renderer;
    const info = renderer.info;

    return {
      triangles: info.render.triangles,
      geometries: info.memory.geometries,
      textures: info.memory.textures,
      materials: info.programs?.length || 0,
      meshCount: this.nailMeshes.size,
    };
  }

  /**
   * Optimize rendering settings based on performance
   */
  public optimizeForPerformance(enableOptimizations: boolean = true): void {
    if (enableOptimizations) {
      // Reduce shadow map size for better performance
      this.directionalLight.shadow.mapSize.width = 2048;
      this.directionalLight.shadow.mapSize.height = 2048;
      this.pointLight.shadow.mapSize.width = 1024;
      this.pointLight.shadow.mapSize.height = 1024;

      // Reduce material quality slightly
      this.metallicMaterial.envMapIntensity = 1.0;
      this.glossyMaterial.envMapIntensity = 1.5;
      this.holographicMaterial.envMapIntensity = 2.0;
    } else {
      // Restore high-quality settings
      this.directionalLight.shadow.mapSize.width = 4096;
      this.directionalLight.shadow.mapSize.height = 4096;
      this.pointLight.shadow.mapSize.width = 2048;
      this.pointLight.shadow.mapSize.height = 2048;

      // Restore full material quality
      this.metallicMaterial.envMapIntensity = 2.0;
      this.glossyMaterial.envMapIntensity = 3.0;
      this.holographicMaterial.envMapIntensity = 5.0;
    }

    this.render();
  }
}
