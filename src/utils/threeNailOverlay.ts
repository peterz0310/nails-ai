/**
 * Three.js 3D Nail Overlay System (REVIexport class ThreeNailOverlay {
  private scene: THREE.Scene;
  private camera: THREE.OrthographicCamera;
  private renderer: THREE.WebGLRenderer;
  private nailMeshes: Map<string, THREE.Mesh> = new Map();
  private config: ThreeNailOverlayConfig;
  private containerElement: HTMLElement;
  private nailMaterial!: THREE.MeshStandardMaterial;
  private ambientLight!: THREE.AmbientLight;
  private dirLight!: THREE.DirectionalLight;
  private textureLoader!: THREE.TextureLoader;
  private nailTexture: THREE.Texture | null = null;ED)
 *
 * This module provides 3D nail overlay functionality using Three.js.
 *
 * IMPROVED Features:
 * - Uses ExtrudeGeometry to create true 3D nails with thickness and curvature.
 * - Supports both full 3D orientation and simple 2D rotation via a config flag.
 * - Uses a full 3D orientation basis (from nailMatching.ts) for robust rotation.
 * - Applies rotation using Quaternions to avoid gimbal lock and instability.
 * - Nail geometry is procedurally generated and cached to optimize performance.
 * - Centralized `updateConfig` method for cleaner state management.
 */

import * as THREE from "three";
import { NailFingerMatch } from "./nailMatching";

export interface ThreeNailOverlayConfig {
  canvasWidth: number;
  canvasHeight: number;
  enableLighting: boolean;
  nailThickness: number;
  nailOpacity: number;
  showWireframe: boolean;
  metallicIntensity: number;
  roughness: number;
  enable3DRotation: boolean;
  nailCurvature: number; // 0=flat, 1=very curved
  nailColor: { r: number; g: number; b: number };
  // Image texture support
  nailTexture?: string | null; // Base64 or URL of uploaded image
  textureOpacity?: number; // Opacity of the texture overlay
}

export class ThreeNailOverlay {
  private scene: THREE.Scene;
  private camera: THREE.OrthographicCamera;
  private renderer: THREE.WebGLRenderer;
  private nailMeshes: Map<string, THREE.Mesh> = new Map();
  private textureMeshes: Map<string, THREE.Mesh> = new Map(); // Separate meshes for textures
  private config: ThreeNailOverlayConfig;
  private containerElement: HTMLElement;
  private nailMaterial!: THREE.MeshStandardMaterial;
  private textureMaterial!: THREE.MeshStandardMaterial; // Separate material for textures
  private ambientLight!: THREE.AmbientLight;
  private dirLight!: THREE.DirectionalLight;
  private textureLoader!: THREE.TextureLoader;
  private nailTexture: THREE.Texture | null = null;

  constructor(containerElement: HTMLElement, config: ThreeNailOverlayConfig) {
    this.containerElement = containerElement;
    this.config = { ...config };
    this.scene = new THREE.Scene();

    // Initialize texture loader
    this.textureLoader = new THREE.TextureLoader();

    const { canvasWidth, canvasHeight } = config;
    // Orthographic camera matches the 2D canvas, with Y-axis pointing up.
    this.camera = new THREE.OrthographicCamera(
      -canvasWidth / 2,
      canvasWidth / 2,
      canvasHeight / 2,
      -canvasHeight / 2,
      1,
      2000
    );
    this.camera.position.z = 1000;

    this.renderer = new THREE.WebGLRenderer({
      alpha: true,
      antialias: true,
    });
    this.renderer.setSize(canvasWidth, canvasHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.domElement.style.position = "absolute";
    this.renderer.domElement.style.top = "0";
    this.renderer.domElement.style.left = "0";
    this.renderer.domElement.style.pointerEvents = "none";
    this.containerElement.appendChild(this.renderer.domElement);

    this.setupLighting();
    this.createMaterials();
    this.updateConfig(this.config); // Set initial config state

    console.log("✅ 3D nail overlay initialized with extruded geometry logic.");
  }

  private setupLighting(): void {
    this.ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    this.scene.add(this.ambientLight);

    this.dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    this.dirLight.position.set(50, 100, 150);
    this.scene.add(this.dirLight);
  }

  private createMaterials(): void {
    // Single material that can display both color and texture
    this.nailMaterial = new THREE.MeshStandardMaterial({
      color: 0xff6b9d, // Default pink
      transparent: true,
      side: THREE.DoubleSide,
      // Ensure texture blending works properly
      alphaTest: 0.01, // Helps with texture transparency
    });

    // Keep texture material for backwards compatibility but not used anymore
    this.textureMaterial = new THREE.MeshStandardMaterial({
      transparent: true,
      side: THREE.DoubleSide,
      opacity: 0.9,
      alphaTest: 0.1,
    });

    // Load initial texture if provided
    if (this.config.nailTexture) {
      this.loadTexture(this.config.nailTexture);
    }
  }

  public updateConfig(newConfig: ThreeNailOverlayConfig) {
    const oldConfig = this.config;
    this.config = { ...newConfig };

    // Update lighting
    this.ambientLight.visible = this.config.enableLighting;
    this.dirLight.visible = this.config.enableLighting;

    // Update material properties on all existing meshes
    this.nailMaterial.opacity = this.config.nailOpacity;
    this.nailMaterial.metalness = this.config.metallicIntensity;
    this.nailMaterial.roughness = this.config.roughness;
    this.nailMaterial.wireframe = this.config.showWireframe;

    // Only update color if there's no texture (texture should override color)
    if (!this.nailTexture) {
      this.nailMaterial.color.setRGB(
        this.config.nailColor.r / 255,
        this.config.nailColor.g / 255,
        this.config.nailColor.b / 255
      );
    }

    // Debug: Log material state
    console.log("Material updated:", {
      hasTexture: !!this.nailMaterial.map,
      wireframe: this.config.showWireframe,
      opacity: this.config.nailOpacity,
      textureOpacity: this.config.textureOpacity,
    });

    // Handle texture changes
    if (oldConfig.nailTexture !== newConfig.nailTexture) {
      if (newConfig.nailTexture) {
        this.loadTexture(newConfig.nailTexture);
      } else {
        this.removeTexture();
      }
    }

    // Update texture opacity if it changed
    if (this.nailTexture && newConfig.textureOpacity !== undefined) {
      if (newConfig.textureOpacity !== oldConfig.textureOpacity) {
        // When there's a texture, use the texture opacity for the entire material
        this.nailMaterial.opacity = newConfig.textureOpacity;
        this.nailMaterial.needsUpdate = true;
      }
    } else if (!this.nailTexture) {
      // When there's no texture, use the nail opacity
      this.nailMaterial.opacity = this.config.nailOpacity;
    }

    // Check if geometry needs to be rebuilt for all meshes
    if (
      oldConfig.nailCurvature !== newConfig.nailCurvature ||
      oldConfig.nailThickness !== newConfig.nailThickness ||
      oldConfig.enable3DRotation !== newConfig.enable3DRotation
    ) {
      this.nailMeshes.forEach((mesh) => {
        mesh.userData.needsNewGeometry = true;
      });
    }
  }

  private async loadTexture(textureSource: string): Promise<void> {
    try {
      // Dispose of existing texture
      if (this.nailTexture) {
        this.nailTexture.dispose();
        this.nailTexture = null;
      }

      console.log("Loading nail texture...");

      // Load the new texture
      this.nailTexture = await new Promise<THREE.Texture>((resolve, reject) => {
        this.textureLoader.load(
          textureSource,
          (texture) => {
            // Configure texture properties for nail surface
            texture.wrapS = THREE.ClampToEdgeWrapping; // Changed from RepeatWrapping
            texture.wrapT = THREE.ClampToEdgeWrapping; // Changed from RepeatWrapping
            texture.repeat.set(1, 1); // Scale texture to fit nail
            texture.flipY = false; // Prevent texture flipping
            texture.minFilter = THREE.LinearFilter; // Better filtering for small textures
            texture.magFilter = THREE.LinearFilter; // Better filtering for small textures
            texture.generateMipmaps = false; // Disable mipmaps for better texture clarity
            texture.needsUpdate = true; // Force texture update
            resolve(texture);
          },
          undefined, // onProgress
          (error) => {
            console.error("Failed to load nail texture:", error);
            reject(error);
          }
        );
      });

      // Apply texture directly to the nail material
      this.nailMaterial.map = this.nailTexture;
      this.nailMaterial.needsUpdate = true;

      // When texture is applied, adjust material properties for better visibility
      // Keep the base color but reduce its intensity so texture shows through
      this.nailMaterial.color.setRGB(1, 1, 1); // White base color lets texture show true colors

      // Set texture opacity (this will affect the entire material)
      if (this.config.textureOpacity !== undefined) {
        this.nailMaterial.opacity = this.config.textureOpacity;
      } else {
        this.nailMaterial.opacity = 0.9; // Default high opacity for texture visibility
      }

      // Force all meshes to update their materials
      this.nailMeshes.forEach((mesh) => {
        mesh.material = this.nailMaterial;
      });

      console.log("Nail texture loaded and applied to nail material", {
        width: this.nailTexture.image.width,
        height: this.nailTexture.image.height,
        format: this.nailTexture.format,
        type: this.nailTexture.type,
        opacity: this.nailMaterial.opacity,
        hasMap: !!this.nailMaterial.map,
      });
    } catch (error) {
      console.error("Error loading nail texture:", error);
      this.removeTexture();
    }
  }

  private removeTexture(): void {
    if (this.nailTexture) {
      this.nailTexture.dispose();
      this.nailTexture = null;
    }

    // Remove texture from nail material
    this.nailMaterial.map = null;
    this.nailMaterial.needsUpdate = true;

    // Restore original nail color and opacity
    this.nailMaterial.color.setRGB(
      this.config.nailColor.r / 255,
      this.config.nailColor.g / 255,
      this.config.nailColor.b / 255
    );
    this.nailMaterial.opacity = this.config.nailOpacity;

    // Force all meshes to update their materials
    this.nailMeshes.forEach((mesh) => {
      mesh.material = this.nailMaterial;
    });

    // Clean up any remaining texture meshes (they shouldn't exist anymore)
    this.textureMeshes.forEach((mesh, key) => {
      this.scene.remove(mesh);
      mesh.geometry.dispose();
    });
    this.textureMeshes.clear();

    console.log("Nail texture removed from nail material, color restored");
  }

  public async setTexture(textureSource: string | null): Promise<void> {
    if (textureSource) {
      await this.loadTexture(textureSource);
    } else {
      this.removeTexture();
    }
    this.render(); // Re-render with new texture
  }

  public updateNailOverlays(
    matches: NailFingerMatch[],
    scaleX: number,
    scaleY: number
  ): void {
    const currentKeys = new Set(
      matches.map((m) => `${m.handedness}_${m.fingertipIndex}`)
    );

    // Remove old meshes
    this.nailMeshes.forEach((mesh, key) => {
      if (!currentKeys.has(key)) {
        this.scene.remove(mesh);
        mesh.geometry.dispose();
        this.nailMeshes.delete(key);
      }
    });

    // Clean up any old texture meshes (shouldn't exist anymore but just in case)
    this.textureMeshes.forEach((mesh, key) => {
      if (!currentKeys.has(key)) {
        this.scene.remove(mesh);
        mesh.geometry.dispose();
        this.textureMeshes.delete(key);
      }
    });

    // Update or create new nail meshes (texture is now part of the nail material)
    for (const match of matches) {
      this.updateNailMesh(match, scaleX, scaleY);
    }

    this.render();
  }

  private updateNailMesh(
    match: NailFingerMatch,
    scaleX: number,
    scaleY: number
  ): void {
    const key = `${match.handedness}_${match.fingertipIndex}`;
    let mesh = this.nailMeshes.get(key);

    const nailScaleFactor = 1.3; // Make nails slightly larger than detected mask
    const width = match.nailWidth * scaleX * nailScaleFactor;
    const length = match.nailHeight * scaleY * nailScaleFactor;

    const needsNewGeometry =
      !mesh ||
      mesh.userData.needsNewGeometry ||
      mesh.userData.width !== width ||
      mesh.userData.length !== length;

    if (needsNewGeometry) {
      const geometry = this.createNailGeometry(width, length);
      if (mesh) {
        mesh.geometry.dispose();
        mesh.geometry = geometry;
      } else {
        mesh = new THREE.Mesh(geometry, this.nailMaterial);
        this.nailMeshes.set(key, mesh);
        this.scene.add(mesh);
      }
      mesh.userData = { width, length, needsNewGeometry: false };
    }

    // Ensure mesh exists before proceeding
    if (!mesh) {
      console.error("Failed to create or retrieve nail mesh");
      return;
    }

    // --- POSITION ---
    const threeX = match.nailCentroid[0] * scaleX - this.config.canvasWidth / 2;
    const threeY =
      -(match.nailCentroid[1] * scaleY) + this.config.canvasHeight / 2;
    mesh.position.set(threeX, threeY, 0);

    // --- ROTATION ---
    this.applyNailRotation(mesh, match);
  }

  private applyNailRotation(mesh: THREE.Mesh, match: NailFingerMatch): void {
    if (this.config.enable3DRotation && match.orientation) {
      let { xAxis, yAxis, zAxis } = match.orientation;

      // **FIX:** The orientation vectors from `nailMatching` are now pre-converted
      // into a right-handed, Y-up coordinate system suitable for Three.js.
      // No further conversion is needed here.
      let threeX = new THREE.Vector3().fromArray(xAxis); // Width
      let threeY = new THREE.Vector3().fromArray(yAxis); // Normal
      let threeZ = new THREE.Vector3().fromArray(zAxis); // Length

      // Use the working debug defaults as the standard mapping:
      // debugAxisMappingX = "threeZ" (Local X → threeZ)
      // debugAxisMappingY = "threeX" (Local Y → threeX)
      // debugAxisMappingZ = "threeY" (Local Z → threeY)
      const mappedX = threeZ; // Map Local X to threeZ
      const mappedY = threeX; // Map Local Y to threeX
      const mappedZ = threeY; // Map Local Z to threeY

      const rotationMatrix = new THREE.Matrix4().makeBasis(
        mappedX, // Map Local X to chosen world axis
        mappedY, // Map Local Y to chosen world axis
        mappedZ // Map Local Z to chosen world axis
      );

      mesh.quaternion.setFromRotationMatrix(rotationMatrix);
    } else {
      // Fallback to simple 2D rotation
      // Reset 3D rotation and apply only 2D rotation around the screen's Z-axis.
      // The angle is adjusted by -90 degrees because PlaneGeometry's length is along its Y-axis.
      mesh.quaternion.setFromEuler(
        new THREE.Euler(0, 0, match.nailAngle - Math.PI / 2)
      );
    }
  }

  private createNailGeometry(
    width: number,
    length: number
  ): THREE.BufferGeometry {
    if (this.config.enable3DRotation) {
      return this.createExtrudedNailGeometry(width, length);
    }
    return this.createCurvedPlaneGeometry(width, length);
  }

  private createExtrudedNailGeometry(
    width: number,
    length: number
  ): THREE.BufferGeometry {
    const { nailCurvature, nailThickness } = this.config;

    const shape = new THREE.Shape();

    if (nailCurvature < 0.01) {
      // Flat nail cross-section
      shape.moveTo(-width / 2, 0);
      shape.lineTo(width / 2, 0);
    } else {
      // Curved nail cross-section using an arc
      const sagitta = width * nailCurvature * 0.3; // How much it curves up
      if (sagitta < 0.1) {
        // Not enough curve for a radius calc
        shape.moveTo(-width / 2, 0);
        shape.lineTo(width / 2, 0);
      } else {
        const radius =
          (Math.pow(width / 2, 2) + Math.pow(sagitta, 2)) / (2 * sagitta);
        const halfAngle = Math.asin(width / 2 / radius);
        const centerX = 0;
        const centerY = sagitta - radius;
        shape.absarc(centerX, centerY, radius, -halfAngle, halfAngle, false);
      }
    }

    const extrudeSettings = {
      steps: 2,
      depth: length,
      bevelEnabled: true,
      bevelThickness: nailThickness * 0.2,
      bevelSize: nailThickness * 0.2,
      bevelOffset: -nailThickness * 0.2,
      bevelSegments: 2,
    };

    const geom = new THREE.ExtrudeGeometry(shape, extrudeSettings);

    // **GEOMETRY ORIENTATION:** Don't apply the -90° X rotation that was used in the original logic.
    // The debug mode (which worked correctly) skipped this rotation, so we do the same.
    // ExtrudeGeometry creates geometry with:
    // - X: width (correct for nail width)
    // - Y: curvature/thickness (normal direction, pointing up from nail surface)
    // - Z: length (extrusion depth along finger length)
    //
    // We keep this original orientation and handle the mapping in applyNailRotation() instead.

    // Center the geometry on its bounding box, which is critical for proper rotation.
    geom.center();

    // **FIX UV COORDINATES FOR TEXTURE MAPPING**
    // ExtrudeGeometry's default UV mapping is complex and doesn't work well for nail textures
    // We need to generate proper UV coordinates for the top face (where the texture should appear)
    this.generateNailUVCoordinates(geom, width, length);

    return geom;
  }

  private generateNailUVCoordinates(
    geometry: THREE.BufferGeometry,
    width: number,
    length: number
  ): void {
    const positionAttribute = geometry.attributes.position;
    const uvArray: number[] = [];

    // Get bounding box for normalization
    geometry.computeBoundingBox();
    const bbox = geometry.boundingBox!;

    for (let i = 0; i < positionAttribute.count; i++) {
      const x = positionAttribute.getX(i);
      const z = positionAttribute.getZ(i);

      // Normalize coordinates to [0, 1] range for UV mapping
      // Map X coordinate (width) to U
      const u = (x - bbox.min.x) / (bbox.max.x - bbox.min.x);
      // Map Z coordinate (length) to V
      const v = (z - bbox.min.z) / (bbox.max.z - bbox.min.z);

      uvArray.push(u, v);
    }

    // Set the UV attribute
    geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvArray, 2));

    console.log("Generated UV coordinates for extruded nail geometry", {
      vertexCount: positionAttribute.count,
      uvCount: uvArray.length / 2,
      bbox: {
        x: [bbox.min.x, bbox.max.x],
        y: [bbox.min.y, bbox.max.y],
        z: [bbox.min.z, bbox.max.z],
      },
    });
  }

  private createCurvedPlaneGeometry(
    width: number,
    length: number
  ): THREE.BufferGeometry {
    const { nailCurvature } = this.config;
    const geom = new THREE.PlaneGeometry(width, length, 10, 10);
    const positions = geom.attributes.position;
    const curveAmount = width * nailCurvature * 0.4;

    if (curveAmount > 0) {
      for (let i = 0; i < positions.count; i++) {
        const x = positions.getX(i);
        const zOffset = -curveAmount * (1.0 - Math.pow((x / width) * 2, 2));
        positions.setZ(i, zOffset);
      }
    }

    // Ensure UV coordinates are properly set for texture mapping
    // PlaneGeometry already has UV coordinates, but let's make sure they're correct
    const uvAttribute = geom.attributes.uv;
    if (uvAttribute) {
      console.log("UV coordinates found on plane geometry");
    } else {
      console.warn("No UV coordinates found on plane geometry");
    }

    geom.computeVertexNormals();
    return geom;
  }

  public resize(width: number, height: number): void {
    this.config.canvasWidth = width;
    this.config.canvasHeight = height;
    this.camera.left = -width / 2;
    this.camera.right = width / 2;
    this.camera.top = height / 2;
    this.camera.bottom = -height / 2;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
    this.render();
  }

  public render(): void {
    this.renderer.render(this.scene, this.camera);
  }

  public dispose(): void {
    // Dispose of texture if it exists
    if (this.nailTexture) {
      this.nailTexture.dispose();
      this.nailTexture = null;
    }

    // Dispose of materials
    this.nailMaterial?.dispose();
    this.textureMaterial?.dispose();

    // Dispose of nail meshes
    this.nailMeshes.forEach((mesh) => {
      this.scene.remove(mesh);
      mesh.geometry.dispose();
    });
    this.nailMeshes.clear();

    // Dispose of texture meshes
    this.textureMeshes.forEach((mesh) => {
      this.scene.remove(mesh);
      mesh.geometry.dispose();
    });
    this.textureMeshes.clear();

    if (this.renderer) {
      this.renderer.dispose();
      if (this.containerElement?.contains(this.renderer.domElement)) {
        this.containerElement.removeChild(this.renderer.domElement);
      }
    }
    console.log("3D nail overlay disposed.");
  }
}
