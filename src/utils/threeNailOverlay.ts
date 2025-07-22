/**
 * Three.js 3D Nail Overlay System (REVISED & IMPROVED)
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
  // Debug rotation controls
  rotationDebugMode?: boolean;
  debugAxisMappingX?: "threeX" | "threeY" | "threeZ";
  debugAxisMappingY?: "threeX" | "threeY" | "threeZ";
  debugAxisMappingZ?: "threeX" | "threeY" | "threeZ";
  preRotationX?: number;
  preRotationY?: number;
  preRotationZ?: number;
  // Mirror correction for camera feed
  debugMirrorCorrection?: boolean;
}

export class ThreeNailOverlay {
  private scene: THREE.Scene;
  private camera: THREE.OrthographicCamera;
  private renderer: THREE.WebGLRenderer;
  private nailMeshes: Map<string, THREE.Mesh> = new Map();
  private config: ThreeNailOverlayConfig;
  private containerElement: HTMLElement;
  private nailMaterial!: THREE.MeshStandardMaterial;
  private ambientLight!: THREE.AmbientLight;
  private dirLight!: THREE.DirectionalLight;

  constructor(containerElement: HTMLElement, config: ThreeNailOverlayConfig) {
    this.containerElement = containerElement;
    this.config = { ...config };
    this.scene = new THREE.Scene();

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
    this.nailMaterial = new THREE.MeshStandardMaterial({
      color: 0xff6b9d, // Default pink
      transparent: true,
      side: THREE.DoubleSide,
    });
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
    this.nailMaterial.color.setRGB(
      this.config.nailColor.r / 255,
      this.config.nailColor.g / 255,
      this.config.nailColor.b / 255
    );

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

    // Update or create new meshes
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

      // **MIRROR CORRECTION:** Handle coordinate system handedness from mirrored camera
      if (this.config.debugMirrorCorrection) {
        // When camera is mirrored, we need to flip the X-axis to convert from
        // left-handed (mirrored camera) to right-handed (Three.js) coordinate system
        threeX.negate();
        // Also negate Z-axis to maintain proper handedness
        threeZ.negate();
      }

      if (this.config.rotationDebugMode) {
        // **DEBUG MODE:** Use manual axis mapping and pre-rotations
        const axisMap = {
          threeX,
          threeY,
          threeZ,
        };

        // Apply manual axis mapping
        const mappedX = axisMap[this.config.debugAxisMappingX || "threeX"];
        const mappedY = axisMap[this.config.debugAxisMappingY || "threeZ"];
        const mappedZ = axisMap[this.config.debugAxisMappingZ || "threeY"];

        const rotationMatrix = new THREE.Matrix4().makeBasis(
          mappedX, // Map Local X to chosen world axis
          mappedY, // Map Local Y to chosen world axis
          mappedZ // Map Local Z to chosen world axis
        );

        mesh.quaternion.setFromRotationMatrix(rotationMatrix);

        // Apply additional pre-rotation for debugging
        const preRotation = new THREE.Euler(
          this.config.preRotationX || 0,
          this.config.preRotationY || 0,
          this.config.preRotationZ || 0
        );
        const preQuaternion = new THREE.Quaternion().setFromEuler(preRotation);
        mesh.quaternion.premultiply(preQuaternion);
      } else {
        // **ORIGINAL LOGIC:** Use the fixed mapping
        // After rotating the ExtrudeGeometry -90° around X, our geometry now has:
        // - Local X: Width (across nail)
        // - Local Y: Length (along finger, was Z before rotation)
        // - Local Z: Thickness/Normal (pointing out from nail surface, was Y before rotation)
        //
        // Map to world axes:
        const rotationMatrix = new THREE.Matrix4().makeBasis(
          threeX, // Map Local X to World X (Width)
          threeZ, // Map Local Y to World Z (Length)
          threeY // Map Local Z to World Y (Normal)
        );

        mesh.quaternion.setFromRotationMatrix(rotationMatrix);
      }
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

    if (!this.config.rotationDebugMode) {
      // **ORIGINAL:** Apply the -90° X rotation when not in debug mode
      // ExtrudeGeometry creates geometry with:
      // - X: width (correct for nail width)
      // - Y: curvature/thickness (normal direction, pointing up from nail surface)
      // - Z: length (extrusion depth along finger length)
      //
      // We want the nail to lie flat by default, with:
      // - X: width (across nail)
      // - Y: length (along finger)
      // - Z: thickness (normal from surface)
      //
      // So we rotate the extruded geometry 90 degrees around X axis
      geom.rotateX(-Math.PI / 2);
    }

    // Center the geometry on its bounding box, which is critical for proper rotation.
    geom.center();
    return geom;
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
    this.nailMaterial?.dispose();
    this.nailMeshes.forEach((mesh) => {
      this.scene.remove(mesh);
      mesh.geometry.dispose();
    });
    this.nailMeshes.clear();
    if (this.renderer) {
      this.renderer.dispose();
      if (this.containerElement?.contains(this.renderer.domElement)) {
        this.containerElement.removeChild(this.renderer.domElement);
      }
    }
    console.log("3D nail overlay disposed.");
  }
}
