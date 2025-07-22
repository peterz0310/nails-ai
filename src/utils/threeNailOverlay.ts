/**
 * Three.js 3D Nail Overlay System (REVISED)
 *
 * This module provides 3D nail overlay functionality using Three.js.
 *
 * FIXED Features:
 * - Uses a full 3D orientation basis (from nailMatching.ts) for robust rotation.
 * - Applies rotation using Quaternions to avoid gimbal lock and instability.
 * - Nail geometry is procedurally generated with adjustable curvature.
 * - Correctly applies nail width/length to geometry for accurate shaping.
 * - Optimized to only recreate nail geometry when its parameters change.
 */

import * as THREE from "three";
import { NailFingerMatch } from "./nailMatching";

export interface ThreeNailOverlayConfig {
  canvasWidth: number;
  canvasHeight: number;
  videoWidth: number;
  videoHeight: number;
  enableLighting: boolean;
  nailThickness: number;
  nailOpacity: number;
  showWireframe: boolean;
  metallicIntensity: number;
  roughness: number;
  enableReflections: boolean;
  enable3DRotation: boolean;
  nailCurvature: number; // 0=flat, 1=very curved
}

export class ThreeNailOverlay {
  private scene: THREE.Scene;
  private camera: THREE.OrthographicCamera;
  private renderer: THREE.WebGLRenderer;
  private nailMeshes: Map<string, THREE.Mesh> = new Map();
  private config: ThreeNailOverlayConfig;
  private containerElement: HTMLElement;
  private nailMaterial!: THREE.MeshStandardMaterial;

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

    console.log("✅ 3D nail overlay initialized with robust rotation logic.");
  }

  private setupLighting(): void {
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(50, 100, 150);
    this.scene.add(dirLight);
  }

  private createMaterials(): void {
    this.nailMaterial = new THREE.MeshStandardMaterial({
      color: 0xff6b9d, // Default pink
      transparent: true,
      opacity: this.config.nailOpacity,
      metalness: this.config.metallicIntensity,
      roughness: this.config.roughness,
      side: THREE.DoubleSide,
    });
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

    // --- DIMENSIONS (FIXED) ---
    // Correctly map nail dimensions. nailWidth is across, nailHeight is along.
    const nailScaleFactor = 1.4; // Make nails slightly larger than detected mask
    const width = match.nailWidth * scaleX * nailScaleFactor;
    const length = match.nailHeight * scaleY * nailScaleFactor;
    const thickness = this.config.nailThickness;

    let mesh = this.nailMeshes.get(key);

    // OPTIMIZATION: Only update geometry if parameters have changed.
    const needsNewGeometry =
      !mesh ||
      mesh.userData.width !== width ||
      mesh.userData.length !== length ||
      mesh.userData.curvature !== this.config.nailCurvature;

    if (needsNewGeometry) {
      const geometry = this.createNailGeometry(
        width,
        length,
        this.config.nailCurvature
      );

      if (mesh) {
        mesh.geometry.dispose();
        mesh.geometry = geometry;
      } else {
        mesh = new THREE.Mesh(geometry, this.nailMaterial.clone());
        this.nailMeshes.set(key, mesh);
        this.scene.add(mesh);
      }
      // Store current parameters to check against next frame
      mesh.userData = { width, length, curvature: this.config.nailCurvature };
    }

    // --- POSITION ---
    // Convert canvas coordinates (top-left origin) to Three.js coordinates (center origin)
    const threeX = match.nailCentroid[0] * scaleX - this.config.canvasWidth / 2;
    const threeY =
      -(match.nailCentroid[1] * scaleY) + this.config.canvasHeight / 2;
    mesh.position.set(threeX, threeY, 0);

    // --- ROTATION ---
    this.applyNailRotation(mesh, match);

    // --- MATERIAL ---
    if (mesh.material instanceof THREE.MeshStandardMaterial) {
      mesh.material.opacity = this.config.nailOpacity;
      mesh.material.metalness = this.config.metallicIntensity;
      mesh.material.roughness = this.config.roughness;
      mesh.material.wireframe = this.config.showWireframe;
    }
  }

  /**
   * Applies the calculated 3D orientation to the nail mesh.
   */
  private applyNailRotation(mesh: THREE.Mesh, match: NailFingerMatch): void {
    const { xAxis, yAxis, zAxis } = match.orientation;

    // The basis vectors from nailMatching give us a right-handed coordinate system
    // relative to the MediaPipe world space (where Y points down).
    //  - zAxis: Points along the finger's length.
    //  - yAxis: Points out from the nail surface (the normal).
    //  - xAxis: Points across the nail's width.

    // Our Three.js scene uses a different coordinate system (Y points up).
    // We must convert the orientation vectors by negating their Y component.
    const threeX = new THREE.Vector3(xAxis[0], -xAxis[1], xAxis[2]);
    const threeY = new THREE.Vector3(yAxis[0], -yAxis[1], yAxis[2]);
    const threeZ = new THREE.Vector3(zAxis[0], -zAxis[1], zAxis[2]);

    // Our nail geometry is a PlaneGeometry created on the XY plane:
    // - Geometry's local X-axis represents nail WIDTH.
    // - Geometry's local Y-axis represents nail LENGTH.
    // - Geometry's local Z-axis represents nail NORMAL.
    //
    // We map these local axes to our calculated world-space axes:
    // - Map local X (width) to `threeX`.
    // - Map local Y (length) to `threeZ`.
    // - Map local Z (normal) to `threeY`.

    // The .makeBasis() method creates a rotation matrix from three basis vectors
    // that will become the object's new X, Y, and Z axes in world space.
    const rotationMatrix = new THREE.Matrix4().makeBasis(
      threeX, // New X axis (maps to geometry's X - width)
      threeZ, // New Y axis (maps to geometry's Y - length)
      threeY // New Z axis (maps to geometry's Z - normal)
    );

    // Set the mesh's rotation from this matrix. Using a quaternion is best
    // practice to avoid issues like gimbal lock.
    mesh.quaternion.setFromRotationMatrix(rotationMatrix);
  }

  /**
   * FIXED: Procedural geometry for the nail, now with correct dimensions.
   * Creates a curved plane that responds to the 'Curvature' slider.
   */
  private createNailGeometry(
    width: number,
    length: number, // Renamed from height for clarity
    curvature: number
  ): THREE.BufferGeometry {
    // A curved plane is efficient. Geometry is created with width on X, length on Y.
    const geom = new THREE.PlaneGeometry(width, length, 10, 10);
    const positions = geom.attributes.position;

    // Apply curvature along the nail's width (local X-axis).
    const curveAmount = width * curvature * 0.4;

    if (curvature > 0.01) {
      for (let i = 0; i < positions.count; i++) {
        const x = positions.getX(i);
        const y = positions.getY(i);

        // Apply a parabolic curve for a smooth arc across the width.
        const zOffset = -curveAmount * (1.0 - Math.pow(x / (width / 2), 2));
        positions.setZ(i, zOffset);

        // Taper the tip slightly for a more natural shape.
        // We only taper the top part of the nail (positive y).
        if (y > length / 4) {
          // Start tapering from a quarter way up
          const taperProgress = (y - length / 4) / (length * (3 / 4));
          const taper = 1.0 - taperProgress * 0.3; // Taper up to 30% at the tip
          positions.setX(i, x * taper);
        }
      }
    }
    // Recalculate normals for correct lighting on the curved surface.
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
    this.nailMeshes.forEach((mesh) => {
      this.scene.remove(mesh);
      mesh.geometry.dispose();
      if (Array.isArray(mesh.material)) {
        mesh.material.forEach((m) => m.dispose());
      } else {
        mesh.material.dispose();
      }
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

  // --- Public methods to update config from UI ---
  public setWireframeMode(enabled: boolean) {
    this.config.showWireframe = enabled;
  }
  public setOpacity(opacity: number) {
    this.config.nailOpacity = opacity;
  }
  public setNailThickness(thickness: number) {
    this.config.nailThickness = thickness;
  }
  public setMetallicIntensity(intensity: number) {
    this.config.metallicIntensity = intensity;
  }
  public setRoughness(roughness: number) {
    this.config.roughness = roughness;
  }
  public setNailCurvature(curvature: number) {
    this.config.nailCurvature = curvature;
  }
  public setReflectionsEnabled(enabled: boolean) {
    this.config.enableReflections = enabled;
  }
  public set3DRotationEnabled(enabled: boolean) {
    this.config.enable3DRotation = enabled;
  }
  public getConfig = () => ({ ...this.config });
}
