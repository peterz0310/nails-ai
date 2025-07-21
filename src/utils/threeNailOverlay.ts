/**
 * Three.js 3D Nail Overlay System (REVISED)
 *
 * This module provides 3D nail overlay functionality using Three.js.
 *
 * FIXED Features:
 * - Uses a full 3D orientation basis (from nailMatching.ts) for robust rotation.
 * - Applies rotation using Quaternions to avoid gimbal lock and instability.
 * - Nail geometry is procedurally generated with adjustable curvature.
 * - Simplified and more reliable logic for positioning and rotating nails.
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
  private animationId: number | null = null;
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
    matches.forEach((match) => {
      this.updateNailMesh(match, scaleX, scaleY);
    });

    this.render();
  }

  private updateNailMesh(
    match: NailFingerMatch,
    scaleX: number,
    scaleY: number
  ): void {
    const key = `${match.handedness}_${match.fingertipIndex}`;

    // Scale dimensions. Use a scaling factor to make nails more visible.
    const nailScaleFactor = 1.4;
    const nailLength = match.nailWidth * scaleY * nailScaleFactor; // Along finger (use scaleY for consistency)
    const nailWidth = match.nailHeight * scaleX * nailScaleFactor; // Across finger (use scaleX for consistency)
    const nailThickness = this.config.nailThickness;

    let mesh = this.nailMeshes.get(key);

    if (!mesh) {
      const geometry = this.createNailGeometry(
        nailWidth,
        nailLength,
        nailThickness,
        this.config.nailCurvature
      );
      mesh = new THREE.Mesh(geometry, this.nailMaterial.clone());
      this.nailMeshes.set(key, mesh);
      this.scene.add(mesh);
    } else {
      // Update geometry if parameters changed
      mesh.geometry.dispose();
      mesh.geometry = this.createNailGeometry(
        nailWidth,
        nailLength,
        nailThickness,
        this.config.nailCurvature
      );
    }

    // --- POSITION ---
    // Convert canvas coordinates (top-left origin) to Three.js coordinates (center origin)
    const threeX = match.nailCentroid[0] * scaleX - this.config.canvasWidth / 2;
    const threeY =
      -(match.nailCentroid[1] * scaleY) + this.config.canvasHeight / 2;
    mesh.position.set(threeX, threeY, 0);

    // --- ROTATION (The Core Fix) ---
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
   * FIXED: This is the new, robust rotation logic.
   * It uses the pre-calculated 3D basis vectors to construct a rotation matrix
   * and applies it via a quaternion, completely avoiding Euler angle issues.
   */
  private applyNailRotation(mesh: THREE.Mesh, match: NailFingerMatch): void {
    const { xAxis, yAxis, zAxis } = match.orientation;

    // The basis vectors from nailMatching are already normalized and orthogonal.
    // The Z-axis points along the finger.
    // The Y-axis points out from the nail surface (the normal).
    // The X-axis points across the nail.

    // We need to map our nail geometry (created on the XY plane) to this basis.
    // Our nail geometry has width along X and length along Y.
    // So, we map: Geometry X -> World X-axis, Geometry Y -> World Z-axis, Geometry Z -> World Y-axis

    // Three.js uses a right-handed coordinate system (Y-up). MediaPipe uses a different
    // convention (Y-down). The `nailMatching` utility has already computed a clean,
    // right-handed basis for us relative to the world. We just need to apply it.

    const threeZ = new THREE.Vector3(zAxis[0], -zAxis[1], zAxis[2]).normalize(); // Along the finger
    const threeX = new THREE.Vector3(xAxis[0], -xAxis[1], xAxis[2]).normalize(); // Across the nail
    const threeY = new THREE.Vector3(yAxis[0], -yAxis[1], yAxis[2]).normalize(); // Out of the nail

    // Create a rotation matrix from the three basis vectors
    const rotationMatrix = new THREE.Matrix4().makeBasis(
      threeX, // Corresponds to the nail's width direction
      threeZ, // Corresponds to the nail's length direction
      threeY // Corresponds to the nail's normal (thickness)
    );

    // Set the mesh's rotation from this matrix using a quaternion
    mesh.quaternion.setFromRotationMatrix(rotationMatrix);
  }

  /**
   * FIXED: New procedural geometry function for the nail.
   * Creates a curved plane that responds smoothly to the 'Curvature' slider.
   */
  private createNailGeometry(
    width: number,
    height: number,
    depth: number,
    curvature: number
  ): THREE.BufferGeometry {
    // A curved plane is more efficient and looks better than an extruded shape.
    const geom = new THREE.PlaneGeometry(width, height, 10, 10);
    const positions = geom.attributes.position;

    // Apply curvature along the nail's width (local X-axis)
    const curveAmount = width * curvature * 0.5;

    if (curvature > 0.01) {
      for (let i = 0; i < positions.count; i++) {
        const x = positions.getX(i);
        const y = positions.getY(i);

        // Apply quadratic curve for a smooth arc
        const zOffset = -curveAmount * (1.0 - Math.pow(x / (width / 2), 2));
        positions.setZ(i, zOffset);

        // Taper the tip slightly for a more natural shape
        if (y > 0) {
          const taper = 1.0 - (y / height) * 0.4; // Taper top 40%
          positions.setX(i, x * taper);
        }
      }
    }
    geom.computeVertexNormals();

    // The geometry is created flat on the XY plane.
    // The rotation logic will orient it correctly in 3D space.
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
    if (this.animationId) cancelAnimationFrame(this.animationId);
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
      if (
        this.containerElement &&
        this.containerElement.contains(this.renderer.domElement)
      ) {
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
