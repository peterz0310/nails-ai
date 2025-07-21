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
    const nailLength = match.nailWidth * scaleX * nailScaleFactor; // Along finger
    const nailWidth = match.nailHeight * scaleY * nailScaleFactor; // Across finger
    const nailThickness = this.config.nailThickness;

    let mesh = this.nailMeshes.get(key);

    if (!mesh) {
      const geometry = this.createNailGeometry(
        nailLength,
        nailWidth,
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
        nailLength,
        nailWidth,
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

    // Convert from MediaPipe coordinates (Y-down, Z-towards-viewer) to
    // Three.js coordinates (Y-up, Z-towards-viewer is correct)
    const threeX = new THREE.Vector3(xAxis[0], -xAxis[1], xAxis[2]);
    const threeY = new THREE.Vector3(yAxis[0], -yAxis[1], yAxis[2]);
    const threeZ = new THREE.Vector3(zAxis[0], -zAxis[1], zAxis[2]);

    // Create a rotation matrix from the three basis vectors
    const rotationMatrix = new THREE.Matrix4().makeBasis(
      threeX,
      threeY,
      threeZ
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
    const shape = new THREE.Shape();
    const halfWidth = width / 2;
    const halfHeight = height / 2;
    const radius = height * 0.4; // Rounded tip

    shape.moveTo(-halfWidth, -halfHeight + radius);
    shape.lineTo(-halfWidth, halfHeight);
    shape.lineTo(halfWidth, halfHeight);
    shape.lineTo(halfWidth, -halfHeight + radius);
    shape.absarc(
      halfWidth - radius,
      -halfHeight + radius,
      radius,
      0,
      Math.PI * 0.5,
      false
    );
    shape.lineTo(-halfWidth + radius, -halfHeight);
    shape.absarc(
      -halfWidth + radius,
      -halfHeight + radius,
      radius,
      Math.PI,
      Math.PI * 1.5,
      false
    );

    const extrudeSettings = {
      steps: 1,
      depth: depth,
      bevelEnabled: true,
      bevelThickness: 1,
      bevelSize: 1,
      bevelSegments: 2,
    };

    // Use a simple Box for performance, or ExtrudeGeometry for shape.
    // Let's create a curved plane which is more efficient.
    const planeGeom = new THREE.PlaneGeometry(width, height, 10, 2);
    const positions = planeGeom.attributes.position;

    // Apply curvature along the nail's width (X-axis)
    const curveRadius = width / 2 / Math.sin(Math.PI * curvature * 0.4);
    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i);
      if (curvature > 0.05) {
        const zOffset =
          curveRadius - Math.sqrt(curveRadius * curveRadius - x * x);
        positions.setZ(i, -zOffset);
      }
    }
    planeGeom.computeVertexNormals();

    // The geometry is created flat (on XY plane). The rotation will orient it correctly.
    // We need to rotate it so it aligns with our basis vectors. X->width, Y->height
    planeGeom.rotateX(Math.PI / 2); // Make it lie on XZ plane initially
    planeGeom.scale(-1, 1, 1); // Flip to align with handedness of basis

    return planeGeom;
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
    this.renderer.dispose();
    if (this.containerElement.contains(this.renderer.domElement)) {
      this.containerElement.removeChild(this.renderer.domElement);
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
