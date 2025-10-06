# Demo Scenes

## Standard Test Cases

### Skeleton Draping Scenario

Demonstrates cloth interaction with skeletal collision primitives under gravity.

**Scene Configuration:**
```cpp
SceneConfiguration createSkeletonDrapingDemo() {
    SceneConfiguration config;

    // Skeleton collision primitives
    config.collisionPrimitives = SKELETON_CAPSULES;
    config.skeletonPath = "assets/skeletons/humanoid_basic.obj";

    // Cloth garment
    config.clothMesh = "assets/garments/jumpsuit.obj";
    config.clothMaterial = createStandardClothMaterial();

    // Simulation parameters
    config.timeStep = 1.0 / 60.0;      // 60 FPS
    config.duration = 5.0;             // 5 second simulation
    config.gravity = Vec3d(0, -9.81, 0); // Standard gravity

    // Initial conditions
    config.clothInitialPosition = Vec3d(0, 2.0, 0); // Above skeleton
    config.skeletonPosition = Vec3d(0, 0, 0);       // Ground level

    return config;
}
```

**Expected Behavior Timeline:**

| Time | Description | Validation Criteria |
|------|-------------|-------------------|
| **T = 0.0s** | Initialization | Cloth collapsed toward body centroid |
| **T = 0.2s** | Centroid collapse | Garment vertices move toward body center |
| **T = 0.5s** | Expansion begins | Repulsive forces push cloth outward from center |
| **T = 1.0s** | First contact | Inflated cloth touches shoulder capsules |
| **T = 1.5s** | Semantic anchoring | Anchor points (shoulders, waist) establish attachment |
| **T = 2.0s** | Torso draping | Cloth conforms to torso and arm contours |
| **T = 3.0s** | Limb interaction | Sleeves and pant legs form naturally |
| **T = 4.0s** | Refinement | Final adjustments under gravity and collisions |
| **T = 5.0s** | Equilibrium | Natural garment shape with proper fit |

### Gravity Physics Demo

Showcases gravity-only physics simulation for cloth draping.

**Configuration:**
```cpp
SceneConfiguration createGravityPhysicsDemo() {
    SceneConfiguration config;

    // Base skeleton with capsule primitives
    config.collisionPrimitives = SKELETON_CAPSULES;
    config.skeletonPath = "assets/skeletons/character_rig.obj";

    // Gravity physics parameters
    config.enableGravity = true;
    config.gravityStrength = -9.81;
    config.deterministicMode = true;

    // Animation parameters
    config.primaryAnimation = "walk_cycle.anim";

    return config;
}
```

**Physics Characteristics:**
- **Primary Motion**: Base skeleton follows keyframe animation
- **Gravity Effects**: Cloth responds naturally to gravitational forces
- **Collision**: Capsules prevent interpenetration
- **Performance**: Real-time physics at 60 FPS

### Validation Metrics

**Cloth-Skeleton Interaction:**
- **Contact Detection**: All cloth vertices within threshold distance attach to capsules
- **Stability**: No cloth self-intersection or excessive stretching
- **Realism**: Natural garment drape following body contours

**Performance Benchmarks:**
- **Frame Rate**: Maintain 60 FPS with 10K+ cloth vertices
- **Memory Usage**: < 100MB for typical character + garment
- **Initialization**: < 500ms for skeleton loading and capsule generation

### Asset Requirements

**Skeleton Assets:**
- OBJ format with vertices (joints) and lines (bones)
- Consistent topology for retargeting compatibility
- Named bones for semantic identification

**Cloth Assets:**
- Triangle mesh in OBJ format
- Reasonable vertex density (1K-10K vertices)
- Physically plausible material properties

**Directory Structure:**
```
assets/
├── skeletons/          # Skeleton OBJ files
├── garments/          # Cloth meshes
├── animations/        # Motion data
└── materials/         # Cloth properties
```
