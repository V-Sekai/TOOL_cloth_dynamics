# DiffCloth Integration

## Simulation Integration

### Primitive System Architecture

TaperedCapsule primitives integrate seamlessly with the DiffCloth collision detection system through the standard Primitive interface.

**Base Primitive Interface**:
```cpp
class Primitive {
public:
    virtual bool isInContact(const Vec3d& point, Vec3d& normal, double& distance) = 0;
    virtual Vec3d closestPointOnSurface(const Vec3d& point) = 0;

    Vec3d center;
    Vec3d velocity;
    double friction_coefficient;
};
```

**TaperedCapsule Implementation**:
```cpp
class TaperedCapsule : public Primitive {
public:
    bool isInContact(const Vec3d& point, Vec3d& normal, double& distance) override;
    Vec3d closestPointOnSurface(const Vec3d& point) override;

    // Capsule-specific properties
    double radius_top, radius_bottom, height;
    Vec3d axis;
    Vec3d color;
};
```

### Scene Configuration

**Primitive Types Enumeration**:
```cpp
enum PrimitiveConfiguration {
    SPHERES_ONLY,
    BOXES_ONLY,
    SKELETON_CAPSULES,      // Capsule-based skeletons
    CUSTOM_PRIMITIVES
};
```

**Scene Initialization**:
```cpp
void Simulation::initializeScene(const SceneConfiguration& config) {
    switch (config.primitive_config) {
        case SKELETON_CAPSULES: {
            loadSkeletonCapsules(config.skeleton_path);
            break;
        }
        // ... other primitive types ...
    }
}
```

### Cloth-Capsule Interaction

**Automatic Attachment**:
- Proximity-based vertex pinning
- Distance threshold configuration
- High-stiffness spring constraints

**Collision Response**:
- Continuous collision detection
- Impulse-based resolution
- Friction and restitution parameters

### Integration Architecture

**DiffCloth Integration**:
- tool_cloth_dynamics integrates with DiffCloth systems
- Leverages DiffCloth's constraint and collision systems
- Maintains differentiable optimization capabilities

**System Responsibilities**:
- **DiffCloth**: Core physics simulation, constraint solving, collision detection
- **tool_cloth_dynamics**: Garment fitting, semantic anchoring, optimization integration
- **CapsuleRig**: Collision primitive management, skeleton-to-capsule conversion

**Data Flow**:
```
Garment Mesh → tool_cloth_dynamics → Semantic Anchors → DiffCloth Constraints
Skeleton OBJ  → CapsuleRig Generation → Collision Primitives → DiffCloth Physics
```

### Performance Considerations

**Spatial Optimization**:
- Broad-phase collision culling
- Capsule-specific acceleration structures
- Hierarchical collision testing

**Memory Management**:
- Efficient primitive storage
- Lazy mesh generation for visualization
- Shared geometry data where possible
