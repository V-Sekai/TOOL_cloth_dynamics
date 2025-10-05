# API Design

## Public Interfaces

### SkeletonLoader Class

Provides functionality for loading and parsing skeletal data from standard file formats.

**Primary Interface:**
```cpp
class SkeletonLoader {
public:
    // File-based loading
    static Skeleton loadFromOBJ(const std::string& filepath);

    // Bone extraction utilities
    static std::vector<std::pair<Vec3d, Vec3d>> loadBones(
        const std::string& filepath);

    // Edge compatibility validation
    static bool validateEdgeCompatibility(
        const Skeleton& skeleton_a,
        const Skeleton& skeleton_b);
};
```

**Key Methods:**
- **loadFromOBJ**: Parse complete skeleton from OBJ format with vertices and lines
- **loadBones**: Extract bone segments as endpoint pairs
- **validateEdgeCompatibility**: Ensure skeletons have identical topology for retargeting

### CapsuleGenerator Class

Handles the conversion of skeletal bones into collision capsule primitives.

**Primary Interface:**
```cpp
class CapsuleGenerator {
public:
    // Basic capsule generation
    static std::vector<std::unique_ptr<TaperedCapsule>> generateCapsules(
        const Skeleton& skeleton,
        double radius = 0.1);

    // Advanced radius estimation
    static std::vector<std::unique_ptr<TaperedCapsule>> generateCapsulesWithAdvancedRadii(
        const Skeleton& skeleton,
        const MatXd& mesh_vertices,
        bool use_tapered_radii = true);

    // Individual bone conversion
    static std::unique_ptr<TaperedCapsule> boneToTaperedCapsule(
        const Bone& bone,
        double radius_top = 0.1,
        double radius_bottom = 0.1,
        Vec3d color = Vec3d(0.8, 0.7, 0.6));
};
```

**Generation Strategies:**
- **Fixed Radius**: Constant dimensions for all capsules
- **Mesh-Based**: Adaptive sizing using proximity queries
- **Tapered**: Variable radius along bone axis

### CapsuleRig Class

Manages collections of capsules derived from skeletons, providing simulation integration.

**Primary Interface:**
```cpp
class CapsuleRig {
public:
    // Construction
    static CapsuleRig generate(const Skeleton& skeleton,
                              double default_radius = 0.1);

    // Simulation integration
    void addToSimulation(Simulation* sim);
    void pinGarmentToCapsules(Simulation* sim, double pin_distance = 0.05);

    // Asset pairing validation
    static bool validateAssetPairing(
        const std::string& skeleton_path,
        const std::string& mesh_path);

    // Accessors
    const Skeleton& getSkeleton() const;
    const std::vector<std::unique_ptr<TaperedCapsule>>& getCapsules() const;
    size_t getCapsuleCount() const;
    bool isValid() const;
};
```

**Core Responsibilities:**
- **Capsule Ownership**: Manages lifetime of collision primitives
- **Simulation Binding**: Integrates capsules into physics simulation
- **Cloth Attachment**: Provides automatic garment pinning
- **Data Consistency**: Maintains bone-to-capsule correspondence

### SpringBoneExtender Class

Extends basic rigs with procedural animation chains for dynamic elements.

**Primary Interface:**
```cpp
class SpringBoneExtender {
public:
    // Configuration structure
    struct Config {
        int subdivisions_per_bone = 3;
        double taper_factor = 0.9;
        double spring_stiffness = 100.0;
        double spring_damping = 0.5;
    };

    // Basic extension
    static CapsuleRig extend(const CapsuleRig& base, const Config& cfg);

    // VRM-compatible extension
    static CapsuleRig extendWithVRMSpringBones(
        const CapsuleRig& base,
        const VRMSpringBoneConfig& vrm_config);
};
```

**Extension Modes:**
- **Standard**: Generic procedural animation chains
- **VRM Compatible**: VRM specification-compliant spring bones

## Error Handling

### Exception Types

**File I/O Errors:**
- `std::runtime_error`: Invalid file paths, missing files, parse errors
- **Recovery**: Validate file existence before operations

**Data Validation Errors:**
- `std::invalid_argument`: Malformed skeleton data, invalid topology
- **Recovery**: Check skeleton validity with `isValid()` methods

**Asset Pairing Errors:**
- `std::runtime_error`: Mismatched skeleton/mesh files
- **Recovery**: Use `validateAssetPairing()` before generation

## Memory Management

### Ownership Semantics

**Capsule Ownership:**
- `CapsuleRig` owns all `TaperedCapsule` instances via `unique_ptr`
- External references use raw pointers for simulation integration
- Automatic cleanup on rig destruction

**Skeleton Data:**
- `Skeleton` structures are passed by const reference
- No internal ownership of skeleton data
- External management of skeleton lifetime

### Resource Management

**Large Mesh Handling:**
- Sparse octree construction for memory efficiency
- Progressive loading for large datasets
- Cached computations where possible

## Thread Safety

### Single-Threaded Design

**Assumptions:**
- All operations execute in single-threaded context
- No concurrent access to shared data structures
- Simulation integration handles multi-threading

**Future Considerations:**
- Potential for parallel radius estimation
- Thread-safe asset loading
- Concurrent mesh processing
