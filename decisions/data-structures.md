# Data Structures

## Core Data Structures

### Bone Structure

The `Bone` structure represents a single skeletal segment connecting two joints, serving as the fundamental unit for capsule generation.

**Definition:**
```cpp
struct Bone {
    Vec3d start;        // Proximal joint position (e.g., shoulder)
    Vec3d end;          // Distal joint position (e.g., elbow)
    int parent_id;      // Parent bone index (-1 for root)
    std::string name;   // Optional semantic identifier

    // Computed properties
    Vec3d get_axis() const;     // Normalized bone direction vector
    double get_length() const;  // Distance between joints
    Vec3d get_center() const;   // Midpoint between joints
};
```

**Key Properties:**
- **start/end**: Define the bone segment in 3D space
- **parent_id**: Enables hierarchical skeleton representation
- **name**: Supports semantic bone identification (e.g., "upper_arm_L")
- **axis**: Normalized direction vector for capsule orientation
- **length**: Determines capsule height parameter
- **center**: Reference point for capsule positioning

**Relationship to Capsules:**
Each bone generates exactly one tapered capsule where:
- Capsule height = bone length
- Capsule axis = bone axis
- Capsule center = bone center
- Capsule radii determined by radius estimation algorithms

### Skeleton Structure

The `Skeleton` structure represents a complete articulated figure composed of interconnected bones.

**Definition:**
```cpp
struct Skeleton {
    std::vector<Bone> bones;        // All bone segments
    std::vector<Vec3d> joints;      // All unique joint positions

    // Construction
    static Skeleton fromOBJ(const std::string& filepath);

    // Accessors
    size_t getBoneCount() const;
    size_t getJointCount() const;
    bool isValid() const;
};
```

**Properties:**
- **bones**: Ordered collection of bone segments
- **joints**: Unique joint positions (derived from bone endpoints)
- **Validation**: Ensures structural integrity and connectivity

**Topology Constraints:**
- Bones form a directed acyclic graph (DAG)
- Joint positions are consistent across connected bones
- Root bones have parent_id = -1

### CapsuleRig Structure

The `CapsuleRig` manages a collection of collision capsules derived from a skeleton, providing simulation integration and cloth attachment functionality.

**Definition:**
```cpp
class CapsuleRig {
public:
    // Core data
    std::vector<std::unique_ptr<TaperedCapsule>> capsules;
    Skeleton skeleton;

    // Construction
    static CapsuleRig generate(const Skeleton& skeleton,
                              double default_radius = 0.1);

    // Simulation integration
    void addToSimulation(Simulation* sim);
    void pinGarmentToCapsules(Simulation* sim, double pin_distance = 0.05);

    // Accessors
    const std::vector<std::unique_ptr<TaperedCapsule>>& getCapsules() const;
    size_t getCapsuleCount() const;
    bool isValid() const;
};
```

**Key Responsibilities:**
- **Capsule Management**: Owns and manages tapered capsule primitives
- **Simulation Integration**: Adds capsules to physics simulation
- **Cloth Attachment**: Provides proximity-based pinning for garments
- **Data Consistency**: Maintains correspondence between bones and capsules
