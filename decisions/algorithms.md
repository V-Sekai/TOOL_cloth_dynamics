# Algorithms

## Core Algorithms

### Radius Estimation

**Recommended Approach: Mesh Proximity Estimation**

Compute capsule radii by sampling distances to nearby mesh vertices, adapting to anatomical variations for non-standard character models (e.g., pointy breasts, unique body shapes).

**Algorithm Process:**

1. Sample points along bone axis
2. Query nearest mesh vertices for each sample using sparse octree
3. Compute statistical radius (median distance)
4. Apply anatomical constraints

**Spatial Acceleration:**

- **Sparse Octree**: O(log N) nearest neighbor queries
- **LCRS Optimization**: Memory-efficient tree representation
- **Leaf Size**: 16 vertices maximum per leaf node

**Mathematical Formulation:**

```
r_bone = median({distance(sample_i, nearest_mesh_vertex) | i ∈ [0, n_samples)})
```

**Parameters:**

- `n_samples`: Number of sample points along bone (default: 10)
- `octree_depth`: Maximum octree subdivision level
- `min_node_size`: Minimum octree node dimensions

**Key Benefits for Non-Standard Models:**

- Adapts to unique anatomical features automatically
- Prevents clipping issues with complex geometries
- Ensures precise capsule fitting to actual mesh contours

### Taper Calculation

**Primary Approach: Smooth Taper**

Create natural limb shapes with continuous radius transitions using smoothstep interpolation, reducing clipping issues during animation for complex geometries.

**Formula:**

```
r(t) = r_start × (1 - septic(t)) + r_end × septic(t)
where septic(t) = 6t⁷ - 35t⁶ + 84t⁵ - 70t⁴ + 20t³
```

**Key Characteristics:**

- **Continuity**: C³ continuous (septic) recommended for jerk minimization; ensures smooth acceleration changes
- **Shape**: S-curve radius profile for organic shapes
- **Naturalness**: Anatomically realistic for complex character models
- **Clipping Reduction**: Smooth transitions minimize animation artifacts

**Smoothstep Properties:**

- **Domain**: [0, 1] → [0, 1] with zero derivatives at boundaries
- **Smooth Interior**: Continuous second derivatives for fluid motion
- **Applications**: Natural limb tapering, muscle definition

**Benefits for Non-Standard Models:**

- Handles pointy anatomical features gracefully
- Maintains continuous capsule surfaces during deformation
- Reduces collision artifacts with complex geometries

### Proximity-Based Pinning

Automatic cloth attachment algorithm that pins garment vertices to nearby collision capsules.

#### Algorithm Overview

**Process:**

1. Iterate through all cloth vertices
2. For each vertex, find nearest capsule within threshold
3. Create attachment spring constraint
4. Apply high stiffness for rigid attachment

**Implementation:**

```cpp
void pinGarmentToCapsules(Simulation* sim, double pin_distance) {
    for (size_t i = 0; i < cloth_vertices.size(); ++i) {
        Vec3d vertex = cloth_vertices[i];

        // Find closest capsule within threshold
        TaperedCapsule* nearest_capsule = nullptr;
        double min_distance = pin_distance;

        for (auto& capsule : capsules) {
            double distance = capsule->distanceToPoint(vertex);
            if (distance < min_distance) {
                min_distance = distance;
                nearest_capsule = capsule.get();
            }
        }

        // Create attachment if within range
        if (nearest_capsule) {
            sim->attachment_springs.push_back(
                new AttachmentSpring(i, vertex, 1000.0)
            );
        }
    }
}
```

**Key Features:**

- **Zero Configuration**: Automatic attachment detection
- **Distance-Based**: Threshold controls attachment sensitivity
- **Single Assignment**: Each vertex attaches to nearest capsule only
- **High Stiffness**: Rigid attachment for stable simulation

**Critical Limitation**: This approach assumes clothing already fits the body. Real garments need center-outward inflation.

### Garment Inflation Algorithm

**Purpose**: Replace proximity pinning with natural cloth draping using center-outward inflation.

**Core Problem Solved**: Previous methods assumed garments fit perfectly. Real clothing must be "inflated" from the body's center outward, like inflating a noodle.

#### Inflation Algorithm Overview

**Four-Phase Process:**

1. **Initialization**: Collapse garment toward body centroid
2. **Expansion**: Apply repulsive forces with collision guidance
3. **Stabilization**: Create bone-based anchor points
4. **Refinement**: Allow natural drape under physics

**Mathematical Formulation:**

```
# Phase 1: Centroid Collapse
target_i = lerp(vertex_i, body_centroid, collapse_factor)

# Phase 2: Repulsive Expansion
force_i = sum(gaussian_repulsion(vertex_i, capsule_j) for all capsules j)
velocity_i += force_i * delta_time

# Phase 3: Bone-Based Anchoring
anchor_force_i = spring_force(vertex_i, bone_center, stiffness)
velocity_i += anchor_force_i * delta_time

# Phase 4: Collision-Guided Drape
collision_force_i = capsule_collision_response(vertex_i, nearest_capsule)
velocity_i += collision_force_i * delta_time
```

#### Bone-Based Anchor Points

**Anchor Point Structure:**

```cpp
struct AnchorPoint {
    std::string bone_name;        // Name of the skeleton bone
    Vec3d bone_position;          // Position of the bone center
    int bone_index;               // Index of the bone in skeleton
    double stiffness;             // Attachment strength (0-1)
    double max_distance;          // Maximum attachment distance
};
```

**Automatic Generation:**

```cpp
std::vector<AnchorPoint> generateBoneAnchors(const CapsuleRig& rig) {
    std::vector<AnchorPoint> anchors;
    const auto& skeleton = rig.getSkeleton();
    const auto& bones = skeleton.getBones();

    for (size_t i = 0; i < bones.size(); ++i) {
        const auto& bone = bones[i];
        Vec3d bone_center = (bone.start + bone.end) * 0.5;

        anchors.emplace_back(bone.name, bone_center, i, 0.8, 0.15);
    }

    return anchors;
}
```

#### Implementation Strategy

**Simplified Fitting:**

```cpp
class GarmentFitter {
public:
    void fitGarmentToRig(
        Simulation* sim,
        const CapsuleRig& rig
    ) {
        // Phase 1: Initialize at center
        collapseToCentroid(sim, rig);

        // Phase 2: Expand with physics
        expandWithCollisions(sim, rig);

        // Phase 3: Apply bone-based anchors
        auto anchors = generateBoneAnchors(rig);
        applyBoneAnchors(sim, rig, anchors);

        // Phase 4: Refine under gravity
        refineDrape(sim, rig);
    }
};
```

**Key Advantages:**

- **Natural Draping**: Clothing expands from body center outward
- **Simplified**: No garment classification or mesh analysis needed
- **Collision Integration**: Leverages existing capsule collision system
- **Bone-Based Anchoring**: Direct mapping from skeleton bones to cloth anchors


### Skeleton Retargeting

**Purpose**: Adapt skeletons between different topologies by treating the source skeleton as a mesh and fitting the target template using differentiable optimization.

**Core Innovation**: Reuse mesh-to-skeleton optimization pipeline for skeleton-to-skeleton retargeting.

#### Algorithm Overview

**Process:**

1. Convert source skeleton joints to mesh vertices
2. Generate connectivity faces from bone structure
3. Use existing differentiable optimizer with target template
4. Preserve anatomical proportions through optimization constraints

**Source-to-Mesh Conversion:**

```cpp
std::pair<Eigen::MatrixXd, Eigen::MatrixXi> skeletonToMesh(const Skeleton& skeleton) {
    // Convert joints to vertices
    Eigen::MatrixXd vertices(3, skeleton.joints.size());
    for (size_t i = 0; i < skeleton.joints.size(); ++i) {
        vertices.col(i) = skeleton.joints[i];
    }

    // Generate faces from bone connectivity
    std::vector<Eigen::Vector3i> faces;
    for (const auto& bone : skeleton.bones) {
        // Find joint indices
        auto start_it = std::find(skeleton.joints.begin(), skeleton.joints.end(), bone.start);
        auto end_it = std::find(skeleton.joints.begin(), skeleton.joints.end(), bone.end);

        if (start_it != skeleton.joints.end() && end_it != skeleton.joints.end()) {
            size_t start_idx = start_it - skeleton.joints.begin();
            size_t end_idx = end_it - skeleton.joints.begin();

            // Create triangular face with midpoint
            Vec3d midpoint = (bone.start + bone.end) * 0.5;
            vertices.conservativeResize(3, vertices.cols() + 1);
            vertices.col(vertices.cols() - 1) = midpoint;

            size_t mid_idx = vertices.cols() - 1;
            faces.push_back(Eigen::Vector3i(start_idx, end_idx, mid_idx));
        }
    }

    // Convert to matrix format
    Eigen::MatrixXi faces_matrix(3, faces.size());
    for (size_t i = 0; i < faces.size(); ++i) {
        faces_matrix.col(i) = faces[i];
    }

    return {vertices, faces_matrix};
}
```

**Retargeting Optimization:**

```cpp
Skeleton retargetSkeleton(const Skeleton& source, const Skeleton& target_template) {
    // Convert source to mesh
    auto [mesh_vertices, mesh_faces] = skeletonToMesh(source);

    // Use existing mesh-to-skeleton optimization
    return optimizeFromMesh(mesh_vertices, mesh_faces, target_template);
}
```

#### Loss Functions

**Multi-Objective Optimization:**

```cpp
double total_loss = w1 * coverage_loss +
                   w2 * plausibility_loss +
                   w3 * retargeting_constraints;

where:
- coverage_loss: How well skeleton covers the "mesh" (source skeleton)
- plausibility_loss: Anatomical correctness constraints
- retargeting_constraints: Proportion preservation from target template
```

**Retargeting Constraints:**

```cpp
double computeRetargetingLoss(
    const Skeleton& current,
    const Skeleton& source,
    const Skeleton& target_template
) {
    double loss = 0.0;

    // Preserve proportions from target template
    if (current.bones.size() == target_template.bones.size()) {
        for (size_t i = 0; i < current.bones.size(); ++i) {
            double current_length = current.bones[i].getLength();
            double template_length = target_template.bones[i].getLength();

            if (template_length > 0) {
                double ratio_diff = std::abs(current_length / template_length - 1.0);
                loss += ratio_diff * ratio_diff * 100.0; // Strong proportion penalty
            }
        }
    }

    return loss;
}
```

#### Key Benefits

**Reuses Existing Infrastructure:**
- Same optimization pipeline as mesh-to-skeleton fitting
- Consistent loss functions and gradient computation
- Proven convergence properties

**Handles Topology Mismatches:**
- Different bone hierarchies gracefully
- Missing or extra bones through optimization
- Anatomical adaptation via constraints

**Preserves Important Properties:**
- Bone length ratios from target template
- Joint angle constraints
- Anatomical plausibility

**Applications:**
- VRMA-to-cloth skeleton compatibility
- Cross-character animation transfer
- Skeleton adaptation for cloth simulation
