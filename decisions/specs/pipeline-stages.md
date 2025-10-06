# Pipeline Stages

## Processing Pipeline

The Skeleton-to-Capsule Pipeline consists of four primary stages that transform skeletal representations into collision-ready capsule primitives.

### Stage 1: Skeleton Loading

**Purpose**: Parse skeletal structure from standard file formats into internal data structures.

**Input**: OBJ file containing vertices (joints) and lines (bones)

**Output**: Skeleton structure with bone segments and joint positions

**Process**:
1. Parse vertex coordinates as joint positions
2. Extract line segments as bone connectivity
3. Construct Bone structures with start/end points
4. Validate topological consistency
5. Return complete Skeleton object

**File Format Requirements**:
```
# Vertices define joint positions
v x y z        # Joint coordinates

# Lines define bone connectivity
l v1 v2        # Bone between joints v1, v2
```

### Stage 2: Capsule Generation

**Purpose**: Convert bone segments into tapered capsule collision primitives.

**Input**: Skeleton structure, optional mesh for radius estimation

**Output**: Collection of TaperedCapsule primitives

**Radius Estimation Strategies**:

**Fixed Radius**:
- Assign constant radius to all capsules
- Computationally efficient
- Suitable for prototyping and uniform anatomy

**Mesh Proximity**:
- Sample distances to nearby mesh vertices
- Use spatial acceleration structures
- Adapt to anatomical variations

**Taper Calculation**:
- Linear interpolation between bone endpoints
- Smooth curves for natural limb shapes
- Anatomical constraints for realism

### Stage 3: Garment Fitting

**Purpose**: Apply bone-based garment anchors for natural cloth draping behavior.

**Input**: CapsuleRig + cloth mesh

**Output**: Cloth simulation with bone-based anchor constraints

**Process**:
1. Generate bone-based anchor points from skeleton
2. Apply anchor constraints to cloth vertices
3. Refine drape under physics simulation

**Anchor Types**:
- **Bone anchors**: Direct mapping from skeleton bones to cloth attachment points

### Stage 4: Garment Attachment

**Purpose**: Attach cloth garments to capsule collision primitives for realistic draping.

**Input**: CapsuleRig + cloth mesh

**Output**: Cloth simulation with stable capsule attachments

**Recommended Approach**: Proximity-based pinning provides reliable attachment for cloth simulation scenarios.

**Attachment Process**:
1. **Distance Evaluation**: Measure cloth vertex distances to capsules
2. **Threshold Filtering**: Select vertices within attachment range
3. **Spring Creation**: Generate attachment constraints with high stiffness
4. **Simulation Integration**: Add constraints to physics simulation

**Garment Classification**:
- **SHIRT**: Upper body focus, shoulder anchors
- **PANTS**: Lower body focus, waist/hip anchors
- **DRESS**: Full body coverage, multiple anchor zones
- **SKIRT**: Lower body only, waist anchor
- **JACKET**: Layered garment with specific attachment patterns

### Stage 5: Simulation Integration

**Purpose**: Incorporate capsule primitives and fitted garments into the physics simulation environment.

**Input**: CapsuleRig with fitted garment

**Output**: Active simulation with natural cloth-capsule interactions

**Integration Steps**:
1. Register capsules as collision primitives
2. Configure material properties
3. Apply bone-based garment anchors
4. Initialize physics constraints
5. Enable real-time collision detection and cloth dynamics

### Stage 6: Skeleton Retargeting

**Purpose**: Adapt skeletons between different topologies for animation compatibility.

**Input**: Source skeleton + target skeleton template

**Output**: Retargeted skeleton with source pose adapted to target structure

**Process**:
1. Convert source skeleton to mesh representation
2. Use differentiable optimization to fit target template
3. Preserve anatomical proportions and constraints
4. Validate retargeted skeleton topology

**Retargeting Algorithm**:
- **Source Conversion**: Transform skeleton joints into mesh vertices
- **Template Fitting**: Optimize target skeleton to match source "mesh"
- **Constraint Preservation**: Maintain bone length ratios and joint angles
- **Topology Adaptation**: Handle different bone hierarchies gracefully

**Applications**:
- Cross-character skeleton transfer
- Animation retargeting for cloth simulation

## Data Flow

**Primary Pipeline:**
```
OBJ File → Skeleton Loading → Capsule Generation → Garment Fitting → Simulation
     ↓             ↓                  ↓                    ↓              ↓
   v/l data    Bone structures   TaperedCapsules    Bone-based anchors   Collision detection
   topology    length/axis       KHR_implicit_shapes  Direct mapping     Cloth physics
```

**Retargeting Pipeline:**
```
Source Skeleton → Mesh Conversion → Template Fitting → Retargeted Skeleton → Capsule Generation
       ↓                ↓                   ↓                    ↓                ↓
   Joint positions   Connectivity       Optimization       Pose transfer     Collision primitives
   Bone topology     generation         Sparse octree      Proportion        Cloth simulation
                      differentiable    gradient descent   preservation      compatibility
```

## Quality Assurance

### Validation Checks

**Topological Consistency**:
- Bone connectivity forms valid graph structure
- Joint positions are consistent across bones
- No orphaned or duplicate elements

**Geometric Validity**:
- Capsule dimensions within reasonable bounds
- No degenerate primitives (zero volume)
- Proper alignment with skeletal structure

**Performance Metrics**:
- Initialization time within acceptable limits
- Memory usage scales appropriately
- Real-time collision detection performance
