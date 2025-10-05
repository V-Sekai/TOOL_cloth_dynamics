# System Architecture

## Overview

The Skeleton-to-Capsule Pipeline implements a bidirectional transformation system between skeletal representations and collision primitives for cloth simulation. The architecture supports both forward (skeleton-to-capsule) and inverse (mesh-to-skeleton) transformations with consistent data structures and algorithmic approaches.

## Pipeline Architecture

### Forward Direction: Skeleton → Capsules

The forward pipeline converts skeletal line segments into tapered capsule collision primitives:

```
Skeleton (OBJ) → Bone Extraction → Radius Estimation → Capsule Generation → Simulation Integration
     ↓                ↓                  ↓                    ↓                    ↓
   v/l data      Bone structures    Mesh proximity      TaperedCapsules      Collision detection
   topology      length/axis        O(log N) queries    KHR_implicit_shapes  Cloth physics
```

**Key Stages:**
1. **Skeleton Loading**: Parse OBJ vertices and line segments into Bone structures
2. **Radius Estimation**: Compute capsule dimensions using mesh proximity queries
3. **Capsule Generation**: Create tapered capsules aligned with bone axes
4. **Simulation Integration**: Add capsules as collision primitives

### Inverse Direction: Mesh → Skeleton

The inverse pipeline generates skeletal structures from mesh geometry using optimization:

```
Template + Mesh → Pose Optimization → Coverage Fitting → Skeleton Generation → Validation
     ↓            ↓                   ↓                  ↓                    ↓
Initialization  Gradient descent   Volume filling     Bone placement      Quality metrics
Humanoid base   Differentiable     Sparse octree      Joint positions     Anatomical checks
```

### Skeleton Retargeting: Skeleton → Skeleton

The retargeting pipeline adapts skeletons between different topologies by treating the source skeleton as a mesh:

```
Source Skeleton → Mesh Conversion → Template Fitting → Retargeted Skeleton → Validation
      ↓                ↓                   ↓                    ↓                ↓
Vertex extraction  Connectivity      Optimization       Pose transfer     Topology checks
Joint positions    Bone topology     Gradient descent   Proportion        Edge matching
                   generation        Sparse octree      preservation      cloth-fit ready
```

**Key Stages:**
1. **Template Initialization**: Start with humanoid base skeleton
2. **Optimization**: Gradient-based fitting to mesh surface and volume
3. **Validation**: Anatomical constraint checking and quality assessment

## Core Components

### Data Flow Components

| Component | Responsibility | Key Operations |
|-----------|----------------|----------------|
| **SkeletonLoader** | OBJ file parsing and bone extraction | Vertex/line parsing, topology validation |
| **CapsuleGenerator** | Bone-to-capsule conversion | Radius estimation, tapering, alignment |
| **SparseOctree** | Spatial acceleration for mesh queries | O(log N) nearest neighbor searches |
| **CapsuleRig** | Rig management and simulation integration | Primitive collection, pinning, rendering |
| **RadiusEstimator** | Adaptive capsule sizing | Mesh proximity, statistical analysis |

### Algorithmic Components

| Component | Purpose | Characteristics |
|-----------|---------|----------------|
| **Bone Extraction** | Convert line segments to structured bones | Topology preservation, axis calculation |
| **Mesh Proximity** | Distance queries for radius estimation | Spatial partitioning, efficient search |
| **Taper Calculation** | Natural limb shape modeling | Linear/smooth interpolation, anatomical ratios |
| **Proximity Pinning** | Automatic cloth attachment | Zero-configuration, distance-based selection |

## Design Principles

### Asset Pairing Constraint
- **Definition**: Skeleton and mesh files must be co-located and structurally compatible
- **Purpose**: Ensures consistent topology between collision and visual representations
- **Implementation**: Directory-based pairing with validation checks

### Flexible Topology Support
- **Definition**: Support for arbitrary skeletal structures beyond humanoid
- **Purpose**: Enable diverse character types (humanoid, custom)
- **Implementation**: Generic bone representation with extensible naming conventions

### Performance Optimization
- **Definition**: Spatial data structures for real-time cloth simulation
- **Purpose**: Maintain interactive framerates with complex skeletal rigs
- **Implementation**: Octree-based acceleration, cached computations

### Extensible Architecture
- **Definition**: Modular design with clean separation of concerns
- **Purpose**: Enable future enhancements without architectural changes
- **Implementation**: Interface-based design, dependency injection patterns
