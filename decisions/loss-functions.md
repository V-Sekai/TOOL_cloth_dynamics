# Loss Functions

## Optimization Objectives

### Capsule Fitting Loss

The capsule fitting loss function optimizes capsule parameters to achieve tight geometric conformity with target mesh geometry.

#### Surface Distance Loss

**Purpose**: Prevent capsule penetration into mesh surfaces while maintaining proximity.

**Mathematical Formulation**:
```
L_surface = Σᵢ max(0, dᵢ)²
where dᵢ = distance from capsule surface to nearest mesh vertex
```

**Characteristics**:
- Penalizes surface intersections
- Allows controlled proximity
- Smooth and differentiable

#### Volume Coverage Loss

**Purpose**: Ensure capsules adequately fill occupied mesh volume using sparse spatial sampling.

**Mathematical Formulation**:
```
L_volume = Σⱼ (1 - occupancyⱼ) × weightⱼ
where occupancyⱼ = capsule coverage of spatial sample j
```

**Spatial Sampling**:
- Sparse octree voxel evaluation
- Mesh vertices as volume proxies
- Efficient O(log N) queries

#### Regularization Loss

**Purpose**: Prevent excessive capsule expansion while maintaining shrinkwrap behavior.

**Mathematical Formulation**:
```
L_regularization = Σᵢ (rᵢ - r_target)² + λ × Σᵢ smoothness_penaltyᵢ
```

**Regularization Terms**:
- Radius bounds enforcement
- Inter-capsule smoothness
- Anatomical plausibility

### Parameter Optimization

#### Capsule Parameters (12 per capsule)

**Geometric Parameters**:
- Position: 3 parameters (x, y, z center)
- Orientation: 6 parameters (6D rotation representation)
- Shape: 3 parameters (radius_top, radius_bottom, height)

**Total Parameter Space**: 12N for N capsules

#### Optimization Strategy

**Gradient-Based Fitting**:
- Differentiable loss computation
- Bidirectional optimization (surface + volume)
- Constraint-aware parameter updates

### Constraint Hierarchy

#### High Priority Constraints

**Joint Angle Limits**:
- Enforce biologically plausible ranges
- Prevent impossible anatomical configurations
- Joint-specific angle constraints

**Surface Continuity**:
- Minimize gaps between adjacent capsules
- Ensure smooth surface transitions
- Prevent clipping artifacts

#### Medium Priority Constraints

**Orientation Optimization**:
- Allow capsules to rotate for better fit
- 6D rotation parameterization
- Improved collision accuracy

**Connectivity Preservation**:
- Maintain bone length relationships
- Prevent unrealistic deformations
- Anatomical proportion constraints

#### Low Priority Constraints

**Symmetry Enforcement**:
- Bilateral symmetry for paired structures
- Left-right consistency
- Aesthetic balance optimization

**Proportional Scaling**:
- Maintain anatomical ratios
- Species-specific proportions
- Growth-based scaling constraints

### Implementation Characteristics

**Differentiable Computation**:
- All loss components support gradient computation
- Efficient automatic differentiation
- Stable optimization convergence

**Sparse Evaluation**:
- Octree-based spatial acceleration
- Memory-efficient large mesh handling
- Scalable to high-resolution geometry
