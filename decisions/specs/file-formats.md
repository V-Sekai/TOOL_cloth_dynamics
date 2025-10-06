# File Formats

## Supported Formats

### OBJ Format for Skeletons

**Standard Elements**:
- `v x y z` - Vertex coordinates defining joint positions
- `l v1 v2` - Line segments defining bone connectivity between joints

**Basic Structure**:
```
# Joint positions as vertices
v -0.000397 0.096079 4.348223  # Joint 0: pelvis center
v 0.000243 0.058761 4.680330   # Joint 1: lower spine
v 0.000910 0.029228 5.721966   # Joint 2: upper spine

# Bone connectivity as lines
l 1 2  # Bone: pelvis → lower spine
l 2 3  # Bone: lower spine → upper spine
l 2 4  # Bone: spine → right shoulder
```

**Requirements**:
- Vertices must be numbered sequentially starting from 1
- Line segments reference vertex indices
- No face elements required (line-only OBJ files supported)

### OBJ Format for Meshes

**Standard Elements**:
- `v x y z` - Vertex coordinates
- `vn x y z` - Vertex normals (optional)
- `f v1 v2 v3` - Triangular faces

**Usage in Pipeline**:
- Provides geometric reference for radius estimation
- Used in proximity queries for capsule sizing
- Supports both character meshes and cloth garments

### Directory Organization

**Recommended Structure**:
```
assets/
├── skeletons/          # Skeleton OBJ files
├── meshes/            # Character/cloth meshes
├── templates/         # Reference skeletons
└── animations/        # Motion data
```

### Extended OBJ Format

**Optional Metadata Comments**:
```
# BONE name=spine v1=1 v2=2 parent=0
# BONE name=upper_arm_L v1=2 v2=4 parent=1
```

**Benefits**:
- Semantic bone identification
- Hierarchical relationship specification
- Enhanced debugging and visualization

## Output Formats

### Generated Skeleton OBJ

**Purpose**: Optimized skeletal structure for simulation use.

**Content**:
- Fitted joint positions from optimization
- Preserved topological connectivity
- Standard OBJ format for broad compatibility

### Capsule Visualization OBJ

**Purpose**: Triangle mesh representation of capsules for debugging and visualization.

**Structure**:
```
# Capsule surface mesh
v x y z     # Triangle vertices
f 1 2 3     # Triangular faces

# Parameter metadata
# CAPSULE bone=0 center=[x,y,z] radius_top=r1 radius_bottom=r2 length=l
```

**Usage**:
- Visual inspection of capsule placement
- Debugging geometric issues
- Documentation and presentation
