# References

## Standards and Specifications

### File Formats

**OBJ Format Specification**
- **Reference**: http://paulbourke.net/dataformats/obj/
- **Usage**: Skeleton and mesh data interchange
- **Elements**: Vertices (joints), lines (bones), faces (mesh triangles)

**glTF KHR_implicit_shapes Extension**
- **Reference**: https://github.com/eoineoineoin/glTF_Physics/blob/master/extensions/2.0/Khronos/KHR_implicit_shapes/schema/glTF.KHR_implicit_shapes.shape.capsule.schema.json
- **Purpose**: Standardized capsule shape parameters
- **Parameters**: height, radiusTop, radiusBottom

### Animation Standards

**VRM Spring Bone Specification**
- **Reference**: https://vrm.dev/en/
- **Purpose**: Procedural animation for dynamic elements
- **Components**: Spring chains, stiffness, damping parameters

### Research Publications

**DiffCloth: Differentiable Cloth Simulator**
- **Reference**: https://people.csail.mit.edu/liyifei/publication/diffcloth-differentiable-cloth-simulator/
- **Relevance**: Differentiable cloth simulation framework
- **Techniques**: Gradient-based optimization, collision handling

### Datasets

**Articulation-XL2.0 Dataset**
- **Reference**: https://huggingface.co/datasets/Seed3D/Articulation-XL2.0
- **Contents**: Large-scale articulated object collection
- **Usage**: Training data for skeleton generation algorithms

## Related Work

### Collision Detection

**Tapered Capsule Primitives**
- Used in character animation and cloth simulation
- Efficient collision detection with smooth surfaces
- Variable radius for anatomical accuracy

### Procedural Animation

**Spring-Mass Systems**
- Lagrangian mechanics for dynamic elements
- Stiffness/damping parameters for motion control
- Hierarchical chains for complex motion

### Skeleton Fitting

**Optimization-Based Approaches**
- Gradient descent for pose estimation
- Volume filling constraints
- Anatomical priors for natural poses

## Implementation Notes

### Performance Considerations

**Spatial Acceleration Structures**
- Octree-based nearest neighbor queries
- Memory-efficient sparse representations
- O(log N) complexity for mesh proximity

### Numerical Stability

**Collision Resolution**
- Constraint-based contact handling
- Impulse-based collision response
- Stability analysis for timestep selection

### Memory Management

**Resource Lifetime**
- RAII principles for automatic cleanup
- Shared ownership for simulation integration
- Progressive loading for large datasets
