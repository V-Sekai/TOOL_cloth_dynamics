# Spring Bone Optimization

## Gravity-Only Simulation

Spring bone systems use gravity-only physics simulation for stability and simplicity.

### Implementation

**Gravity-Based Dynamics**:
- Spring bones use simple gravity forces only
- Deterministic physics behavior
- Stable integration with optimization processes

**Configuration**:
```cpp
struct SpringBoneConfig {
    bool enable_gravity = true;        // Gravity-only simulation
    double gravity_strength = -9.81;   // Standard gravity acceleration
    bool deterministic_mode = true;    // Stable for optimization
    // Simplified parameter set
};
```

### Optimization Integration

**Always-Active Design**:
- Gravity simulation remains active during optimization
- Deterministic behavior with consistent gravity forces
- No phase-dependent enable/disable complexity
- Stable gradients throughout optimization process

**Technical Benefits**:
- Simplified state management
- Predictable physics behavior
- Reduced computational overhead
- Improved convergence reliability

### Design Principles

**Simplicity First**:
- Minimal parameter space for stability
- Deterministic physics simulation
- Clean integration with existing systems
- Focus on core functionality over complexity

**Optimization Compatibility**:
- Gravity forces don't interfere with loss computation
- Consistent behavior across optimization iterations
- Reliable gradient computation
- Stable convergence properties
