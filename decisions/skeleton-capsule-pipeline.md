# Skeleton-to-Capsule Pipeline

## Overview

The Skeleton-to-Capsule Pipeline provides a systematic approach for converting skeletal representations into collision primitives suitable for cloth simulation. This specification defines the core concepts, data structures, algorithms, and integration patterns for implementing bidirectional skeleton-capsule transformations, including skeleton-to-skeleton retargeting for animation compatibility.

## Recommended Design Paths

Based on code analysis and practical requirements for cloth simulation:

### Core Algorithms
- **Radius Estimation**: Mesh proximity with sparse octree (handles non-standard anatomy)
- **Taper Calculation**: C³ septic interpolation (jerk minimization, smooth transitions)
- **Garment Attachment**: Proximity-based pinning (implemented, reliable for cloth draping)
- **Optimization**: Template-based fitting (production-ready)
- **Skeleton Retargeting**: Source-as-mesh optimization (VRMA-to-cloth compatibility)

### Testing Infrastructure
- **Dataset Integration**: Large-scale testing framework (validation, benchmarking, edge cases)

## Table of Contents

### Core Concepts
- [**System Architecture**](system-architecture.md) - Pipeline design and component relationships
- [**Data Structures**](data-structures.md) - Bone, Skeleton, and CapsuleRig specifications
- [**Pipeline Stages**](pipeline-stages.md) - Forward and inverse transformation processes

### Technical Specifications
- [**File Formats**](file-formats.md) - OBJ format conventions and asset pairing
- [**Algorithms**](algorithms.md) - Radius estimation, tapering, and proximity methods
- [**API Design**](api-design.md) - Class interfaces and method specifications

### Integration & Applications
- [**DiffCloth Integration**](diffcloth-integration.md) - Simulation system integration
- [**Demo Scenes**](demo-scenes.md) - Standard test cases and expected behaviors

### Advanced Topics
- [**Loss Functions**](loss-functions.md) - Optimization objectives and constraints

### Appendices
- [**Future Extensions**](future-extensions.md) - Advanced capabilities and research directions
- [**References**](references.md) - External specifications and related work
