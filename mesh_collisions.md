# Unified Approach to Arbitrary Mesh Collision and Differential Cloth with FCL

The DiffCloth repository is currently focused on implementing animated cloth simulation for 3D avatars. A key challenge that we're facing is the lack of collision detection between the cloth and an arbitrary 3D mesh, which is crucial for creating realistic avatar simulations.

## GitHub Issue Discussion

A recent [GitHub issue](https://github.com/omegaiota/DiffCloth/issues/10) discusses enhancing the DiffCloth cloth simulation system by adding more complex obstacles. Currently, collision detection only supports point-to-mesh tests between the cloth and obstacles. The author inquires about detecting collisions when a cloth triangle edge penetrates an obstacle. While only nodal contacts are directly supported, the responder suggests that edge-edge and vertex-face collisions could still be detected and treated as nodal constraints. Implementing collision detection for a cube primitive is also discussed. Additionally, the possibility of using arbitrary meshes from .obj files for obstacles rather than just defined primitives is raised. This conversation underscores the need for enhancing DiffCloth's collision handling for more sophisticated simulations.

## Possible Solutions

One potential solution could be to implement point-triangle collision detection between each cloth vertex and avatar mesh triangle. However, this approach might be computationally expensive for large meshes. An alternative could be using an acceleration structure like a Bounding Volume Hierarchy (BVH) tree, which can significantly enhance performance by avoiding unnecessary collision checks.

## Insights from Contributors

Our contributor, alelordelo, has previously implemented cloth-mesh collision, albeit in Swift, not C++. Despite the difference in language, their experience and insights could prove invaluable. Additionally, our contributor fire has successfully streamlined the simulation code and made it compatible with CMake on Windows.

Alelordelo has also shared two possible methods for collision detection: triangle-triangle or particle-particle. Each method has its own advantages and disadvantages, and the choice would depend on the specific requirements of your project.

## Resources and Support

Here's the link to the library: [Flexible Collision Library (FCL)](https://github.com/flexible-collision-library/fcl)

Implementing body-cloth collision is another aspect that needs attention. While I currently don't have the bandwidth to delve into this at the moment, you might find this issue helpful: [Body-Cloth Collision Issue](https://github.com/omegaiota/DiffCloth/issues/10)

If you decide to proceed with adding arbitrary mesh Continuous Collision Detection (CCD) to the library, I'm here to provide support. I can answer any questions you might have and assist you through Zoom calls. Please feel free to reach out if you need further assistance or guidance.