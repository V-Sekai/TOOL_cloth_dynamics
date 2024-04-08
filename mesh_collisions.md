# Unified Approach to Arbitrary Mesh Collision and Differential Cloth with FCL

The DiffCloth repository is currently focused on implementing animated cloth simulation for 3D avatars. A key challenge that we're facing is the lack of collision detection between the cloth and an arbitrary 3D mesh, which is crucial for creating realistic avatar simulations.

## Possible Solutions

One potential solution could be to implement point-triangle collision detection between each cloth vertex and avatar mesh triangle. However, this approach might be computationally expensive for large meshes. An alternative could be using an acceleration structure like a Bounding Volume Hierarchy (BVH) tree, which can significantly enhance performance by avoiding unnecessary collision checks.

## Insights from Contributors

Our contributor, alelordelo, has previously implemented cloth-mesh collision, albeit in Swift, not C++. Despite the difference in language, their experience and insights could prove invaluable. Additionally, our contributor fire has successfully streamlined the simulation code and made it compatible with CMake on Windows.

Alelordelo has also shared two possible methods for collision detection: triangle-triangle or particle-particle. Each method has its own advantages and disadvantages, and the choice would depend on the specific requirements of your project.

## Resources and Support

Here's the link to the library: [Flexible Collision Library (FCL)](https://github.com/flexible-collision-library/fcl)

Implementing body-cloth collision is another aspect that needs attention. While @omegaiota currently don't have the bandwidth to delve into this at the moment, you might find this issue helpful: [Body-Cloth Collision Issue](https://github.com/omegaiota/DiffCloth/issues/10)

If you decide to proceed with adding arbitrary mesh Continuous Collision Detection (CCD) to the library, @omegaiota is here to provide support. @omegaiota can answer any questions you might have and assist you through Zoom calls. Please feel free to reach out if you need further assistance or guidance.
