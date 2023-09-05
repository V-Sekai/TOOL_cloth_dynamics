# Summary of DiffCloth Applications

This section presents various cloth-related applications that can benefit from the proposed differentiable cloth simulation with dry frictional contact. The experiments are conducted with multiple random seeds and use the same random seed set for all methods.

## System Identification

Two system identification examples are presented: "T-shirt" and "Sphere".

- In the "T-shirt" example, the objective is to estimate a material parameter in the cloth and identify the wind model parameters from the motion data. All three methods (L-BFGS-B, CMA-ES, and (1+1)-ES) succeed in optimizing system parameters leading to motion sequences visually identical to the given input, but L-BFGS-B converges much faster due to the extra knowledge of gradients.

- In the "Sphere" example, the objective is to match the motion sequence of a cloth interacting with a sphere by estimating the frictional coefficient between the sphere and the cloth. All methods can optimize to a frictional coefficient that generates a motion sequence visually identical to the given input.

## Robot-assisted Dressing

Two examples demonstrate the usage of gradients in robot-assisted dressing: "Hat" and "Sock". In both examples, the objective is to find trajectories for a kinematic robotic manipulator to put on the hat or the sock. With the gradient information at hand, L-BFGS-B optimizer is used to tune the parameters of the trajectories and it converges substantially faster than the gradient-free baselines to a better solution.

## Inverse Design

The "Dress" application aims to optimize cloth material parameters in a dress so that its dynamic motion can satisfy certain design intents. Specifically, the material parameters of a twirl dress are optimized so that after the dress spins, the apex angle of the cone-like dress agrees with the target value. L-BFGS-B achieves better optimized results using fewer time steps.

## A Real-to-Sim Example

In the "Flag" example, the real-world motion sequence captured on a flag flapping in the wind is used to reconstruct a digital twin of the scene in simulation. This includes not only estimating the material parameters of the flag but also modeling the unknown wind condition at the capture time. L-BFGS-B achieves a lower final loss.

## Hat Controller

An advanced "Hat" task is presented where the objective is to train a generalizable closed-loop controller that can put on the hat from a random starting position sampled from a fixed-radius hemisphere around the head. Both gradient-based method and PPO reach a similar final loss, but with the differentiable simulation framework, the gradient-based method reaches its final loss with an 85× speedup.

### Material Parameter Estimation Tool

To create this tool, you would need to write a script that takes as input the motion data of a virtual clothing item. The script should then use this data to estimate the material parameters of the clothing.

```gdscript
# Sample script for estimating material parameters
func estimate_material_parameters(motion_data):
    # Your code here to process the motion data and estimate parameters
    return estimated_parameters
```

### Clothing Interaction Modeling Tool

This tool would involve creating a script that models interactions between clothing and an avatar's body. This could be done by defining collision shapes for the avatar and the clothing, and then writing code to handle these collisions.

```gdscript
# Sample script for modeling clothing interactions
func model_clothing_interactions(clothing, avatar):
    # Your code here to define collision shapes and handle collisions
```

### Cloth Property Optimization Tool

This tool would involve writing a script that optimizes cloth material parameters to achieve a specific fit or look. This could be done using various optimization algorithms.

```gdscript
# Sample script for cloth property optimization
func optimize_cloth_properties(clothing, target_fit):
    # Your code here to adjust the properties of the clothing to achieve the target fit
```

### Real-to-Sim Motion Recreation Tool

To create this tool, you would need to write a script that recreates real-world motion in a simulation. This could involve using physics simulations or other techniques.

```gdscript
# Sample script for real-to-sim motion recreation
func recreate_motion(real_world_data):
    # Your code here to recreate the real-world motion in the simulation
```

### Character Animation Matching Tool

The tool involves writing a script that matches an existing animation to a different character. This is achieved by mapping the joint movements of the source character to the target character, considering the stiffness of skin and cloth fabrics.

```python
def match_animation(source_character, target_character, animation_data):
    map_animation(source_character, target_character, animation_data)
```

In the `map_animation` function, each frame of the animation is iterated over, the transformation from the source character's pose to the target character's pose is calculated, and this transformation is applied to the target character.

```python
def map_animation(source_character, target_character, animation_data):
    # Iterate over each frame in the animation data
    for frame in animation_data:
        # Calculate the transformation from source to target character's pose
        transformation = calculate_transformation(source_character.pose(frame), target_character.pose(frame))

        # Apply the transformation to the target character
        apply_transformation(target_character, transformation)
```

This approach ensures that the trajectories of the skeleton are correctly mapped from the source character to the target character, taking into account the stiffness of skin and cloth fabrics. The actual implementation will depend on the specifics of your animation system.

---

| Application                        | Input                                                                           | Output                                                                                                                                                |
| ---------------------------------- | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| System Identification (T-shirt)    | Motion data of a T-shirt                                                        | Estimated material parameter in the cloth and identified wind model parameters                                                                        |
| System Identification (Sphere)     | Motion sequence of a cloth interacting with a sphere                            | Estimated frictional coefficient between the sphere and the cloth                                                                                     |
| Robot-assisted Dressing (Hat)      | Kinematic robotic manipulator                                                   | Optimized trajectories for the manipulator to put on the hat                                                                                          |
| Robot-assisted Dressing (Sock)     | Kinematic robotic manipulator                                                   | Optimized trajectories for the manipulator to put on the sock                                                                                         |
| Inverse Design (Dress)             | Dynamic motion of a dress                                                       | Optimized cloth material parameters so that the apex angle of the cone-like dress agrees with the target value after spinning                         |
| A Real-to-Sim Example (Flag)       | Real-world motion sequence of a flag flapping in the wind                       | Reconstructed digital twin of the scene in simulation, including estimated material parameters of the flag and modeled wind condition at capture time |
| Hat Controller                     | Random starting position sampled from a fixed-radius hemisphere around the head | Trained generalizable closed-loop controller that can put on the hat                                                                                  |
| Material Parameter Estimation Tool | Motion data of a virtual clothing item                                          | Estimated material parameters of the clothing                                                                                                         |
| Clothing Interaction Modeling Tool | Clothing and avatar's body                                                      | Modeled interactions between clothing and an avatar's body                                                                                            |
| Cloth Property Optimization Tool   | Clothing and target fit                                                         | Optimized cloth material parameters to achieve a specific fit or look                                                                                 |
| Real-to-Sim Motion Recreation Tool | Real-world motion data                                                          | Recreated real-world motion in a simulation                                                                                                           |
| Clothing Controller Training Tool  | Training data                                                                   | Trained controller to perform complex tasks such as dressing an avatar                                                                                |
| Character Animation Matching Tool  | Source character, target character, animation data                              | Matched existing animation to a different character considering the stiffness of skin and cloth fabrics                                               |
