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

## Material Parameter Estimation Tool

To create this tool, you would need to write a script that takes as input the motion data of a virtual clothing item. The script should then use this data to estimate the material parameters of the clothing.

## Clothing Interaction Modeling Tool

This tool would involve creating a script that models interactions between clothing and an avatar's body. This could be done by defining collision shapes for the avatar and the clothing, and then writing code to handle these collisions.

## Cloth Property Optimization Tool

This tool would involve writing a script that optimizes cloth material parameters to achieve a specific fit or look. This could be done using various optimization algorithms.

## Real-to-Sim Motion Recreation Tool

To create this tool, you would need to write a script that recreates real-world motion in a simulation. This could involve using physics simulations or other techniques.

## Chart

| Application                     | Input                                                     | Output                                                                                                                                                |
| ------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| System Identification (T-shirt) | Motion data of a T-shirt                                  | Estimated material parameter in the cloth and identified wind model parameters                                                                        |
| System Identification (Sphere)  | Motion sequence of a cloth interacting with a sphere      | Estimated frictional coefficient between the sphere and the cloth                                                                                     |
| Robot-assisted Dressing (Hat)   | Kinematic robotic manipulator                             | Optimized trajectories for the manipulator to put on the hat                                                                                          |
| Robot-assisted Dressing (Sock)  | Kinematic robotic manipulator                             | Optimized trajectories for the manipulator to put on the sock                                                                                         |
| Inverse Design (Dress)          | Dynamic motion of a dress                                 | Optimized cloth material parameters so that the apex angle of the cone-like dress agrees with the target value after spinning                         |
| A Real-to-Sim Example (Flag)    | Real-world motion sequence of a flag flapping in the wind | Reconstructed digital twin of the scene in simulation, including estimated material parameters of the flag and modeled wind condition at capture time |
