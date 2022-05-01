# Mujoco_PPO

MuJoCo is a general purpose physics engine that aims to facilitate research and development in robotics. It stands for Multi-Joint dynamics with contact. Mujoco has different environments from which we use Hopper. Hopper has a 11-dimensional state space, that is position and velocity of each joint. The initial states are uniformly randomized. The action is a 3-dimensional continuous space. This environment is terminated when the agent falls down.

![hopper](https://user-images.githubusercontent.com/68981504/166164551-03e02cd5-3550-4c27-b5dd-e880bab2b681.png)

## Mujoco Installation
We'll be using mujoco210 in this project. This page contains the mujoco210 releases: https://github.com/deepmind/mujoco/releases/tag/2.1.0 Download the distribution compatible to your OS and extract the downloaded mujoco210 directory into ~/.mujoco/.

## We use the CPU only machine for this project, for installing mujoco on a CPU only machine do as follows:

Create the conda environment using `conda create --name mujoco --file environment.yml`.

Set the conda environment variable to: `LD_LIBRARY_PATH=/usr/local/pkgs/cuda/latest/lib64:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia`.

You can change the conda environment variable using `conda env config vars set LD_LIBRARY_PATH=....`

## Install the required packages to run mujoco environment
```
!apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common

!apt-get install -y patchelf
```
```
!pip3 install gym
!pip install free-mujoco-py
```
## Experiment Setup
1. Implement PPO with step-wise style.
2. Combine the benefits of model-based and model-free methods together.
3. Experiment with 3 hidden layers MLP for both actor and critic NN, with Tanh and RELU as activation function respectively.
4. Test with performance of different hidden units(32, 64, 128, 256, 512).
5. Compare the performance with and without the reward scaling technique.

## Experiment Report
See Report.pdf
