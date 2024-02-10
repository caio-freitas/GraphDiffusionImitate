# GraphDiffusionImitate

[Caio Freitas](https://caio-freitas.github.io/)

## Installation

To install the dependencies, run 
```
$ mamba env create -f conda_environment.yaml
```

### On using Robosuite

When using robosuite, the environments are based on v1.4.1 but using the direct joint values as well in the actions. To add that to the robosuite environments, the file `robosuite/robosuite/models/robots/robot.py` should be modified to include (flag as True) the `robot{i}_joint_pos` in the `actives` list in the `setup_observables()` method (line 204).

### Acknowledgements
This repo relies on the following existing codebases / datasets:
* Dataset from lift, square and transport tasks from [robosuite](https://robomimic.github.io/docs/datasets/robosuite.html)
* Franka Kitchen environment and dataset, as well as benchmark model from [diffusion_policy](https://github.com/columbia-ai-robotics/diffusion_policy)