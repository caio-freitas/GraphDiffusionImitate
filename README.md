# GraphDiffusionImitate

[Caio Freitas](https://caio-freitas.github.io/)

## Installation

To install the dependencies, run 
```
$ mamba env create -f conda_environment.yaml
```

### Downloading training data

To download the training data used (from the Robomimic project), use the following commands from within the `robomimic/robomimic/scrips` directory:

```bash
# Download from robomimic
python download_datasets.py --tasks lift square transport --dataset_types ph mh --hdf5_types low_dim
# Move to /data directory
cd ../../datasets
mv * ~/workspace/GraphDiffusionImitate/data/
```

### Acknowledgements
This repo relies on the following existing codebases / datasets:
* Dataset from lift, square and transport tasks from [robosuite](https://robomimic.github.io/docs/datasets/robosuite.html)
* Franka Kitchen environment and dataset, as well as benchmark model from [diffusion_policy](https://github.com/columbia-ai-robotics/diffusion_policy)