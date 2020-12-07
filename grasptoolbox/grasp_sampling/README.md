# Grasp Sampling
A GPD-like grasping sampling and generation algorithm using darboux frame. Grasp clouds and grasp images are generated from [GraspNet](https://graspnet.net/).

This repo shows the complete process for data generation (object cloud generation -> normal estimation -> darboux frame estimation -> grasp cloud cropping -> grasp image rendering). It needs modifications for real-time grasp proposal generation.

The key functions is ``estimate_darboux_frame`` in [``gen_grasp_cloud.py``](gen_grasp_cloud.py) and ``transform_cloud_to_image`` in [``gen_grasp_image.py``](gen_grasp_image.py).

## Installation
Install other requirements from Pip. If there are errors when importing open3d, try ``pip install open3d==0.8``.
```bash
pip install -r requirements.txt
```

## Data Generation
Before running commands, you should modify dataset root and split in [``gen_grasp_cloud.py``](gen_grasp_cloud.py) and [``gen_grasp_image.py``](gen_grasp_image.py).

Grasp cloud generation:
```bash
python gen_grasp_cloud.py
```

Grasp image generation:
```bash
python gen_grasp_image.py
```

Grasp clouds are directly sampled from GraspNet, and grasp images are rendered from grasp clouds.