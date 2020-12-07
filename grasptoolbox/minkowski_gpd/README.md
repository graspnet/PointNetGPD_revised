# Mingkowski-GPD
GPD algorithm with 3D sparse convolution.

Current code is just a demo using [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine), and far from the best result. The network architecture and training params can be further improved. You may try new architecture built upon MinkowskiEngine or add data augmentation functions.

## Requirements
- [Anaconda](https://www.anaconda.com/) with Python 3.7
- PyTorch 1.6 with CUDA 10.2
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) v0.4.3

## Installation
1. Follow MinkowskiEngine [instructions](https://github.com/NVIDIA/MinkowskiEngine#anaconda) to install [Anaconda](https://www.anaconda.com/), cudatoolkit, Pytorch and MinkowskiEngine.

2. Install other requirements from Pip. If there are errors when importing open3d, try ``pip install open3d==0.8``.
```bash
    pip install -r requirements.txt
```

## Training
```bash
CUDA_VISIBLE_DEVICES=[GPU_ID] python graspnet.py --log_dir log --batch_size 256
```
The model is trained on 2.5m+ grasp cloud data. Other training configuration could be found in [``graspnet.py``](graspnet.py).