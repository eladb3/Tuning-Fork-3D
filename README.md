# Tuning-Fork-3D
This is the code for the Free-Form Tuning Fork project.

In this colab notebook we provide a usage example:
https://colab.research.google.com/drive/1jbMiMMLFxkXaZlNwFbU-UzglVcHeHYkd?usp=sharing

More examples (with stl output files) can be found in the folder `examples`.

## The code
There are three main modules,  `tuning_fork/optimization.py` that contains the code the the optimization for the cross section of the prongs, `tuning_fork/gen_stl.py` that contains the code for generating an stl file (mesh) from the cross section of the prongs, and `tuning_fork/tuning_fork.py` that unify the whole process under one class names `TuningFork`.

## Environment
In order to run this code, you should have pytorch installed (https://pytorch.org/), and:
```
pip install numpy
pip install opencv-python
pip install trimesh open3d
pip install scikit-image
pip install matplotlib
```
