# MIT License
# Copyright (c) 2025 Anton Schreiner

import os
os.environ["KAOLIN_NO_WIDGETS"] = "1"

import numpy as np
import torch
import pyvista as pv
import numpy as np
from py.utils import *
import argparse

args = argparse.ArgumentParser()
args.add_argument("--type", type=str, default="mesh")
args = args.parse_args()

if args.type == "voxels":
    # Your volume: [D, H, W] density/alpha values
    volume = np.random.randn(64, 64, 64).astype(np.float32) * 0.1

    # Create uniform grid
    grid = pv.ImageData(dimensions=volume.shape)
    grid.point_data["density"] = volume.flatten(order="F")

    # Ray marching volume render
    plotter = pv.Plotter()
    plotter.add_volume(
        grid, 
        scalars="density",
        opacity="sigmoid",  # or custom transfer function
        cmap="viridis",
        shade=True,
        
    )
    plotter.show()
else:
    print(f"Unknown visualization type: {args.type}")