# MIT License
# Copyright (c) 2025 Anton Schreiner

import os
os.environ["KAOLIN_NO_WIDGETS"] = "1"

import numpy as np
import torch
import kaolin as kal
import polyscope as ps
from py.utils import *
import argparse

args = argparse.ArgumentParser()
args.add_argument("--type", type=str, default="mesh")
args = args.parse_args()

if args.type == "mesh":

    asset_folder = find_file_or_folder('assets', os.getcwd())

    mesh = kal.io.import_mesh(str(asset_folder / 'cube.obj'), triangulate=True).cuda()
    mesh.vertices = kal.ops.pointcloud.center_points(mesh.vertices.unsqueeze(0), normalize=True).squeeze(0) 
    orig_vertices = mesh.vertices.clone()
    print(mesh)

    # Sample points
    num_samples = 1000000
    uniform_pts = torch.rand(num_samples, 3, device='cuda') * (
        orig_vertices.max(dim=0).values - orig_vertices.min(dim=0).values
    ) + orig_vertices.min(dim=0).values

    boolean_signs = kal.ops.mesh.check_sign(
        mesh.vertices.unsqueeze(0), 
        mesh.faces, 
        uniform_pts.unsqueeze(0), 
        hash_resolution=512
    )

    pts = uniform_pts[boolean_signs.squeeze()]
    print(f"Points inside mesh: {pts.shape[0]}")

    # Visualize with polyscope
    ps.init()
    ps.set_up_dir("y_up")

    # Add mesh (transparent)
    ps.register_surface_mesh(
        "mesh", 
        mesh.vertices.cpu().numpy(), 
        mesh.faces.cpu().numpy(),
        transparency=0.3
    )

    # Add interior points (subsample for performance)
    vis_pts = pts[::max(1, len(pts)//50000)].cpu().numpy()
    ps.register_point_cloud("interior_points", vis_pts, radius=0.003)

    ps.show()
elif args.type == "voxels":
    import polyscope as ps
    import numpy as np

    ps.init()

    # Your 3D texture stack: [D, H, W] or [D, H, W, C]
    volume_data = np.random.rand(64, 128, 128).astype(np.float32)

    # Register as a volume grid
    ps_vol = ps.register_volume_grid(
        "texture_stack",
        node_dims=volume_data.shape,
        bound_low=(0, 0, 0),
        bound_high=(1, 1, 1)
    )

    # Add scalar data
    ps_vol.add_scalar_quantity("density", values=volume_data, defined_on='nodes')

    ps.show()
else:
    print(f"Unknown visualization type: {args.type}")