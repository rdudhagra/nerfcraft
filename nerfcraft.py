import argparse
import glob
import numpy as np
import os
import sys
import torch

sys.path.insert(0, "anvil-parser")
import anvil

sys.path.insert(0, "torch-ngp")
from nerf.network_tcnn import NeRFNetwork
from nerf.utils import seed_everything

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(opts):
    seed_everything(opts.seed)

    model = NeRFNetwork(
        encoding="hashgrid",
        bound=opts.bound,
        cuda_ray=True,
        density_scale=1,
        min_near=opts.min_near,
        density_thresh=opts.density_thresh,
        bg_radius=opts.bg_radius,
    )

    # Load latest model checkpoint
    checkpoint_list = sorted(glob.glob(f"torch-ngp/{opts.workspace}/checkpoints/ngp_ep*.pth"))
    if checkpoint_list:
        checkpoint = checkpoint_list[-1]
        print(f"Loading latest checkpoint {checkpoint}")
    checkpoint_dict = torch.load(checkpoint, map_location=device)

    if "model" not in checkpoint_dict:
        model.load_state_dict(checkpoint_dict)
    else:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_dict["model"], strict=False)
        if len(missing_keys) > 0:
            print(f"Warning: Missing keys {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Warning: Unexpected keys {unexpected_keys}")

    # Sample whether each block is occupied from NGP model
    # Use YZX iteration order, following Anvil format
    block_xs = torch.arange(opts.world_min[0], opts.world_max[0], dtype=torch.int64, device=device) # nx,
    block_ys = torch.arange(opts.world_min[1], opts.world_max[1], dtype=torch.int64, device=device) # ny,
    block_zs = torch.arange(opts.world_min[2], opts.world_max[2], dtype=torch.int64, device=device) # nz,
    block_pos = torch.cartesian_prod(block_ys, block_zs, block_xs) # nx*ny*nz, 3

    ngp_xs = torch.linspace(-opts.bound, opts.bound, block_xs.shape[0], device=device) # nx,
    ngp_ys = torch.linspace(-opts.bound, opts.bound, block_xs.shape[0], device=device) # ny,
    ngp_zs = torch.linspace(-opts.bound, opts.bound, block_xs.shape[0], device=device) # nz,
    ngp_pos = torch.cartesian_prod(ngp_ys, ngp_zs, ngp_xs) # nx*ny*nz, 3

    ngp_dir = torch.tensor([1, 1, 1], dtype=torch.float32, device=device) / np.sqrt(3) # 3,
    ngp_dir = ngp_dir[None].expand(ngp_pos.shape[0], -1) # nx*ny*nz, 3

    sigma, color = model.forward(ngp_pos, ngp_dir) # nx*ny*nz, | nx*ny*nz, 3
    occupied = (sigma >= opts.density_thresh) # nx*ny*nz,

    # Write to Minecraft Anvil format
    region = anvil.EmptyRegion(0, 0)
    stone = anvil.Block("minecraft", "stone")
    dirt = anvil.Block("minecraft", "dirt")

    block_pos = block_pos.cpu().numpy()
    occupied_idxs = [int(x) for x in torch.nonzero(occupied).cpu().numpy()]
    for idx in occupied_idxs:
        x = int(block_pos[idx, 0])
        y = int(block_pos[idx, 1])
        z = int(block_pos[idx, 2])
        region.set_block(stone, x, y, z)

    region.save(f"torch-ngp/{opts.workspace}/r.0.0.mca")
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, default="trial_nerf")
    parser.add_argument("--seed", type=int, default=0)

    # Dataset options
    parser.add_argument("--bound", type=float, default=2,
            help="Assume scene is bounded in box[-bound, bound]^3, will invoke adaptive ray marching if > 1")
    parser.add_argument("--scale", type=float, default=0.33,
            help="Scale camera location into box[-bound, bound]^3")
    parser.add_argument("--offset", type=float, nargs="*", default=[0, 0, 0],
            help="Offset of camera location")
    parser.add_argument("--min_near", type=float, default=0.2,
            help="Minimum near distance for camera")
    parser.add_argument("--density_thresh", type=float, default=10,
            help="Threshold for density grid to be occupied")
    parser.add_argument("--bg_radius", type=float, default=-1,
            help="If positive, use a background model at sphere(bg_radius)")

    # Minecraft options
    parser.add_argument("--world_min", type=int, nargs="*", default=[0, 0, 0],
            help="Minimum corner of scene in world")
    parser.add_argument("--world_max", type=int, nargs="*", default=[256, 256, 256],
            help="Maximum corner of scene in world")

    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parse_args()
    main(opts)
