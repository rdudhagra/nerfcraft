import argparse
import glob
import numpy as np
import os
import sys
import time
import torch
from PIL import Image

sys.path.insert(0, "anvil-parser")
import anvil

sys.path.insert(0, "torch-ngp")
from nerf.utils import seed_everything

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def avg_rgb(block_idx):
    img = np.array(Image.open(f"colors/textures/{block_idx}.png"), dtype=np.float32)[..., :3] # H, W, 3
    img_avg = np.mean(img.reshape(-1, 3), axis=0) # 3,
    return img_avg


def main(opts):
    seed_everything(opts.seed)

    if opts.ff:
        from nerf.network_ff import NeRFNetwork
    elif opts.tcnn:
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork

    # Load latest model checkpoint
    start_time = time.time()
    model = NeRFNetwork(
        encoding="hashgrid",
        bound=opts.bound,
        cuda_ray=True,
        density_scale=1,
        min_near=opts.min_near,
        density_thresh=opts.density_thresh,
        bg_radius=opts.bg_radius,
    ).to(device)

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
    print(f"Loaded model checkpoint in {time.time() - start_time:.3f}s")

    # Load color map
    start_time = time.time()
    with open("colors/blocks_full.txt", "r") as f:
        allow_blocks = set(f.read().splitlines())
    block_ids = set([filename.replace(".png", "") for filename in os.listdir("colors/textures")])
    block_ids = list(block_ids & allow_blocks)
    block_avg_rgbs = torch.tensor(
        np.array([avg_rgb(idx) for idx in block_ids]), dtype=torch.float32, device=device
    ) # ncolors, 3
    print(f"Loaded color map in {time.time() - start_time:.3f}s")

    # Sample whether each block is occupied from NGP model
    # Use YZX iteration order, following Anvil format
    start_time = time.time()
    block_xs = torch.arange(opts.world_min[0], opts.world_max[0], dtype=torch.int64, device=device) # nx,
    block_ys = torch.arange(opts.world_min[1], opts.world_max[1], dtype=torch.int64, device=device) # ny,
    block_zs = torch.arange(opts.world_min[2], opts.world_max[2], dtype=torch.int64, device=device) # nz,
    block_pos = torch.cartesian_prod(block_ys, block_zs, block_xs) # nx*ny*nz, 3

    ngp_xs = torch.linspace(-opts.bound, opts.bound, block_xs.shape[0], device=device) # nx,
    ngp_ys = torch.linspace(-opts.bound, opts.bound, block_xs.shape[0], device=device) # ny,
    ngp_zs = torch.linspace(-opts.bound, opts.bound, block_xs.shape[0], device=device) # nz,
    grid_size = torch.diff(torch.stack([ngp_xs[:2], ngp_ys[:2], ngp_zs[:2]], dim=-1), dim=0) # 1, 3
    ngp_pos = torch.cartesian_prod(ngp_ys, ngp_zs, ngp_xs) # nx*ny*nz, 3

    ngp_dir = torch.tensor(
        [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]],
        dtype=torch.float32, device=device
    ) # 6, 3
    ngp_dir = ngp_dir[:, None].expand(-1, ngp_pos.shape[0], -1) # 6, nx*ny*nz, 3
    print(f"Computed sample points in {time.time() - start_time:.3f}s")

    # Perturb ngp_pos in cardinal directions
    ngp_pos = ngp_pos[None] + ngp_dir * grid_size[None] # 6, nx*ny*nz, 3

    # Write to Minecraft Anvil format in parts
    region = anvil.EmptyRegion(0, 0)
    blocks = [anvil.Block("minecraft", idx) for idx in block_ids]

    chunk_size = 2**18
    n_parts = (ngp_pos.shape[1] + chunk_size - 1) // chunk_size
    for part_idx in range(n_parts):
        start_time = time.time()
        i = part_idx * chunk_size

        block_pos_ch = block_pos[i:i+chunk_size] # part, 3
        ngp_pos_ch = ngp_pos[:, i:i+chunk_size] # 6, part, 3
        ngp_dir_ch = ngp_dir[:, i:i+chunk_size] # 6, part, 3
        T = block_pos_ch.shape[0]

        # Compute density and color from six view directions
        sigma_ch = torch.zeros(T, dtype=torch.float32, device=device) # part, 3
        color_ch = torch.zeros(T, 3, dtype=torch.float32, device=device) # part, 3
        for ngp_pos_part, ngp_dir_part in zip(ngp_pos_ch, ngp_dir_ch):
            sigma_part, color_part = model(ngp_pos_part, ngp_dir_part) # part, | part, 3
            sigma_ch += sigma_part; del sigma_part
            color_ch += color_part; del color_part
        sigma_ch /= ngp_pos_ch.shape[0]
        color_ch /= ngp_pos_ch.shape[0]

        # Compute nearest color in block list
        block_idxs_ch = torch.zeros(T, dtype=torch.int64, device=device) # part,
        color_dists_ch = torch.full((T,), np.inf, dtype=torch.float32, device=device) # part,
        for i, block_rgb in enumerate(block_avg_rgbs):
            color_dist_ch = torch.sum((color_ch - block_rgb[None]) ** 2, dim=-1) # part,
            block_idxs_ch = torch.where(color_dist_ch < color_dists_ch, i, block_idxs_ch) # part,
            color_dists_ch = torch.min(color_dist_ch, color_dists_ch) # part,

        # Compute occupancy
        occupied_ch = (sigma_ch >= opts.density_thresh) # part,
        block_pos_ch = block_pos_ch.cpu().numpy() # part, 3
        occupied_idxs_ch = [int(x) for x in torch.nonzero(occupied_ch).cpu().numpy()]
        print(f"Sampled density and color for part {part_idx}/{n_parts} in {time.time() - start_time:.3f}s")

        # Placing blocks
        start_time = time.time()
        for idx in occupied_idxs_ch:
            x = int(block_pos_ch[idx, 0])
            y = int(block_pos_ch[idx, 1])
            z = int(block_pos_ch[idx, 2])
            block_idx = block_idxs_ch[idx]
            region.set_block(blocks[block_idx], x, y, z)
        print(f"Placed blocks in region for part {part_idx}/{n_parts} in {time.time() - start_time:.3f}s")

    start_time = time.time()
    region.save(f"torch-ngp/{opts.workspace}/r.0.0.mca")
    print(f"Saved region file in {time.time() - start_time:.3f}s")
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ff", action="store_true")
    parser.add_argument("--tcnn", action="store_true")
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
