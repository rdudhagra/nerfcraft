import argparse
import torch
from nerf.network_tcnn import NeRFNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    seed_everything(opt.seed)

    model = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=True,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, default="workspace")
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



if __name__ == "__main__":
    opts = parse_args()
    main(opts)
