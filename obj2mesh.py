import numpy as np
import os
import sys
import torch
from PIL import Image

sys.path.insert(0, "anvil-parser")
import anvil

#filename = "meshes/posner_basement_00_02_02"
#filename = "meshes/gates"
#filename = "meshes/owen_bad"
#filename = "meshes/shawn"
#filename = "meshes/carl"
#filename = "gates.obj"
filename = sys.argv[1]
print(filename)

device = torch.device("cuda")

def avg_rgb(block_idx):
    img = np.array(Image.open(f"colors/textures/{block_idx}.png"), dtype=np.float32)[..., :3]
    img_avg = np.mean(img.reshape(-1, 3), axis=0) # 3,
    return img_avg


# Load color map
with open("colors/blocks_full.txt", "r") as f:
    allow_blocks = set(f.read().splitlines())
block_ids = set([filename.replace(".png", "") for filename in os.listdir("colors/textures")])
block_ids = list(block_ids & allow_blocks)
block_avg_rgbs = torch.tensor(
    np.array([avg_rgb(idx) for idx in block_ids]), dtype=torch.float32, device=device
)

# Read occupancy data
RESOLUTION = 256
if os.path.exists(f"{filename}/textured_output.jpg"):
    os.system(f"./obj2voxel/obj2voxel-v1.3.4-linux {filename}/textured_output.obj {filename}/textured_output.xyzrgb -t {filename}/textured_output.jpg -r {RESOLUTION} -s max")
elif os.path.exists(f"{filename}/textured_output.png"):
    os.system(f"./obj2voxel/obj2voxel-v1.3.4-linux {filename}/textured_output.obj {filename}/textured_output.xyzrgb -t {filename}/textured_output.png -r {RESOLUTION} -s max")
else:
    raise AssertionError

xyzrgb = []
with open(f"{filename}/textured_output.xyzrgb", "r") as f:
    for line in f.readlines():
        xyzrgb.append(tuple(int(x) for x in line.split(" ")))
xyzrgb = torch.tensor(xyzrgb, dtype=torch.int64, device=device)

region = anvil.EmptyRegion(0, 0)
blocks = [anvil.Block("minecraft", idx) for idx in block_ids]

chunk_size = 2**18
n_parts = (xyzrgb.shape[0] + chunk_size - 1) // chunk_size
for part_idx in range(n_parts):
    i = part_idx * chunk_size
    xyz_ch = xyzrgb[i:i+chunk_size, :3]
    color_ch = xyzrgb[i:i+chunk_size, 3:]
    T = xyz_ch.shape[0]

    block_idxs_ch = torch.zeros(T, dtype=torch.int64, device=device) # part,
    color_dists_ch = torch.full((T,), np.inf, dtype=torch.float32, device=device) # part,
    for i, block_rgb in enumerate(block_avg_rgbs):
        color_dist_ch = torch.sum((color_ch - block_rgb[None]) ** 2, dim=-1) # part,
        block_idxs_ch = torch.where(color_dist_ch < color_dists_ch, i, block_idxs_ch) # part,
        color_dists_ch = torch.min(color_dist_ch, color_dists_ch) # part,
    xyz_ch = xyz_ch.cpu().numpy()
    block_idxs_ch = block_idxs_ch.cpu().numpy()

    for idx in range(T):
        x = int(xyz_ch[idx, 0])
        y = int(xyz_ch[idx, 1])
        z = int(xyz_ch[idx, 2])
        if y < 256:
            block_idx = block_idxs_ch[idx]
            region.set_block(blocks[block_idx], x, y, z)

region.save(f"{filename}/r.0.0.mca")
