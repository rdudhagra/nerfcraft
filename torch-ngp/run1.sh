#CUDA_VISIBLE_DEVICES=1 python3 main_nerf.py data/fox --workspace trial_nerf --fp16 --tcnn --cuda_ray --preload --bound 2
#CUDA_VISIBLE_DEVICES=1 python3 main_nerf.py data/box --workspace box_nerf --fp16 --tcnn --cuda_ray --bound 2
#CUDA_VISIBLE_DEVICES=1 python3 main_nerf.py data/drone_mall --workspace drone_mall_nerf --fp16 --tcnn --cuda_ray
#CUDA_VISIBLE_DEVICES=1 python3 main_nerf.py data/doherty --workspace doherty_nerf --fp16 --tcnn --cuda_ray
#CUDA_VISIBLE_DEVICES=1 python3 main_nerf.py data/wean --workspace wean_nerf --fp16 --tcnn --cuda_ray
#CUDA_VISIBLE_DEVICES=1 python3 main_nerf.py data/hamerschlag --workspace hamerschlag_nerf --fp16 --cuda_ray --bound 4 --bg_radius 8
#CUDA_VISIBLE_DEVICES=1 python3 main_nerf.py data/gates --workspace gates_nerf --fp16 --tcnn --cuda_ray
CUDA_VISIBLE_DEVICES=1 python3 main_nerf.py data/jefftantheman --workspace jeff_nerf --fp16 --cuda_ray --bound 4
