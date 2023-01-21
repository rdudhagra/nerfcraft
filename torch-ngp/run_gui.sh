#python3 main_nerf.py data/fox --workspace fox_nerf --fp16 --cuda_ray --preload --gui --bound 4
#python3 main_nerf.py data/box --workspace box_nerf --fp16 --tcnn --cuda_ray --gui
#python3 main_nerf.py data/drone_mall --workspace drone_mall_nerf --fp16 --tcnn --cuda_ray --gui
#python3 main_nerf.py data/doherty --workspace doherty_nerf --fp16 --tcnn --cuda_ray --gui
#python3 main_nerf.py data/wean --workspace wean_nerf --fp16 --tcnn --cuda_ray --gui
python3 main_nerf.py data/hamerschlag --workspace hamerschlag_nerf --fp16 --cuda_ray --bound 2 --gui
#python3 main_nerf.py data/gates --workspace gates_nerf --fp16 --tcnn --cuda_ray --gui
