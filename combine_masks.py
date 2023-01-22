import numpy as np
import cv2
import os

# Variables
DATA_DIR = "./data/hamerschlag"

files = [f for f in os.listdir(os.path.join(DATA_DIR, "images")) if ".JPG" in f]

for f in files:
    img = cv2.imread(os.path.join(DATA_DIR, "images", f))
    mask = cv2.imread(os.path.join(DATA_DIR, "images", "dynamic_mask_" + f.replace(".JPG", ".png")))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask > 0
    random_colors = np.random.randint(0, 255, (mask.shape[0], mask.shape[1], 3))
    img[mask] = random_colors[mask]
    cv2.imwrite(os.path.join(DATA_DIR, "masked_images", f), img)