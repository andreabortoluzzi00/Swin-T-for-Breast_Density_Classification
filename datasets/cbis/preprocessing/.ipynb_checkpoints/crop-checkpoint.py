import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import cv2 


def np_CountUpContinuingOnes(b_arr):
    left = np.arange(len(b_arr))
    left[b_arr > 0] = 0
    left = np.maximum.accumulate(left)

    rev_arr = b_arr[::-1]
    right = np.arange(len(rev_arr))
    right[rev_arr > 0] = 0
    right = np.maximum.accumulate(right)
    right = len(rev_arr) - 1 - right[::-1]
    return right - left - 1

def adjust_bounding_box(original_coords, left_crop, top_crop):
    x1, y1, x2, y2 = original_coords
    return x1 - left_crop, y1 - top_crop, x2 - left_crop, y2 - top_crop

def ExtractBreast(img, true_bounding_box):
    img_copy = img.copy()
    img = np.where(img <= 40, 0, img)
    height, _ = img.shape

    y_a = height // 2 + int(height * 0.4)
    y_b = height // 2 - int(height * 0.4)
    b_arr = img[y_b:y_a].std(axis=0) != 0
    continuing_ones = np_CountUpContinuingOnes(b_arr)
    col_ind = np.where(continuing_ones == continuing_ones.max())[0]
    img = img[:, col_ind]

    _, width = img.shape
    x_a = width // 2 + int(width * 0.4)
    x_b = width // 2 - int(width * 0.4)
    b_arr = img[:, x_b:x_a].std(axis=1) != 0
    continuing_ones = np_CountUpContinuingOnes(b_arr)
    row_ind = np.where(continuing_ones == continuing_ones.max())[0]

    adjusted_coords = adjust_bounding_box(true_bounding_box, col_ind[0], row_ind[0])
    return img_copy[row_ind][:, col_ind], adjusted_coords


input_folder = "thesis/datasets/cbis/clean_cbis_png/"
output_folder = "thesis/datasets/cbis/cropped_png/"
os.makedirs(output_folder, exist_ok=True)

results = []


for filename in tqdm(os.listdir(input_folder)):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)

    if img is None:
        print(f"Error: {img_path}")
        continue

   
    true_box = [0, 0, img.shape[1], img.shape[0]]

    cropped_img, adj_box = ExtractBreast(img, true_box)

   
    out_path = os.path.join(output_folder, filename)
    cv2.imwrite(out_path, cropped_img)


print("cropped images saved in:", output_folder)
