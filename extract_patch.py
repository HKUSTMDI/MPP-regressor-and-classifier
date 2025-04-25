import numpy as np
from openslide import OpenSlide
from PIL import Image
import os
import random
from multiprocessing import Pool

def keep_tile(tile, threshold=220, threshold_percent=0.85):
    tile_array = np.array(tile)
    channel_above_threshold = tile_array > threshold
    pixel_above_threshold = np.prod(channel_above_threshold, axis=-1)
    percent_background_pixels = np.sum(pixel_above_threshold) / (tile_array.shape[0] * tile_array.shape[1])

    if percent_background_pixels > threshold_percent:
        return False
    tile_array = tile_array[:, :, :3].reshape(-1, 3)
    avg = np.mean(tile_array, axis=0)
    if avg[1] > avg[0] - 10 or avg[1] > avg[2] - 10:
        return False
    if np.abs(avg[0] - avg[1]) < 10 and np.abs(avg[1] - avg[2]) < 10:
        return False
    return True

def process_single_wsi(wsi_path, output_dir, patch_size=717, max_patches_per_level=10, threshold=220, threshold_percent=0.85):
    wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
    slide = OpenSlide(wsi_path)
    os.makedirs(output_dir, exist_ok=True)

    #base_mpp = float(slide.properties.get("openslide.mpp-x", 0.25))
    num_levels = 1
    patch_id = 0

    for level in range(num_levels):
        dim = slide.level_dimensions[level]
        #mpp = round(base_mpp * (2 ** level), 2)
        level_patches = []

        for y in range(0, dim[1], patch_size):
            for x in range(0, dim[0], patch_size):
                if x + patch_size <= dim[0] and y + patch_size <= dim[1]:
                    patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")
                    patch = patch.resize((448, 448), Image.Resampling.LANCZOS)
                    if keep_tile(patch, threshold, threshold_percent):
                        level_patches.append(patch)

        if not level_patches:
            print(f"[{wsi_name}] Level {level} 有效 patch 数量为 0，跳过")
            continue

        selected_patches = random.sample(level_patches, min(len(level_patches), max_patches_per_level))
        print(f"[{wsi_name}] Level {level} 筛选出 {len(selected_patches)} 个 patch（共 {len(level_patches)}）")

        for patch_img in selected_patches:
            patch_path = os.path.join(output_dir, f"{wsi_name}_patch_{patch_id:05d}_25.png")
            patch_img.save(patch_path)
            patch_id += 1

    slide.close()
    print(f"[{wsi_name}] 总共保存 {patch_id} 张 patch")

def process_wsi_folder(input_dir, output_dir, patch_size=717, max_patches_per_level=10, threshold=220, threshold_percent=0.85, num_workers=32):
    wsi_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith((".tif", ".svs"))]

    if not wsi_files:
        print("❌ 没有找到 WSI 文件")
        return

    print(f"✅ 共找到 {len(wsi_files)} 个 WSI 文件，开始处理…")

    os.makedirs(output_dir, exist_ok=True)

    with Pool(processes=32) as pool:
        pool.starmap(
            process_single_wsi,
            [(wsi_path, output_dir, patch_size, max_patches_per_level, threshold, threshold_percent) for wsi_path in wsi_files]
        )

# === 示例调用 ===
input_dir = "/hpc2hdd/home/xingmu/ABP提供乳腺癌WSI和病理报告/河北医科大学第四医院乳腺病例"
output_dir = "data_regression_random10_all"
process_wsi_folder(input_dir, output_dir)

