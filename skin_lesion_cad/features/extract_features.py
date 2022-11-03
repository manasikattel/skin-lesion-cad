from imageio import save
import pandas as pd
from xgboost import train
from skin_lesion_cad.features import get_glcm, get_lbp, ColorFeaturesExtractor, shape_features
import cv2
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
from pathlib import Path
import functools as ft
import random

CHALLENGE = "chall1"


def shape_feature(img_path):
    mask = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img_name = str(img_path).replace("mask", "inpaint")
    imf_features = shape_features(mask)
    imf_features['image'] = img_name
    return imf_features


def get_shape(image_paths):
    res = [shape_feature(image) for image in tqdm(image_paths)]
    return pd.DataFrame(res)


def process_img(img_path, mask_type, cfe):
    # img_path, mask_type, cfe = x
    img = cv2.imread(str(img_path))
    img = cv2.medianBlur(img, 3)

    if mask_type == 'ground_truth':
        mask = cv2.imread(str(img_path).replace(
            'inpaint', 'mask'), cv2.IMREAD_GRAYSCALE)
    if mask_type == 'full_image':
        mask = np.ones(img.shape[:2], dtype=np.uint8)
    if isinstance(mask_type, int):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = cv2.circle(
            mask, (mask.shape[1]//2, mask.shape[0]//2), mask_type, 1, -1)

    imf_fearures = cfe.extract_masked(img, mask)
    imf_fearures['image'] = str(img_path)
    return imf_fearures


def get_color(image_paths, mask_type, color_spaces, mshift_params=None, rel_col=None):

    cfe = ColorFeaturesExtractor(color_spaces)
    if mshift_params is not None:
        cfe = ColorFeaturesExtractor(color_spaces, mshift_params)
    elif rel_col is not None:
        cfe = ColorFeaturesExtractor(color_spaces, rel_col=True)
    else:
        cfe = ColorFeaturesExtractor(color_spaces)
    # candidates_features = []
    # breakpoint()
    # with mp.Pool(8) as pool:
    #     for result in tqdm(pool.imap(process_img, zip(image_paths, [mask_type]*len(image_paths), [cfe]*len(image_paths))), total=len(image_paths)):
    #         candidates_features.append(result)
    candidates_features = [process_img(
        image_path, mask_type, cfe) for image_path in tqdm(image_paths)]
    df = pd.DataFrame(candidates_features)
    return df


def extract_features(image_paths, save_path, mode="train",):
    color_spaces = {'bgr': cv2.COLOR_RGB2BGR, 'hsv': cv2.COLOR_RGB2HSV,
                    'lab': cv2.COLOR_RGB2LAB,  'YCrCb': cv2.COLOR_RGB2YCrCb}
    mshift_params = {'sp': 10, 'sr': 15}  # spatial and color range radius

    print("Extracting GLCM features")
    # glcm_df = get_glcm(image_paths).reset_index(drop=True)
    # glcm_df_masked = get_glcm(image_paths, masked=True).reset_index(drop=True)

    print("Extracting LBP features")
    # lbp_df = get_lbp(image_paths).reset_index(drop=True)
    # lbp_df_masked = get_lbp(image_paths, masked=True).reset_index(drop=True)

    # print("Extracting raw color features")
    color_df_raw = get_color(
        image_paths, mask_type="ground_truth", color_spaces=color_spaces).reset_index(drop=True)

    # print("Extracting meanshift color features")
    # color_df_ms = get_color(
    #     image_paths, mask_type="ground_truth", color_spaces=color_spaces, mshift_params=mshift_params).reset_index(drop=True)
    mask_paths = [Path(str(image_path.parent).replace(
        "raw", "processed")) / Path(image_path.name.replace("inpaint", "mask")) for image_path in image_paths]

    shape_df = get_shape(mask_paths)
    shape_df.to_feather(
        save_path/Path(f"{CHALLENGE}_{mode}_shape.feather"))

    # glcm_df.to_feather(
    #     save_path/Path(f"{CHALLENGE}_{mode}_glcm_original_image.feather"))
    # glcm_df_masked.to_feather(
    #     save_path/Path(f"{CHALLENGE}_{mode}_glcm_masked.feather"))

    # lbp_df.to_feather(
    #     save_path/Path(f"{CHALLENGE}_{mode}_lbp_original_image.feather"))
    # lbp_df_masked.to_feather(
    #     save_path/Path(f"{CHALLENGE}_{mode}_lbp_masked.feather"))

    color_df_raw.to_feather(
        save_path/Path(f"{CHALLENGE}_{mode}_color_raw.feather"))
    # color_df_ms.to_feather(
    #     save_path/Path(f"{CHALLENGE}_{mode}_color_meanshift.feather"))

    # dfs = [glcm_df, lbp_df, color_df_raw]
    # all_feat = ft.reduce(
    #     lambda left, right: pd.merge(left, right, on='image'), dfs)
    # all_feat.to_feather(
    #     save_path/Path(f"{CHALLENGE}_{mode}_all_feat.feather"))
    return


if __name__ == "__main__":
    color_spaces = {'bgr': cv2.COLOR_RGB2BGR, 'hsv': cv2.COLOR_RGB2HSV,
                    'lab': cv2.COLOR_RGB2LAB,  'YCrCb': cv2.COLOR_RGB2YCrCb}
    mshift_params = {'sp': 10, 'sr': 15}  # spatial and color range radius

    # train_path = Path(f"data/raw/{CHALLENGE}/train")
    # val_path = Path(f"data/raw/{CHALLENGE}/val")
    test_path = Path(f"data/processed/{CHALLENGE}/test")

    save_path = Path("data/processed/features")
    save_path.mkdir(exist_ok=True, parents=True)

    # training_names = train_path.rglob("*.jpg")
    # image_paths_training = [i for i in training_names]

    # val_names = val_path.rglob("*.jpg")
    # image_paths_val = [i for i in val_names]

    test_names = test_path.rglob("*_inpaint*")
    image_paths_test = [i for i in test_names]

    # extract_features(
    #     image_paths_training, save_path=save_path, mode="train")
    # extract_features(
    #     image_paths_val, save_path=save_path, mode="val")
    extract_features(
        image_paths_test, save_path=save_path, mode="test")
