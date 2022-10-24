# %%
import random
from skimage.feature import graycomatrix, graycoprops
import numpy as np
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm


def glcm_features(image, features=["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]):
    m_glcm = graycomatrix(image, distances=[2, 5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                          symmetric=True, normed=True)
    feature_dict = {}
    for feature in features:
        feature_dict[feature] = [np.concatenate(
            graycoprops(m_glcm, feature)).tolist()]
    return feature_dict


def get_chall2_class(path):
    if "bcc" in str(path):
        return "bcc"
    elif "mel" in str(path):
        return "mel"
    elif "scc" in str(path):
        return "scc"
    else:
        raise ValueError("class needs to be bcc, mel or scc")


def get_glcm(image_paths):
    feature_dfs = []
    for image_path in tqdm(image_paths):
        img = cv2.imread(str(image_path))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        glcm_features_dict = glcm_features(gray_img)
        feature_df = pd.DataFrame(glcm_features_dict)
        feature_df["image"] = str(image_path)
        feature_dfs.append(feature_df)

    features = pd.concat(feature_dfs)

    all_feat = []
    for col in tqdm(features.columns.values):
        features_dis = features[col].apply(pd.Series)
        features_dis.rename(
            columns={i: f"{col}_{i}" for i in range(8)}, inplace=True)
        all_feat.append(features_dis)
    feat_final_df = pd.concat(all_feat, axis=1)
    return feat_final_df


if __name__ == "__main__":

    output_dir = Path("data/processed/features")
    output_dir.mkdir(exist_ok=True, parents=True)
    chall = "chall2"
    train_path = Path(f"data/processed/{chall}/train")

    training_names = train_path.rglob("*/*inpaint_0*")
    image_paths = [i for i in training_names]
    image_classes = [0 if ("nevus" in str(i)) else 1 for i in image_paths]
    # mask_paths = [Path(str(image_path.parent).replace("raw", "processed")) /
    #               Path(image_path.stem+"_mask_1_0.png") for image_path in image_paths]

    print("Extracting GLCM features")

    feat_final_df = get_glcm(image_paths)

    if chall == "chall1":
        feat_final_df["class"] = feat_final_df["image_0"].apply(
            lambda x: "nevus" if "nevus" in x else "others")
    else:
        feat_final_df["class"] = feat_final_df["image_0"].apply(
            get_chall2_class)
    feat_final_df.reset_index(inplace=True)
    feat_final_df.to_feather(output_dir/Path(f"glcm_features_{chall}.feather"))

    print(f"GLCM features saved to {output_dir}")
