import random
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import numpy as np
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm


def glcm_features(image, features=["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]):
    m_glcm = graycomatrix(image, distances=[2, 5, 7, 10, 15], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
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
            columns={i: f"{col}_{i}" for i in range(len(features["contrast"].iloc[0]))}, inplace=True)
        all_feat.append(features_dis)
    feat_final_df = pd.concat(all_feat, axis=1)
    return feat_final_df.rename(columns={"image_0": "image"})


def lbph(image, n_points_radius=[(32, 3), (24, 3), (16, 2), (12, 5), (12, 3), (12, 2), (8, 3), (8, 2), (8, 1)], method="uniform", eps=1e-7):
    hist_concat = np.array([])
    for (n_points, radius) in n_points_radius:

        lbp = local_binary_pattern(
            image, n_points, radius, method)
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, n_points + 3),
                                 range=(0, n_points + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        hist_concat = np.append(hist_concat, hist)
    return hist_concat


def get_lbp(image_paths):

    lbp_feats = []
    for image_path in tqdm(image_paths):
        im = cv2.imread(str(image_path))
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        lbp_feats.append(lbph(im_gray))

    lbp_df = pd.DataFrame(lbp_feats, columns=[
        "lbp"+str(i) for i in range(len(lbp_feats[0]))])
    lbp_df["image"] = [str(i) for i in image_paths]
    return lbp_df