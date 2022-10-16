import cv2
import numpy as np
from scipy.stats import entropy, skew, kurtosis


""" 
color_spaces = {'bgr':cv2.COLOR_RGB2BGR, 'hsv':cv2.COLOR_RGB2HSV, 'lab':cv2.COLOR_RGB2LAB,  'YCrCb':cv2.COLOR_RGB2YCrCb}
mshift_params = {'sp': 10, 'sr': 15} # spatial and color range radius
"""
class ColorFeaturesExtractor:
    def __init__(self, color_spaces:dict, meanshift=None, rel_col=None) -> None:
        self.color_spaces = color_spaces
        self.meanshift = meanshift
        self.rel_col = rel_col
        
    def extract_masked(self, img, mask):
        if self.meanshift:
            img = cv2.pyrMeanShiftFiltering(img, **self.meanshift)
        
        img = img.astype(np.float32)
        
        if self.rel_col:
            img = ColorFeaturesExtractor.relative_color(img, mask)
        features = {}
        
        for csp_name, csp in self.color_spaces.items():
            
            img_csp = cv2.cvtColor(img, csp)
            for i in range(img_csp.shape[2]):
                features.update(ColorFeaturesExtractor.masked_features(img_csp[:,:,i][mask>0], csp_name, i))
        return features
    @staticmethod  
    def masked_features(pixels, clrsp, clrsp_idx):
        res = dict()
        if len(pixels):
            res[f'{clrsp}_{clrsp[clrsp_idx]}_mean'] = np.mean(pixels)
            res[f'{clrsp}_{clrsp[clrsp_idx]}_std'] = np.std(pixels)
            res[f'{clrsp}_{clrsp[clrsp_idx]}_skew'] = skew(pixels)
            res[f'{clrsp}_{clrsp[clrsp_idx]}_kurt'] = kurtosis(pixels)
            res[f'{clrsp}_{clrsp[clrsp_idx]}_max'] = np.max(pixels)
            res[f'{clrsp}_{clrsp[clrsp_idx]}_min'] = np.min(pixels)
            res[f'{clrsp}_{clrsp[clrsp_idx]}_entrp'] = entropy(pixels)
            res[f'{clrsp}_{clrsp[clrsp_idx]}_unq'] = len(np.unique(pixels))
        return res
    @staticmethod
    def relative_color(img, mask):
        non_leasion_means = img[mask==0].mean(axis=0)
        return img - non_leasion_means