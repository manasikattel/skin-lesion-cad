import cv2
from math import copysign, log10
import numpy as np

def shape_features(mask):

    contours,hierarchy = cv2.findContours(mask, 1, 2)

    if len(contours) != 0:
        cont_areas = []

        # get maximum area contour
        max_cont = contours[0]
        max_cont_area = cv2.contourArea(max_cont)
        for cont in contours:
            cnt_area = cv2.contourArea(cont)
            if  cnt_area > max_cont_area:
                max_cont_area = cv2.contourArea(cont)
                max_cont = cont
            cont_areas.append(cnt_area)
        
        x,y,w,h = cv2.boundingRect(max_cont)
        hull = cv2.convexHull(max_cont)
        hull_area = cv2.contourArea(hull)
        
        features = {'cont_num':len(contours),
                    'mean_cont_area':np.mean(cont_areas),
                    'std_cont_area':np.std(cont_areas),
                    'area':max_cont_area,
                    'perim': cv2.arcLength(max_cont,True),
                    'aspect_ratio':float(w)/h,
                    'extent':float(max_cont_area)/ w*h,
                    'solidity':float(max_cont_area)/hull_area,
                    'equi_diameter':np.sqrt(4*max_cont_area/np.pi),
                    }
        
        moments = cv2.moments(max_cont)
        hu_moments = cv2.HuMoments(moments)
        for i in range(0,7):
            features[f'hu_{i}'] = hu_moments[i][0]# -1* copysign(1.0, hu_moments[i]) * log10(abs(hu_moments[i]))
    else:
        features = {'cont_num':len(contours)}
    return features