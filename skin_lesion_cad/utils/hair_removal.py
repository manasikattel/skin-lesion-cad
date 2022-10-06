import cv2
import numpy as np


class HairRemoval:
    def __init__(self, se_width=17, se_height=13, se_agnles=8,
                 min_area=50, er_threshold=10, hat_threshold=0.9,
                 max_area=2500) -> None:
        """Constructor for HairRemoval class.

        Args:
            se_width (int): Odd number. Width of the structuring element
                used for the blackhat and whitehat operations.
            se_height (int): Odd number. Height of the structuring element
                used for the blackhat and whitehat operations.
            se_agnles (int): Number of angles to use for the structuring element.
            min_area (int, optional): Minimum area of connected component
                to be considered a hair. Defaults to 20.
            er_threshold (int, optional): Elongation threshold. Defaults to 1.
            hat_threshold (float, optional): Percentile threshold to filter
                hat transform results (keeps intensity values of the
                result that are higher then given value). Defaults to 0.9.
            max_area (int, optional): Maximum area of connected component of hair.
                Defaults to 2500.
        """
        self.se_width = se_width
        self.se_height = se_height
        self.se_agnles = se_agnles
        self.min_area = min_area
        self.er_threshold = er_threshold
        self.hat_threshold = hat_threshold
        self.max_area = max_area

    def get_hair_mask(self, image: np.ndarray) -> np.ndarray:
        """Removes hair from an image.

        Args:
            image (np.ndarray): Image to remove hair from.
                Should be singe channel. (H, W)

        Returns:
            np.ndarray(np.uint8): Binary mask of the hair.
                0 - background, 255 - hair.
        """

        # detects darker hairs in the image
        blackhat = self.brias_hair(image, 'black')
        blackhat = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)
        _, mask_black = cv2.threshold(blackhat, np.quantile(
            blackhat.ravel(), self.hat_threshold), 255, cv2.THRESH_BINARY)

        # detect bright hairs in the image
        whitehat = self.brias_hair(image, 'white')
        whitehat = cv2.GaussianBlur(whitehat, (3, 3), cv2.BORDER_DEFAULT)
        _, mask_white = cv2.threshold(whitehat, np.quantile(
            whitehat.ravel(), self.hat_threshold), 255, cv2.THRESH_BINARY)

        # filter detected connected components based on enlongation ctiteria
        mask_black_filtered = self.filter_cc(mask_black)
        mask_white_filtered = self.filter_cc(mask_white)

        # combine both masks and saturate to so 255 is the max value
        merged_masks = 255*mask_black_filtered + 255*mask_white_filtered
        merged_masks[merged_masks>255] = 255
        
        # remove cc from the mask that are potentially lesions
        
        # for this first remove small cc (like hair)
        dist_img = cv2.morphologyEx(merged_masks, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5),),
            iterations=1)
        # and leave only the largest ccs
        dist_img = cv2.morphologyEx(dist_img, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20),),
            iterations=1)
        # filter the mask based on the max are
        _, labels, stats, __ = cv2.connectedComponentsWithStatsWithAlgorithm(
                dist_img, 4, cv2.CV_32S, cv2.CCL_WU)
        remove_labels = np.arange(len(stats))[stats[:,4]>self.max_area]

        remove_labels = [x for x in remove_labels if x != 0]
        labels[~np.isin(labels, remove_labels)] = 0
        dist_img = labels>0

        merged_masks_post = cv2.dilate(merged_masks, np.ones((3,3), np.uint8), iterations=2)
        merged_masks_post[dist_img] = 0
        
        return merged_masks_post.astype(np.uint8)

    def inpaint_hair(self, image: np.ndarray,
                     hair_mask: np.ndarray) -> np.ndarray:
        """Inpaints hair from an image.

        Args:
            image (np.ndarray): Image with hair to inpaint.
            hair_mask (np.ndarray): np.uint8 binary mask of the hair.

        Returns:
            np.ndarray: image with hair inpainted.
        """
        return cv2.inpaint(image, hair_mask, 3, cv2.INPAINT_TELEA)

    @staticmethod
    def elongate_function(outspread_len: dict, real_areas: dict) -> dict:
        """Calculated elongation function for each hair connected component.

        Based on "PDE-based unsupervised repair of hair-occluded information
        in dermoscopy images of melanoma," 
        https://doi.org/10.1016/j.compmedimag.2009.01.003

        Args:
            outspread_len (dict): Lengths of hair's thinned line.
            real_areas (dict): Area of each hair's connected component.
        Returns:
            dict: Elongation ratio for each hair's connected component.
        """
        # ignore background label 0
        return {k: (outspread_len[k]**2)/real_areas[k] for k in outspread_len.keys() if k != 0}

    @staticmethod
    def plot_hair_contours(image, hair_mask, thickness=2, colour=(0, 255, 0)):
        contours, _ = cv2.findContours(
            hair_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        image_copy = image.copy()  # otherwise it modifies the original image
        return cv2.drawContours(image_copy, contours, -1, colour, thickness=thickness)

    def filter_cc(self, hair_mask):
        """Filter CC based on elongation"""
        _, labels, stats, __ = cv2.connectedComponentsWithStatsWithAlgorithm(
            hair_mask, 4, cv2.CV_32S, cv2.CCL_WU)

        real_area = {cc_label: area for cc_label,
                     area in enumerate(stats[:, 4])}

        # skelotonize mask so we have think lines representing
        # the hair that are contained inside each hairs connected component
        skelet_labels_bin = cv2.ximgproc.thinning(255*(labels > 0).astype(np.uint8),
                                                  thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        # backpropagate the hair labels to the thinned region so we can
        # count lenghts of each hair's thinned line
        skelet_labels = (skelet_labels_bin > 0).astype(np.uint8) * labels
        expanded_l = dict(zip(*(np.unique(skelet_labels, return_counts=True))))

        er = HairRemoval.elongate_function(expanded_l, real_area)
        er_labels = np.array(list(er.keys()))
        er_vals = np.array(list(er.values()))

        # remove cc with elongation ratio below threshold
        filtered_cc = er_labels[er_vals < self.er_threshold]
        # and area smaller than min_area or bigger than max area
        filtered_cc = np.insert(
            filtered_cc, 0, [c for c in real_area.keys() if real_area[c] < self.min_area])

        # remove filtered cc
        filtered_hairs_mask = labels.copy()
        filtered_hairs_mask[np.isin(filtered_hairs_mask, filtered_cc)] = 0
        return (255*(filtered_hairs_mask)>0).astype(np.uint8)

    @staticmethod
    def create_fb(width, height, n):
        """create 'n' rectangular Structuring Elements (SEs) at different orientations spanning the whole 360°"""
        if width % 2 == 0:
            raise ArithmeticError('width must be odd')
        if height % 2 == 0:
            raise ArithmeticError('height must be odd')

        base = np.zeros((width, width))
        # workaround: cv::line does not work properly when thickness > 1.
        # So we draw line by line.
        for k in range(width//2-height//2, width//2+height//2 + 1):
            cv2.line(base, (0,k), (width, k), 255)
        SEs = []
        # compute rotated SEs
        SEs.append(base.astype(np.uint8))
        angle_step = 180.0/n
        for k in range(1, n):
            se = cv2.warpAffine(base, cv2.getRotationMatrix2D((base.shape[1]//2, base.shape[0]//2), k*angle_step, 1.0), (width, width))
            SEs.append(se.astype(np.uint8))

        return SEs

    def brias_hair(self, image, hat='black'):
        """Hair removal based on A. Bria 'Image Processing' course tutorial"""
        sum_of_hats = np.zeros(image.shape)
        SEs = HairRemoval.create_fb(self.se_width, self.se_height, self.se_agnles)
        for se in SEs:
            if hat == 'black':
                sum_of_hats += cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, se)
            else:
                sum_of_hats += cv2.morphologyEx(image, cv2.MORPH_TOPHAT, se)

        sum_of_hats = (255*(sum_of_hats/sum_of_hats.max())).astype(np.uint8)
        return sum_of_hats
