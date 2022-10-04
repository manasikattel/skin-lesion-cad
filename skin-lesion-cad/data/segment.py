import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from utils import dice
import click


class Segment:

    def FindPoint(self, x1, y1, x2, y2, x, y):
        """
        FindPoint returns True if the point (x,y) is with in the rectangle (x1,y1) and (x2,y2)

        """
        if (x > x1 and x < x2 and
                y > y1 and y < y2):
            return True
        else:
            return False

    def check_corners(self, x, y, img, percentage=0.15):
        """
        check_corners check if point (x,y) lies in the border of the image img

        _extended_summary_

        Parameters
        ----------
        x : int
        y : int
        img : np.ndarray
        percentage : float, optional
            the proportion of the image to consider as a corner, by default 0.15

        Returns
        -------
        bool
            True if the point is in the corner, False otherwise
        """
        height = img.shape[0]
        width = img.shape[1]

        corner_tl = (0, 0), (int(height*percentage), int(width*percentage))
        corner_tr = (int(height*(1-percentage)),
                     0), (int(width), int(height*percentage))
        corner_bl = (0, int((1-percentage)*height)
                     ), (int(width*percentage), int(height))
        corner_br = (int((1-percentage)*width),
                     int((1-percentage)*height)), (int(width), int(height))
        return self.FindPoint(corner_tl[0][0], corner_tl[0][1], corner_bl[1][0], corner_bl[1][1], x=x, y=y) |\
            self.FindPoint(corner_tl[0][0], corner_tl[0][1], corner_tr[1][0], corner_tr[1][1], x=x, y=y) |\
            self.FindPoint(corner_tr[0][0], corner_tr[0][1], corner_br[1][0], corner_br[1][1], x=x, y=y) |\
            self.FindPoint(corner_bl[0][0], corner_bl[0][1],
                           corner_br[1][0], corner_br[1][1], x=x, y=y)

    def asf(self, img, kernel_size):
        """
        asf Alternate Sequential Filtering

        Parameters
        ----------
        img : np.ndarray
            Image to perform the operation on
        kernel_size : int
            size of the kernel to be used
        Returns
        -------
        np.ndarray
            Image after ASF.
        """
        closing = img
        for i in range(1, kernel_size):
            opening = cv2.morphologyEx(
                closing, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i, i)))
            closing = cv2.morphologyEx(
                opening, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i, i)))
        return closing

    def segment(self, img, img_name, save=False):
        """
        segment segment the lesion image

        _extended_summary_

        Parameters
        ----------
        img : np.ndarray
            Image to be segmented
        img_name : Path
            Filename of the Image
        save : bool, optional
            Flag, whether or not to save the masks, by default False

        Returns
        -------
        _type_
            _description_
        """
        gray_img = img[:, :, 2]
        clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(8, 8))
        gray_img_enh = clahe.apply(gray_img)

        gray_img_enh_asf = self.asf(
            gray_img_enh, kernel_size=(5, 5))
        blur = cv2.GaussianBlur(gray_img_enh_asf, (7, 7), 0)

        ret3, th3 = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)

        closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)

        analysis = cv2.connectedComponentsWithStats(closing, 4,
                                                    cv2.CV_32S)
        (totalLabels, label_ids, values, centroids) = analysis

        output = np.zeros(gray_img.shape, dtype="uint8")

        for i in range(1, totalLabels):
            area = values[i, cv2.CC_STAT_AREA]

            if (area > 1000) and not self.check_corners(centroids[i][0], centroids[i][1], img):
                componentMask = (label_ids == i).astype("uint8") * 255
                # Creating the Final output mask
                output = cv2.bitwise_or(output, componentMask)

        # TODO: Hole filling

        if save == True:
            save_dir = Path(str(img_name.parent).replace(
                "raw", "processed"))
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir/Path(img_name.stem+"_mask.png")
            cv2.imwrite(str(save_path), output)
        return output


if __name__ == "__main__":
    img_path_list = Path(
        "data/raw/skin lesion segmentation HAM10000 dataset/images").iterdir()
    dice_scores = []
    for img_path in tqdm(img_path_list):
        if img_path.suffix in [".jpg", ".png"]:
            img = cv2.imread(str(img_path))
            mask_path = Path(str(img_path.parent).replace(
                "images", "masks")) / Path(str(img_path.stem)+"_segmentation.png")
            mask = cv2.imread(str(mask_path))
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except:
                print(str(img_path))
            output = Segment().segment(img, img_path, save=True)
            dice_scores.append(dice(mask[:, :, 0], output))

    print(np.mean(dice_scores))
    print(np.var(dice_scores))
