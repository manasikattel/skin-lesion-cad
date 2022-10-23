from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from skin_lesion_cad.features.colour import ColorFeaturesDescriptor
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from joblib import Parallel, delayed, parallel_backend
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import random
import cv2
import platform
import cpuinfo

if "Intel" in cpuinfo.get_cpu_info()['brand_raw']:
    from sklearnex import patch_sklearn
    patch_sklearn()


class DenseDescriptor:
    """Generate Descriptors with dense sampling"""

    def __init__(self, descriptor, min_keypoints=100, max_keypoints=500, kp_size=10):
        """
        __init__ Constructor for DenseDescriptor class


        Parameters
        ----------
        descriptor : object
            Descriptor object with detect and compute methods
        min_keypoints : int, optional
            Min keypoints to be sampled from an image, by default 100
        max_keypoints : int, optional
            Max keypoints to be sampled from an image, by default 500
        kp_size : int, optional
            Radius of the keypoint for dense sampling, by default 10
        """

        self.descriptor = descriptor
        self.min_keypoints = min_keypoints
        self.max_keypoints = max_keypoints
        self.kp_size = kp_size

    def _sample_keypoints(self, img,  num):
        """
        _sample_keypoints sample the "num" number of keypoints 
        randomly from the entire image


        Parameters
        ----------
        img : np.ndarray
        num : int
            number of keypoints to return

        Returns
        -------
        list
            list of keypoint objects
        """
        x, y = img.shape[0], img.shape[1]
        additional_kp = [cv2.KeyPoint(
            random.randint(0, y-1), random.randint(0, x-1), size=self.kp_size) for i in range(num)]
        return additional_kp

    def _sample_keypoints_with_mask(self, mask,  num):
        """
        _sample_keypoints_with_mask sample the "num" number of keypoints 
        with in the provided mask 

        Parameters
        ----------
        mask : np.ndarray
            Region to sample the keypoints from.
        num : int
            Required number of keypoints

        Returns
        -------
        list
            list of keypoint objects
        """
        all_mask_points = np.argwhere(mask)
        num_available = len(all_mask_points)

        if num <= num_available:
            additional_kp = [cv2.KeyPoint(int(i[1]), int(
                i[0]), size=self.kp_size) for i in random.sample(list(all_mask_points), num)]
        else:
            additional_kp = self._sample_keypoints(mask, num)

        return additional_kp

    def detect(self, img, mask=None):
        """
        detect Detect the keypoints from input image

        Parameters
        ----------
        img : np.ndarray
            Image to detect keypoints on
        mask : np.ndarray, optional
            Region to sample the keypoints from, by default None

        Returns
        -------
        tuple
            tuple of keypoint objects
        """
        desc_keypoints = self.descriptor.detect(img, None)

        if len(desc_keypoints) < self.min_keypoints:
            if mask is not None:
                additional_kp = self._sample_keypoints_with_mask(mask,
                                                                 num=self.min_keypoints-len(desc_keypoints))
            else:
                additional_kp = self._sample_keypoints(img,
                                                       num=self.min_keypoints-len(desc_keypoints))
            desc_keypoints = desc_keypoints + tuple(additional_kp)

        if len(desc_keypoints) > self.max_keypoints:
            desc_keypoints = random.sample(desc_keypoints, self.max_keypoints)

        return desc_keypoints

    def compute(self, img, keypoints):
        """
        compute Compute the descriptors from given keypoints.

        Parameters
        ----------
        img : np.ndarray
        keypoints : tuple
            tuple with keypoints

        Returns
        -------
        _type_
            _description_
        """
        des = self.descriptor.compute(img, keypoints=keypoints)
        return des

    def detectAndCompute(self, img, mask=None):
        kp = self.detect(img, mask=mask)
        des = self.compute(img, kp)
        return kp, des[1]


class BagofWords(TransformerMixin, BaseEstimator):
    """
    A generic BagofWords for any input descriptors
    """

    def __init__(self, n_words, batch_size=1024, n_jobs=None, random_state=None):
        """
        __init__ Constructor for BagofWords class

        _extended_summary_

        Parameters
        ----------
        n_words : int
            Number of words for the visual features to be used. 
            Also, the number of classes to cluster into.
        batch_size : int, optional
            Batch size for the MiniBatch Kmeans, by default 1024
        n_jobs : int, optional
            The maximum number of concurrently running jobs, by default None
        random_state : int, RandomState instance or None, optional
            Controls the pseudo random number generation for, by default None
        """
        self.n_words = n_words
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _descriptors_to_histogram(self, descriptors):

        # apply prepocessing before prediction if color features
        # TODO: remove this line after descr recalc
        descriptors = np.array(descriptors)
        # already fixed color descr feature extra to return np.array
        descriptors[np.isnan(descriptors)] = 0
        descriptors[np.isinf(descriptors)] = 0
        descriptors = self.scaler.transform(descriptors).astype(np.float32)

        return np.histogram(
            self.dictionary.predict(descriptors), bins=range(self.dictionary.n_clusters+1)
        )[0]

    def fit_transform(self, X, y=None):
        """
        fit_transform 

        Parameters
        ----------
        X : list
            list of generated descriptors for each image

        Returns
        -------
        np.ndarray
            tfidf values of Bag of Words to be used as image features 
        """
        random_state = check_random_state(self.random_state)

        descriptors = np.vstack(X).astype(np.float32)

        # learn and save scaling for color features
        descriptors[np.isnan(descriptors)] = 0
        descriptors[np.isinf(descriptors)] = 0
        self.scaler = StandardScaler()
        descriptors = self.scaler.fit_transform(descriptors)

        self.dictionary = KMeans(n_clusters=self.n_words, random_state=random_state,
                                 max_iter=100).fit(descriptors)

        X_trans = [self._descriptors_to_histogram(X[i]) for i in range(len(X))]
        # doesn't work with color features
        # and perhaps with other features as well
        # X_trans = Parallel(n_jobs=self.n_jobs)(
        #     delayed(self._descriptors_to_histogram)(
        #         X[i])
        #     for i in range(len(X))
        # )

        frequency_vectors = np.stack(X_trans)
        # df is the number of images that a visual word appears in
        self.tfidftransformer = TfidfTransformer(smooth_idf=False)
        tfidf = self.tfidftransformer.fit_transform(
            frequency_vectors)

        return tfidf

    def transform(self, X, y=None):

        # doesn't work with color features
        # and perhaps with other features as well
        # X_trans = Parallel(n_jobs=-1)(
        #     delayed(self._descriptors_to_histogram)(
        #         X[i])
        #     for i in range(len(X))
        # )

        X_trans = [self._descriptors_to_histogram(X[i]) for i in range(len(X))]
        frequency_vectors = np.stack(X_trans)
        tfidf = self.tfidftransformer.transform(frequency_vectors)
        return tfidf


class ColorDescriptor(DenseDescriptor):
    def __init__(self, descriptor,  color_spaces: dict, meanshift=None, min_keypoints=100, max_keypoints=500, kp_size=25):
        super().__init__(descriptor, min_keypoints, max_keypoints, kp_size)
        self.fe = ColorFeaturesDescriptor(color_spaces, meanshift, kp_size)

    def compute(self, img, keypoints):
        """
        compute Compute the descriptors from given keypoints.

        Parameters
        ----------
        img : np.ndarray
        keypoints : tuple
            tuple with keypoints

        Returns
        -------
        _type_
            _description_
        """

        # des = self.descriptor.compute(img, keypoints=keypoints)

        return self.fe.extract_masked(img, keypoints)

    def detectAndCompute(self, img, mask=None):
        kp = self.detect(img, mask=mask)
        des = self.compute(img, kp)
        return kp, des


class LBPDescriptor(DenseDescriptor):
    def __init__(self, descriptor, n_points_radius=[(24, 8), (8, 3), (12, 3), (8, 2), (8, 1)], kp_size=25, min_keypoints=100, max_keypoints=500, method="default") -> None:
        super().__init__(descriptor, min_keypoints, max_keypoints, kp_size)
        self.n_points_radius = n_points_radius
        self.method = method

    def lbp_hist(self, img, kp, eps=1e-7):
        # slice a patch around the keypoint
        x1 = max(0, int(kp.pt[1] - self.kp_size))
        x2 = min(img.shape[0], int(kp.pt[1] + self.kp_size))
        y1 = max(0, int(kp.pt[0] - self.kp_size))
        y2 = min(img.shape[1], int(kp.pt[0] + self.kp_size))
        patch = img[x1:x2, y1:y2]

        hist_concat = np.array([])
        for (n_points, radius) in self.n_points_radius:
            lbp = local_binary_pattern(
                patch, n_points, radius, self.method)
            (hist, _) = np.histogram(lbp.ravel(),
                                     bins=np.arange(0, n_points + 3),
                                     range=(0, n_points + 2))
            # normalize the histogram
            hist = hist.astype("float")
            hist /= (hist.sum() + eps)
            hist_concat = np.append(hist_concat, hist)

        return hist_concat

    def compute(self, img, keypoints):

        kp_features = []

        for kp_idx in range(len(keypoints)):
            kp_features.append(self.lbp_hist(
                img, keypoints[kp_idx]))
        return np.array(kp_features).astype(np.float32)

    def detectAndCompute(self, img, mask=None):
        kp = self.detect(img, mask=mask)
        des = self.compute(img, kp)
        return kp, des
