import cv2
import random
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans, MiniBatchKMeans
from joblib import Parallel, delayed, parallel_backend
from sklearn.feature_extraction.text import TfidfTransformer


class DenseDescriptor:
    """Generate Descriptors with dense sampling"""

    def __init__(self, descriptor, minKeypoints=30, kpSize=10):
        """
        __init__ Constructor for DenseDescriptor class


        Parameters
        ----------
        descriptor : object
            Descriptor object with detect and compute methods
        minKeypoints : int, optional
            Min keypoints to be sampled from an image, by default 30
        kpSize : int, optional
            Radius of the keypoint for dense sampling, by default 10
        """

        self.descriptor = descriptor
        self.minKeypoints = minKeypoints
        self.kpSize = kpSize

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
            random.randint(0, y-1), random.randint(0, x-1), size=self.kpSize) for i in range(num)]
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
                i[0]), size=self.kpSize) for i in random.sample(list(all_mask_points), num_available)]
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
        if len(desc_keypoints) < self.minKeypoints:
            if mask is not None:
                additional_kp = self._sample_keypoints_with_mask(mask,
                                                                 num=self.minKeypoints-len(desc_keypoints))
            else:
                additional_kp = self._sample_keypoints(img,
                                                       num=self.minKeypoints-len(desc_keypoints))
            desc_keypoints = desc_keypoints + tuple(additional_kp)

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

    def _descriptors_to_histogram(self, descriptors, dictionary):
        return np.histogram(
            dictionary.predict(descriptors), bins=range(dictionary.n_clusters+1)
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

        self.des_list = X
        descriptors = np.vstack(self.des_list).astype(float)
        self.dictionary = MiniBatchKMeans(n_clusters=self.n_words, random_state=random_state,
                                          batch_size=self.batch_size, max_iter=10).fit(descriptors)

        X_trans = Parallel(n_jobs=self.n_jobs)(
            delayed(self._descriptors_to_histogram)(
                self.des_list[i], self.dictionary)
            for i in range(len(self.des_list))
        )
        frequency_vectors = np.stack(X_trans)
        # df is the number of images that a visual word appears in
        self.tfidftransformer = TfidfTransformer(smooth_idf=False)
        tfidf = self.tfidftransformer.fit_transform(
            frequency_vectors)

        return tfidf

    def transform(self, X, y=None):
        self.des_list = X

        X_trans = Parallel(n_jobs=-1)(
            delayed(self._descriptors_to_histogram)(
                self.des_list[i], self.dictionary)
            for i in range(len(self.des_list))
        )
        frequency_vectors = np.stack(X_trans)
        tfidf = self.tfidftransformer.transform(frequency_vectors)
        return tfidf
