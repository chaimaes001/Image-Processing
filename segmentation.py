"""
Created on Thu May 08 17:05:17 2021
@author: Chaimae
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Segmentation:

    def __init__(self,image):
        self.image = image

    def k_means(self):
        pixel_values = self.image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 3
        ret, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(self.image.shape)
        return segmented_image

    def partition_regions(self):
        gray_r = self.image.reshape(self.image.shape[0] * self.image.shape[1])
        for i in range(gray_r.shape[0]):
            if gray_r[i] > gray_r.mean():
                gray_r[i] = 3
            elif gray_r[i] > 0.5:
                gray_r[i] = 2
            elif gray_r[i] > 0.25:
                gray_r[i] = 1
            else:
                gray_r[i] = 0
        gray = gray_r.reshape(self.image.shape[0], self.image.shape[1])
        plt.imshow(gray, cmap='gray')
        return gray