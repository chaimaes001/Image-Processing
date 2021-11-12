"""
Created on Thu May 05 19:18:33 2021
@author: Chaimae
"""
from numpy import *
import numpy as np
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from skimage.color import *
import cv2
from PIL import Image


class Operation:
    def __init__(self,image):
        self.image = image

    def rotate_image(self, angle):
        image_size = (self.image.shape[1], self.image.shape[0])
        image_center = tuple(np.array(image_size) / 2)

        # Convert the OpenCV 3x2 rotation matrix to 3x3
        rot_mat = np.vstack(
            [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
        )

        rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

        # Shorthand for below calcs
        image_w2 = image_size[0] * 0.5
        image_h2 = image_size[1] * 0.5

        # Obtain the rotated coordinates of the image corners
        rotated_coords = [
            (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
        ]

        # Find the size of the new image
        x_coords = [pt[0] for pt in rotated_coords]
        x_pos = [x for x in x_coords if x > 0]
        x_neg = [x for x in x_coords if x < 0]

        y_coords = [pt[1] for pt in rotated_coords]
        y_pos = [y for y in y_coords if y > 0]
        y_neg = [y for y in y_coords if y < 0]

        right_bound = max(x_pos)
        left_bound = min(x_neg)
        top_bound = max(y_pos)
        bot_bound = min(y_neg)

        new_w = int(abs(right_bound - left_bound))
        new_h = int(abs(top_bound - bot_bound))

        # We require a translation matrix to keep the image centred
        trans_mat = np.matrix([
            [1, 0, int(new_w * 0.5 - image_w2)],
            [0, 1, int(new_h * 0.5 - image_h2)],
            [0, 0, 1]
        ])

        # Compute the tranform for the combined rotation and translation
        affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

        # Apply the transform
        result = cv2.warpAffine(
            self.image,
            affine_mat,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR
        )
        img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return result

    def hist(self):
        k = 0
        try:
            test = self.image.shape[2]
        except IndexError:
            k = 1
        if k == 1:
            h = Operation.histo(self.image)
            plt.subplot(1, 1, 1)
            plt.plot(h)
            plt.show()

        else:
            for i in range(0, 3):
                h = Operation.histo(self.image[:, :, i])
                plt.subplot(1, 3, i + 1)
                plt.plot(h)
            plt.show()

    def histo(image):
        h = zeros(256, dtype=float32)
        s = image.shape
        for j in range(s[0]):
            for i in range(s[1]):
                valeur = image[j, i]
                h[valeur] += 1
        return h

    def imhist(im):
        m, n = im.shape
        h = [0.0] * 256
        for i in range(m):
            for j in range(n):
                h[im[i, j]] += 1
        return np.array(h) / (m * n)

    def cumsum(h):
        return [sum(h[:i + 1]) for i in range(len(h))]

    def histeq(self):
        h = Operation.imhist(self.image)

        cdf = np.array(Operation.cumsum(h))
        print ('ssss')
        sk = np.uint8(255 * cdf)
        s1, s2 = self.image.shape
        Y = np.zeros_like(self.image)
        for i in range(0, s1):
            for j in range(0, s2):
                Y[i, j] = sk[self.image[i, j]]
        plt.plot(Y)
        plt.show()
        return Y

    def etire(self):
        MaxV = np.max(self.image)
        MinV = np.min(self.image)
        Y = np.zeros_like(self.image)
        m = self.image.shape
        for i in range(m[0]):
            for j in range(m[1]):
                Y[i, j] = (255 / (MaxV - MinV) * self.image[i, j] - MinV)
        plt.plot(Y)
        plt.show()
        return Y

