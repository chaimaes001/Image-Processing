"""
Created on Thu Apr 26 22:23:53 2021
@author: Chaimae
"""
from numpy import *
import cv2
from matplotlib.pyplot import *
import matplotlib.pyplot as plt

class Contours:

    def __init__(self,image):
        self.image = image

    def grad(self,seuil):
        imageX = self.image.copy()
        imageY = self.image.copy()
        for i in range(1, self.image.shape[0] - 1):
            for j in range(1, self.image.shape[1] - 1):
                imageX[i, j] = self.image[i, j+1] -self.image[i, j]
                imageY[i, j] = self.image[i+1,j] -self.image[i, j]
        imageXY = self.image.copy()
        for i in range(1, self.image.shape[0] - 1):
            for j in range(1, self.image.shape[1] - 1):
                imageXY[i, j] = sqrt(imageX[i, j] ** 2 + imageY[i, j] ** 2)
                if imageXY[i, j] < seuil:
                    imageXY[i, j] = 0
                else:
                    imageXY[i, j] = 255
        return imageXY

    def Robert(self,seuil):
        imageX = self.image.copy()
        imageY = self.image.copy()
        for i in range(1, self.image.shape[0] - 1):
            for j in range(1, self.image.shape[1] - 1):
                imageX[i, j] = self.image[i, j] - self.image[i - 1, j - 1]
                imageY[i, j] = self.image[i, j] - self.image[i + 1, j + 1]
        imageXY = self.image.copy()
        for i in range(1, self.image.shape[0] - 1):
            for j in range(1, self.image.shape[1] - 1):
                imageXY[i, j] = sqrt(imageX[i, j] ** 2 + imageY[i, j] ** 2)
                if imageXY[i, j] < seuil:
                    imageXY[i, j] = 0
                else:
                    imageXY[i, j] = 255
        return imageXY

    def Sobel(self,seuil):
        imageX = self.image.copy()
        imageY = self.image.copy()
        for i in range(1, self.image.shape[0] - 1):
            for j in range(1, self.image.shape[1] - 1):
                imageY[i, j] = -self.image[i-1, j-1] -2*self.image[i, j-1] -self.image[i+1, j-1] \
                               + self.image[i - 1, j + 1] +2*self.image[i, j + 1] +self.image[i + 1, j + 1]
                imageX[i, j] = self.image[i-1, j-1] + 2*self.image[i-1, j] +self.image[i - 1, j + 1]\
                                -self.image[i+1, j-1] - 2*self.image[i+1, j] - self.image[i + 1, j + 1]
        imageXY = self.image.copy()
        for i in range(1, self.image.shape[0] - 1):
            for j in range(1, self.image.shape[1] - 1):
                imageXY[i, j] = sqrt(imageX[i, j] ** 2 + imageY[i, j] ** 2)
                if imageXY[i, j] < seuil:
                    imageXY[i, j] = 0
                else:
                    imageXY[i, j] = 255
        return imageXY

    def Laplacien(self,seuil):
        imageXY = self.image.copy()
        for i in range(1, self.image.shape[0] - 1):
            for j in range(1, self.image.shape[1] - 1):
                imageXY[i, j] = -4*self.image[i, j] +self.image[i-1, j] +self.image[i+1, j] \
                               + self.image[i , j - 1] +self.image[i, j + 1]
                if imageXY[i, j] < seuil:
                    imageXY[i, j] = 0
                else:
                    imageXY[i, j] = 255
        return imageXY
    
    def kirsch(self,seuil):
        (thresh, Imagea) = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY)
        m, n = Imagea.shape
        list = []
        kirsch = np.zeros((m, n))
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                d1 = np.square(5 * Imagea[i - 1, j - 1] + 5 * Imagea[i - 1, j] + 5 * Imagea[i - 1, j + 1] -
                               3 * Imagea[i, j - 1] - 3 * Imagea[i, j + 1] - 3 * Imagea[i + 1, j - 1] -
                               3 * Imagea[i + 1, j] - 3 * Imagea[i + 1, j + 1])
                d2 = np.square((-3) * Imagea[i - 1, j - 1] + 5 * Imagea[i - 1, j] + 5 * Imagea[i - 1, j + 1] -
                               3 * Imagea[i, j - 1] + 5 * Imagea[i, j + 1] - 3 * Imagea[i + 1, j - 1] -
                               3 * Imagea[i + 1, j] - 3 * Imagea[i + 1, j + 1])
                d3 = np.square((-3) * Imagea[i - 1, j - 1] - 3 * Imagea[i - 1, j] + 5 * Imagea[i - 1, j + 1] -
                               3 * Imagea[i, j - 1] + 5 * Imagea[i, j + 1] - 3 * Imagea[i + 1, j - 1] -
                               3 * Imagea[i + 1, j] + 5 * Imagea[i + 1, j + 1])
                d4 = np.square((-3) * Imagea[i - 1, j - 1] - 3 * Imagea[i - 1, j] - 3 * Imagea[i - 1, j + 1] -
                               3 * Imagea[i, j - 1] + 5 * Imagea[i, j + 1] - 3 * Imagea[i + 1, j - 1] +
                               5 * Imagea[i + 1, j] + 5 * Imagea[i + 1, j + 1])
                d5 = np.square((-3) * Imagea[i - 1, j - 1] - 3 * Imagea[i - 1, j] - 3 * Imagea[i - 1, j + 1] - 3
                               * Imagea[i, j - 1] - 3 * Imagea[i, j + 1] + 5 * Imagea[i + 1, j - 1] +
                               5 * Imagea[i + 1, j] + 5 * Imagea[i + 1, j + 1])
                d6 = np.square((-3) * Imagea[i - 1, j - 1] - 3 * Imagea[i - 1, j] - 3 * Imagea[i - 1, j + 1] +
                               5 * Imagea[i, j - 1] - 3 * Imagea[i, j + 1] + 5 * Imagea[i + 1, j - 1] +
                               5 * Imagea[i + 1, j] - 3 * Imagea[i + 1, j + 1])
                d7 = np.square(5 * Imagea[i - 1, j - 1] - 3 * Imagea[i - 1, j] - 3 * Imagea[i - 1, j + 1] +
                               5 * Imagea[i, j - 1] - 3 * Imagea[i, j + 1] + 5 * Imagea[i + 1, j - 1] -
                               3 * Imagea[i + 1, j] - 3 * Imagea[i + 1, j + 1])
                d8 = np.square(5 * Imagea[i - 1, j - 1] + 5 * Imagea[i - 1, j] - 3 * Imagea[i - 1, j + 1] +
                               5 * Imagea[i, j - 1] - 3 * Imagea[i, j + 1] - 3 * Imagea[i + 1, j - 1] -
                               3 * Imagea[i + 1, j] - 3 * Imagea[i + 1, j + 1])

                # : Take the maximum value in each direction, the effect is not good, use another method
                list = [d1, d2, d3, d4, d5, d6, d7, d8]
                if int(np.sqrt(max(list))) > seuil:
                    kirsch[i, j] = 255

                else:
                    kirsch[i, j] = 0
                # : Rounding the die length in all directions
                # kirsch[i, j] =int(np.sqrt(d1+d2+d3+d4+d5+d6+d7+d8))
        return kirsch
    
    def Robirson(self,seuil):
        (thresh, Imagea) = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY)
        m, n = Imagea.shape
        list = []
        Ro = np.zeros((m, n))
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                N = np.square(Imagea[i - 1, j - 1] + Imagea[i - 1, j] + Imagea[i - 1, j + 1] -
                                - Imagea[i + 1, j - 1] -
                                 Imagea[i + 1, j] -  Imagea[i + 1, j + 1])
                NE = np.square(Imagea[i - 1, j] + Imagea[i - 1, j + 1] -
                                Imagea[i, j - 1] + Imagea[i, j + 1] -  Imagea[i + 1, j - 1] -
                                Imagea[i + 1, j] )
                E = np.square((-1) * Imagea[i - 1, j - 1] + Imagea[i - 1, j + 1] -
                               Imagea[i, j - 1] + Imagea[i, j + 1] -  Imagea[i + 1, j - 1]
                                + Imagea[i + 1, j + 1])
                SE = np.square((-1) * Imagea[i - 1, j - 1] -  Imagea[i - 1, j]  -
                               Imagea[i, j - 1] + Imagea[i, j + 1]  +
                               Imagea[i + 1, j] + Imagea[i + 1, j + 1])
                S = np.square((-1) * Imagea[i - 1, j - 1] -  Imagea[i - 1, j] - Imagea[i - 1, j + 1]
                              + Imagea[i + 1, j - 1] +
                                Imagea[i + 1, j] +  Imagea[i + 1, j + 1])
                SO = np.square(-1 * Imagea[i - 1, j] -  Imagea[i - 1, j + 1] +
                                Imagea[i, j - 1] - Imagea[i, j + 1] +  Imagea[i + 1, j - 1] +
                               Imagea[i + 1, j] )
                O = np.square(Imagea[i - 1, j - 1]  -  Imagea[i - 1, j + 1] +
                                Imagea[i, j - 1] -  Imagea[i, j + 1] +  Imagea[i + 1, j - 1] -
                                 Imagea[i + 1, j + 1])
                NO = np.square(Imagea[i - 1, j - 1] + Imagea[i - 1, j]  +
                                Imagea[i, j - 1] -  Imagea[i, j + 1]  -
                                Imagea[i + 1, j] - Imagea[i + 1, j + 1])

                # : Take the maximum value in each direction, the effect is not good, use another method
                list = [N, NE, E, SE, S, SO, O, NO]
                if int(np.sqrt(max(list))) > seuil:
                    Ro[i, j] = 255

                else:
                    Ro[i, j] = 0

                # : Rounding the die length in all directions
                # kirsch[i, j] =int(np.sqrt(d1+d2+d3+d4+d5+d6+d7+d8))
        return Ro
