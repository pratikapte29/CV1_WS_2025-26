import sys
import os
from kiwisolver import strength
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import multivariate_normal

'''
BG_pivot is the same shape as the input image but with single channel, all pixels have value 1.
'''

class MOG():
    def __init__(self,height=None, width=None, number_of_gaussians=None, background_thresh=None, lr=None):
        self.number_of_gaussians = number_of_gaussians
        self.background_thresh = background_thresh
        self.dist_thresh = 20
        self.lr = lr
        self.height = height
        self.width = width
        self.mus = np.zeros((self.height,self.width, self.number_of_gaussians,3)) ## assuming using color frames
        self.sigmaSQs = np.zeros((self.height, self.width, self.number_of_gaussians)) ## all color channels share the same sigma and covariance matrices are diagnalized
        self.omegas = np.zeros((self.height, self.width, self.number_of_gaussians))
        for i in range(self.height):
            for j in range(self.width):
                self.mus[i,j]=np.array([[122, 122, 122]]*self.number_of_gaussians) ##assuming a [0,255] color channel
                self.sigmaSQs[i,j]=[36.0] * self.number_of_gaussians
                self.omegas[i,j]=[1.0 / self.number_of_gaussians] * self.number_of_gaussians

    def gaussianPDF(self, X, mu, sigmaSQ):
        num_dim = 3  # RGB IMAGE OF DIMENSION 3
        diff = X - mu

        exponent = -np.sum(diff * diff) / (2 * sigmaSQ)
        denominator = np.power(2 * np.pi * sigmaSQ, num_dim / 2.0)
        
        return np.exp(exponent) / denominator      
                
    def updateParam(self, img, BG_pivot): #finish this function
        
        for i in range(self.height):
            for j in range(self.width):
                X_t = img[i, j]  # this is the current pixel

                matched_gaussian = -1  # store the index of the matched gaussian

                for k in range(self.number_of_gaussians):
                    mu = self.mus[i, j, k]
                    sigma = np.sqrt(self.sigmaSQs[i, j, k])
                    
                    # If X_t <= 2.5 standard deviations from the mean then labelmatched and stop 
                    # reference: https://www.researchgate.net/publication/3813345_Adaptive_Background_Mixture_Models_for_Real-Time_Tracking

                    if np.linalg.norm(X_t - mu) <= 2.5 * sigma:
                        matched_gaussian = k
                        break

                # If gaussian is marked as matched:
                if matched_gaussian != -1:

                    # all the formulas are from the above link - which is a summary of the original paper
                    # i could not access the full paper, so i have used formulas from the above summary

                    # increase the weight ofthe matched Gaussian
                    self.omegas[i, j, matched_gaussian] = (1 - self.lr) * self.omegas[i, j, matched_gaussian] + self.lr

                    # Decrease weights of unmatched Gaussians
                    for k in range(self.number_of_gaussians):
                        if k != matched_gaussian:
                            self.omegas[i, j, k] = (1 - self.lr) * self.omegas[i, j, k]
                    
                    mu_t_1 = self.mus[i, j, matched_gaussian]  # mu at time t - 1
                    sigmaSQ_t_1 = self.sigmaSQs[i, j, matched_gaussian]  # sigma squared at time t - 1
                    
                    rho = self.lr * self.gaussianPDF(X_t, mu_t_1, sigmaSQ_t_1) 
                    mu_t = (1 - rho) * mu_t_1 + rho * X_t  # updated mu at time t
                    sigmaSQ_t = (1 - rho) * sigmaSQ_t_1 + rho * np.dot((X_t - mu_t), (X_t - mu_t))  # updated sigma squared at time t

                    self.mus[i, j, matched_gaussian] = mu_t
                    self.sigmaSQs[i, j, matched_gaussian] = sigmaSQ_t

                else:
                    # mark X_t as foreground pixel
                    # BG_pivot[i, j] = 0

                    # find the least probable Gaussian

                    least_probable_gaussian = np.argmin(self.omegas[i, j])

                    self.mus[i, j, least_probable_gaussian] = X_t
                    self.sigmaSQs[i, j, least_probable_gaussian] = 36.0  # initial variance
                    self.omegas[i, j, least_probable_gaussian] = 0.05  # initial weight

                # Normalize the weights
                self.omegas[i, j] = self.omegas[i, j] / np.sum(self.omegas[i, j])

                # Order the Gaussians by omega/sigma
                strength = self.omegas[i, j] / np.sqrt(self.sigmaSQs[i, j])

                indices = np.argsort(-strength)  # negative for descending order

                # Sum the weights in this ordering until the sum is greater thana preset threshold - here self.background_thresh

                sum = 0
                background_indices = []
                for idx in range(len(indices)):
                    sum += self.omegas[i, j, idx]
                    background_indices.append(idx)
                    if sum > self.background_thresh:
                        break
                
                # compute background pixel as mean of the selected Gaussians
                # BG_pixel = np.zeros(3)
                # for idx in background_indices:
                #     BG_pixel += self.omegas[i, j, idx] * self.mus[i, j, idx]
                    
                # BG_pivot[i, j] = np.clip(BG_pixel, 0, 255)
                
                # BG_pixel = np.zeros(3)
                # total_weight = 0
                # for idx in background_indices:
                #     BG_pixel += self.omegas[i, j, idx] * self.mus[i, j, idx]
                #     total_weight += self.omegas[i, j, idx]

                # BG_pixel /= total_weight

                # # Decide if current pixel is foreground or background
                # avg_sigma = np.mean(np.sqrt(self.sigmaSQs[i, j, background_indices]))
                bg_weight_sum = 0
                bg_pixel = False

                for k in range(self.number_of_gaussians):
                    bg_weight_sum += self.omegas[i, j, k]
                    dist = np.linalg.norm(X_t - self.mus[i, j, k])

                    if bg_weight_sum > self.background_thresh:
                        if dist <= self.dist_thresh * np.sqrt(self.sigmaSQs[i, j, k]):
                            bg_pixel = True
                            break
                        if not bg_pixel:
                            BG_pivot[i, j] = 255 # mark as foreground

        return BG_pivot.astype(np.uint8)


for i in range(1, 3+1):#display first 3 labeled foreground images
    img = cv2.imread('imgs/{:04d}.jpg'.format(i))
    height, width = img.shape[:2]

    mog=MOG(
        height=height,
        width=width,
        number_of_gaussians=3,
        background_thresh=0.5,
        lr=0.01
    ) #finish this line of code
    label_img = mog.updateParam(img, np.ones(img.shape[:2]))
    cv2.imwrite('label{:04d}.jpg'.format(i), label_img)

