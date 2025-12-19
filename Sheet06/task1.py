import sys
import os
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

        return np.exp(-0.5 * np.dot(diff, diff) / sigmaSQ) / np.power(2 * np.pi * sigmaSQ, num_dim / 2)      
                
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
                    self.omegas[i, j, matched_gaussian] = (1 - self.lr) * self.omegas[i, j, matched_gaussian] + self.lr
                    
                    mu_t_1 = self.mus[i, j, matched_gaussian]  # mu at time t - 1
                    sigmaSQ_t_1 = self.sigmaSQs[i, j, matched_gaussian]  # sigma squared at time t - 1
                    
                    rho = self.lr * self.gaussianPDF(X_t, mu_t_1, sigmaSQ_t_1) 
                    mu_t = (1 - rho) * mu_t_1 + rho * X_t  # updated mu at time t
                    sigmaSQ_t = (1 - rho) * sigmaSQ_t_1 + rho * np.dot((X_t - mu_t), (X_t - mu_t))  # updated sigma squared at time t

     
for i in range(1, 3+1):#display first 3 labeled foreground images
    img = cv2.imread('imgs/{:04d}.jpg'.format(i))
    mog=MOG() #finish this line of code
    label_img = mog.updateParam(img, np.ones(img.shape[:2]))
    cv2.imwrite('label{:04d}.jpg'.format(i), label_img)

