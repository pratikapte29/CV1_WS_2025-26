# Template for Exercise 5 – Canny Edge Detector

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def gaussian_smoothing(img, sigma=1.0):
    ksize = int(2 * np.ceil(3 * sigma) + 1)  # Kernel size
    ax = np.arange(-ksize // 2 + 1., ksize // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel /= np.sum(kernel)
    return cv2.filter2D(img, -1, kernel)


def compute_gradients(img):
    Gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    gradient_magnitude = np.hypot(Gx, Gy)
    gradient_magnitude = gradient_magnitude / gradient_magnitude.max() * 255  # Normalize
    
    gradient_angle = np.arctan2(Gy, Gx) * (180 / np.pi)  # Convert to degrees
    gradient_angle[gradient_angle < 0] += 180  # Map angles to [0, 180)
    
    return gradient_magnitude, gradient_angle


def nonmax_suppression(mag, ang):
    H, W = mag.shape
    Z = np.zeros((H, W), dtype=np.float32)
    angle = ang.copy()
    
    angle[angle < 0] += 180

    for i in range(1, H-1):
        for j in range(1, W-1):
            q = 255
            r = 255
            # 0 degrees
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = mag[i, j+1]
                r = mag[i, j-1]
            # 45 degrees
            elif 22.5 <= angle[i,j] < 67.5:
                q = mag[i+1, j-1]
                r = mag[i-1, j+1]
            # 90 degrees
            elif 67.5 <= angle[i,j] < 112.5:
                q = mag[i+1, j]
                r = mag[i-1, j]
            # 135 degrees
            elif 112.5 <= angle[i,j] < 157.5:
                q = mag[i-1, j-1]
                r = mag[i+1, j+1]
            
            if (mag[i,j] >= q) and (mag[i,j] >= r):
                Z[i,j] = mag[i,j]
            else:
                Z[i,j] = 0
    return Z


def double_threshold(nms, low_ratio=0.05, high_ratio=0.15):

    high = nms.max() * high_ratio
    low = high * low_ratio
    res = np.zeros_like(nms, dtype=np.uint8)
    
    strong = np.uint8(255)
    weak = np.uint8(50)
    
    strong_i, strong_j = np.where(nms >= high)
    weak_i, weak_j = np.where((nms <= high) & (nms >= low))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return res, weak, strong


def hysteresis(edge_map, weak, strong):
    H, W = edge_map.shape
    res = edge_map.copy()
    changed = True
    
    while changed:
        changed = False
        for i in range(1, H-1):
            for j in range(1, W-1):
                if res[i,j] == weak:
                    if ((res[i+1, j-1] == strong) or (res[i+1, j] == strong) or 
                        (res[i+1, j+1] == strong) or (res[i, j-1] == strong) or
                        (res[i, j+1] == strong) or (res[i-1, j-1] == strong) or
                        (res[i-1, j] == strong) or (res[i-1, j+1] == strong)):
                        res[i,j] = strong
                        changed = True
                    else:
                        res[i,j] = 0
    res[res != strong] = 0
    return res



def compute_metrics(manual_edges, cv_edges):
    manual_bin = (manual_edges > 0).astype(np.uint8)
    cv_bin = (cv_edges > 0).astype(np.uint8)
    
    mad = np.mean(np.abs(manual_bin - cv_bin))
    
    tp = np.sum((manual_bin==1) & (cv_bin==1))
    fp = np.sum((manual_bin==0) & (cv_bin==1))
    fn = np.sum((manual_bin==1) & (cv_bin==0))
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    return mad, precision, recall, f1


# ==========================================================

# TODO: 1. Load the grayscale image 'bonn.jpg'
# TODO: 2. Smooth the image using your Gaussian function
# TODO: 3. Compute gradients (magnitude and direction)
# TODO: 4. Apply non-maximum suppression
# TODO: 5. Apply double threshold (choose suitable low/high values)
# TODO: 6. Perform hysteresis to obtain final edges
# TODO: 7. Compare your result with cv2.Canny using MAD and F1-score
# TODO: 8. Display original image, your edges, and OpenCV edges

img = cv2.imread('data/bonn.jpg', cv2.IMREAD_GRAYSCALE)

smoothed = gaussian_smoothing(img, sigma=0.5)
mag, ang = compute_gradients(smoothed)
nms = nonmax_suppression(mag, ang)
dt, weak, strong = double_threshold(nms, low_ratio=0.05, high_ratio=0.15)
edges_manual = hysteresis(dt, weak, strong)

edges_cv = cv2.Canny(img, 100, 200)

mad, precision, recall, f1 = compute_metrics(edges_manual, edges_cv)
print("MAD:", mad, "F1-score:", f1)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(img, cmap='gray'); plt.title('Original')
plt.subplot(1,3,2); plt.imshow(edges_manual, cmap='gray'); plt.title('Manual Canny')
plt.subplot(1,3,3); plt.imshow(edges_cv, cmap='gray'); plt.title('OpenCV Canny')
plt.show()

