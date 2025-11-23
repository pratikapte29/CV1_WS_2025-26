import numpy as np
import cv2
import skimage
import matplotlib
import math
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# TODO your implementation

def compute_edge_indicator(image, sigma=2.0):
    """stops contour at edges"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (5, 5), sigma)
    
    Gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    grad_mag = Gx**2 + Gy**2
    
    g = 1.0 / (1.0 + grad_mag / grad_mag.max())
    return g

def evolve_level_set(phi, g, iterations=50, dt=0.5):
    """
    Evolve level set using gradient descent
    """
    for it in range(iterations):
        # Compute gradients of phi
        phi_x = cv2.Sobel(phi, cv2.CV_32F, 1, 0, ksize=3)
        phi_y = cv2.Sobel(phi, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(phi_x**2 + phi_y**2 + 1e-8)
        
        # Curvature term (Laplacian)
        curvature = cv2.Laplacian(phi, cv2.CV_32F)
        
        # Update phi
        phi = phi + dt * g * curvature
        
        if it % 10 == 0:
            print(f"Iteration {it}/{iterations}", end='\r')
    
    print()
    return phi

def refine_with_level_set(image, initial_mask, iterations=100, dt=0.5):
    """Refine using level set method"""
    
    # Edge indicator
    g = compute_edge_indicator(image)
    
    # Initialize level set from mask (signed distance)
    # phi < 0 inside, phi > 0 outside
    phi = np.where(initial_mask > 0, -1.0, 1.0).astype(np.float32)
    
    # Smooth initial phi
    phi = cv2.GaussianBlur(phi, (5, 5), 2.0)
    phi = evolve_level_set(phi, g, iterations, dt)
    
    # Create mask for where phi < 0
    refined_mask = np.where(phi < 0, 255, 0).astype(np.uint8)
    
    return refined_mask

# Load
img = cv2.imread('data/img_mosaic.tif')
img = cv2.resize(img, (900, 600))
initial_mask = cv2.imread('data/initial_segmentation.tif', cv2.IMREAD_GRAYSCALE)
initial_mask = cv2.resize(initial_mask, (900, 600))

print("Refining with level set...")
refined_mask = refine_with_level_set(img, initial_mask, iterations=10, dt=0.3)

# Post-processing
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)

# Fill all holes
h, w = refined_mask.shape
mask_floodfill = refined_mask.copy()
flood_mask = np.zeros((h+2, w+2), np.uint8)
# found this method at: https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
cv2.floodFill(mask_floodfill, flood_mask, (0,0), 255)
holes = cv2.bitwise_not(mask_floodfill)
refined_mask = refined_mask | holes

# IoU
gt = cv2.imread('data/img_mosaic_label.tif', cv2.IMREAD_GRAYSCALE)
gt = cv2.resize(gt, (900, 600))
iou = np.logical_and(refined_mask > 0, gt > 0).sum() / np.logical_or(refined_mask > 0, gt > 0).sum()
print(f"IoU: {iou:.4f} ({iou*100:.2f}%)")

# Save
cv2.imwrite('data/img_mosaic_refined.tif', refined_mask)

# Visualize
plt.figure(figsize=(15, 5))
plt.subplot(131); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title('Original')
plt.subplot(132); plt.imshow(initial_mask, cmap='gray'); plt.title('Initial')
plt.subplot(133); plt.imshow(refined_mask, cmap='gray'); plt.title(f'Refined (IoU: {iou:.3f})')
plt.tight_layout()
plt.show()