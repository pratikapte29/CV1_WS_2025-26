import numpy as np
import cv2
import skimage
import scipy
import time
import matplotlib as mlp
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
# TODO your implementation

def vectorize_buildings_90deg(refined_mask, min_area=500):
    """Approximating contours of buildings to polygons"""
    
    # Find the contours
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    
    polygons = []
    
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        
        # Simplify contour
        epsilon = 0.008 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        polygons.append(approx)
    
    """
    Below is my attempt to align all angles to 90 degrees
    It kinda works but im not particularly happy with the output
    """
    # trying to convert all angles to 90 degrees

    new_polygons = []

    for poly in polygons:
        pts = poly[:, 0, :]  # Extract points
        new_pts = [pts[0]]  # Start with the first point

        # orient all angles so that they align to the actual house orientation
        rect = cv2.minAreaRect(poly)
        dominant_angle = rect[2] * np.pi / 180

        # print(pts)
        
        for i in range(len(pts)):
            prev = new_pts[-1]
            curr = pts[i]

            # vector from prev to curr
            vec = curr - prev
            theta = np.arctan2(vec[1], vec[0])

            # round angle to 90
            relative_theta = theta - dominant_angle
            relative_theta_90 = round(relative_theta / (np.pi / 2)) * (np.pi / 2)
            theta_90 = relative_theta_90 + dominant_angle

            # Create new point at same distance but aligned to 90°
            length = np.linalg.norm(vec)
            new_x = prev[0] + length * np.cos(theta_90)
            new_y = prev[1] + length * np.sin(theta_90)

            new_pts.append([new_x, new_y])

        # Close polygon if not closed
        if np.linalg.norm(np.array(new_pts[0]) - np.array(new_pts[-1])) > 1e-3:
            new_pts.append(new_pts[0])
        
        # Convert to same format as polygons variable
        new_pts = np.array(new_pts, dtype=np.int32).reshape(-1, 1, 2)
        new_polygons.append(new_pts)

    return new_polygons


def plot_vectorized_map(image, refined_mask, polygons):
    """Plot all the outuputs"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Initial segmentation
    initial_mask = cv2.imread('data/initial_segmentation.tif', cv2.IMREAD_GRAYSCALE)
    initial_mask = cv2.resize(initial_mask, (refined_mask.shape[1], refined_mask.shape[0]))
    axes[0, 0].imshow(initial_mask, cmap='gray')
    axes[0, 0].set_title('(a) Segmentation')
    axes[0, 0].axis('off')
    
    # Refined segmentation
    axes[0, 1].imshow(refined_mask, cmap='gray')
    axes[0, 1].set_title('(b) Refined segmentation')
    axes[0, 1].axis('off')
    
    # Vectorized map
    vector_map = np.zeros(refined_mask.shape, dtype=np.uint8)
    cv2.drawContours(vector_map, polygons, -1, 255, 1)
    axes[1, 0].imshow(vector_map, cmap='gray')
    axes[1, 0].set_title('(c) Vectorized map')
    axes[1, 0].axis('off')
    
    # overlay on image
    overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    cv2.drawContours(overlay, polygons, -1, (255, 0, 0), 2)  # Red outlines
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('(d) Final result')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('results_figure.png', dpi=150)
    plt.show()

    return vector_map, overlay

# implementation and function calls below:

# Load refined mask
refined_mask = cv2.imread('data/img_mosaic_refined.tif', cv2.IMREAD_GRAYSCALE)

if refined_mask is None:
    print("ERROR: Could not load refined mask!")
    exit()

img = cv2.imread('data/img_mosaic.tif')
img = cv2.resize(img, (refined_mask.shape[1], refined_mask.shape[0]))

# Vectorize
print("Vectorizing buildings...")
polygons = vectorize_buildings_90deg(refined_mask, min_area=500)

# Plot
vector_map, overlay = plot_vectorized_map(img, refined_mask=refined_mask, polygons=polygons)

# Save vector map
cv2.imwrite('data/vectorized_map.tif', vector_map)
cv2.imwrite('data/vectorized_overlay.tif', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))