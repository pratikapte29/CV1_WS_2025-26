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
    """Rectilinear polygons using contour approximation with chain approx"""
    
    # Use CHAIN_APPROX_NONE to get all points, then simplify
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    
    polygons = []
    
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        
        # Simplify contour
        epsilon = 0.008 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        polygons.append(approx)
    
    return polygons

def plot_vectorized_map(image, refined_mask, polygons):
    """Plot like Fig 1 in PDF"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # (a) Initial segmentation
    initial_mask = cv2.imread('data/initial_segmentation.tif', cv2.IMREAD_GRAYSCALE)
    initial_mask = cv2.resize(initial_mask, (refined_mask.shape[1], refined_mask.shape[0]))
    axes[0, 0].imshow(initial_mask, cmap='gray')
    axes[0, 0].set_title('(a) Segmentation')
    axes[0, 0].axis('off')
    
    # (b) Refined segmentation
    axes[0, 1].imshow(refined_mask, cmap='gray')
    axes[0, 1].set_title('(b) Refined segmentation')
    axes[0, 1].axis('off')
    
    # (c) Vectorized map - outlines only
    vector_map = np.zeros(refined_mask.shape, dtype=np.uint8)
    cv2.drawContours(vector_map, polygons, -1, 255, 1)
    axes[1, 0].imshow(vector_map, cmap='gray')
    axes[1, 0].set_title('(c) Vectorized map')
    axes[1, 0].axis('off')
    
    # (d) Final result - overlay on image
    overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    cv2.drawContours(overlay, polygons, -1, (255, 0, 0), 2)  # Red outlines
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('(d) Final result')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('results_figure.png', dpi=150)
    plt.show()

def print_polygon_coordinates(polygons):
    """Print coordinates of each building polygon"""
    print("\nBuilding Polygon Coordinates:")
    print("=" * 40)
    
    for i, poly in enumerate(polygons):
        area = cv2.contourArea(poly)
        print(f"\nBuilding {i+1} (Area: {area:.0f} px²):")
        coords = poly.reshape(-1, 2)
        for j, (x, y) in enumerate(coords):
            print(f"  Corner {j+1}: ({x}, {y})")

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
print(f"Found {len(polygons)} buildings")

# Print coordinates
print_polygon_coordinates(polygons)

# Plot
vector_map = plot_vectorized_map(img, refined_mask=refined_mask, polygons=polygons)

# Save vector map
cv2.imwrite('data/vectorized_map.tif', vector_map)