"""
Task 1: Distance Transform using Chamfer 5-7-11
Template for MA-INF 2201 Computer Vision WS25/26
Exercise 03
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def chamfer_distance_transform_5_7_11(binary_image):
    """
    Compute Chamfer distance transform using 5-7-11 mask.
    
    Based on Borgefors "Distance transformations in digital images" (1986).
    
    Chamfer 5-7-11:
    - Horizontal/vertical neighbors: weight = 5
    - Diagonal neighbors: weight = 7
    - Knight's move neighbors: weight = 11
    
    Args:
        binary_image: Binary image where features are 255, background is 0
    
    Returns:
        Distance transform image
    """
    h, w = binary_image.shape
    dt = np.full((h, w), np.inf, dtype=np.float32)
    
    # Initialize: 0 if feature pixel, infinity otherwise
    dt[binary_image > 0] = 0
    
    # Define forward and backward masks
    # Forward mask (as shown in slide 37)
    
    forward_mask = [
        (-1, 0, 5),   # horizontal/vertical neighbor
        (0, -1, 5),   # horizontal/vertical neighbor
        (-1, -1, 7),  # diagonal neighbor
        (-2, -1, 11), # knight's move neighbor
        (-1, -2, 11)  # knight's move neighbor
    ]
    
    # Backward mask (as shown in slide 37)
    backward_mask = [
        (1, 0, 5),    # horizontal/vertical neighbor
        (0, 1, 5),    # horizontal/vertical neighbor
        (1, 1, 7),    # diagonal neighbor
        (2, 1, 11),   # knight's move neighbor
        (1, 2, 11)    # knight's move neighbor
    ]

    # Forward pass
    # Update distances based on the forward mask
    for i in range(h):
        for j in range(w):
            # Update the distance values with the given weights
            for di, dj, weight in forward_mask:
                neighbour_i, neighbour_j = i + di, j + dj
                if 0 <= neighbour_i < h and 0 <= neighbour_j < w:
                    dt[i, j] = min(dt[i, j], dt[neighbour_i, neighbour_j] + weight)

    # Backward pass
    # Update distances based on the backward mask
    for i in range(h - 1, -1, -1):
        for j in range(w - 1, -1, -1):
            # Update the distance values with the given weights
            for di, dj, weight in backward_mask:
                neighbour_i, neighbour_j = i + di, j + dj
                if 0 <= neighbour_i < h and 0 <= neighbour_j < w:
                    dt[i, j] = min(dt[i, j], dt[neighbour_i, neighbour_j] + weight)
    
    return dt


def main():    
    
    print("=" * 70)
    print("Task 1: Distance Transform using Chamfer 5-7-11")
    print("=" * 70)
    
    img_path = 'data/bonn.jpg'
    # img_path = 'data/circle.png'      # play with different images
    # img_path = 'data/square.png'      
    # img_path = 'data/triangle.png'    
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found!")
        return
    
    # Load image and convert to grayscale
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Uncomment to visualize (just for personal debugging)
    # cv2.imshow("Edges", edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Compute distance transform with the function chamfer_distance_transform_5_7_11
    chamfer_dt = chamfer_distance_transform_5_7_11(edges)

    # Compute distance transform using cv2.distanceTransform
    cv2_dt = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)

    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Original image
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # 2. Edge image
    axes[0, 1].imshow(edges, cmap='gray')
    axes[0, 1].set_title("Edge Image")
    axes[0, 1].axis("off")

    # 3. Distance transform
    axes[1, 0].imshow(chamfer_dt, cmap='gray')
    axes[1, 0].set_title("Chamfer Distance Transform")
    axes[1, 0].axis("off")

    # 4. Distance transform using OpenCV
    axes[1, 1].imshow(cv2_dt, cmap='gray')
    axes[1, 1].set_title("OpenCV Distance Transform")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()
        
    print("\n" + "=" * 70)
    print("Task 1 complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
    