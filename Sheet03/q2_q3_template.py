"""
Task 2: Hough Transform for Circle Detection
Task 3: Mean Shift for Peak Detection in Hough Accumulator
Template for MA-INF 2201 Computer Vision WS25/26
Exercise 03
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os


def myHoughCircles(edges, min_radius, max_radius, threshold, min_dist, r_ssz, theta_ssz):
    """
    Your implementation of HoughCircles
    
    Args:
        edges: single-channel binary source image (e.g: edges)
        min_radius: minimum circle radius
        max_radius: maximum circle radius
        param threshold: minimum number of votes to consider a detection
        min_dist: minimum distance between two centers of the detected circles. 
        r_ssz: stepsize of r
        theta_ssz: stepsize of theta
        return: list of detected circles as (a, b, r, v), accumulator as [r, y_c, x_c]
    """
    max_radius = min(max_radius, int(np.linalg.norm(edges.shape)))

    edges_points = np.array(np.nonzero(edges))
    h, w = edges.shape

    # set range of radii
    radii = np.arange(min_radius, max_radius + 1, r_ssz)
    num_radii = len(radii)


    # Initialize Hough accumulator 
    accumulator = np.zeros((num_radii, h, w), dtype=np.float32)

    # Voting in the Hough accumulator
    for i in range(edges_points.shape[1]):
        y_edge = edges_points[0, i]
        x_edge = edges_points[1, i]

        for r_idx, r in enumerate(radii):
            for theta in range(0, 360, theta_ssz):
                theta_rad = np.deg2rad(theta)
                a = int(x_edge - r * np.cos(theta_rad))
                b = int(y_edge - r * np.sin(theta_rad))

                if 0 <= a < w and 0 <= b < h:
                    accumulator[r_idx, b, a] += 1

    # Detecting circles from the accumulator
    detected_circles = []
    for r_idx, r in enumerate(radii):
        acc_slice = accumulator[r_idx]
        # Find local maxima above the threshold
        y_idxs, x_idxs = np.where(acc_slice >= threshold)
        for y_c, x_c in zip(y_idxs, x_idxs):
            v = acc_slice[y_c, x_c]
            # Check for minimum distance
            too_close = False
            for (a, b, r_det, _) in detected_circles:
                if np.sqrt((x_c - a) ** 2 + (y_c - b) ** 2) < min_dist:
                    too_close = True
                    break
            if not too_close:
                detected_circles.append((x_c, y_c, r, v))
    

    return detected_circles, accumulator

def myMeanShift(accumulator, bandwidth, threshold=None):
    """
    Find peaks in Hough accumulator using mean shift.
    
    Args:
        accumulator: 3D Hough accumulator (n_radii, h, w)
        bandwidth: Bandwidth for mean shift
        threshold: Minimum value to consider (if None, use fraction of max)
        
    Returns:
        peaks: List of (x, y, r_idx, value) tuples
    """
    n_r, h, w = accumulator.shape
    
    # TODO
    
    # return peaks

def main():
    
    print("=" * 70)
    print("Task 2: Hough Transform for Circle Detection")
    print("=" * 70)
        
    img_path = 'data/coins.jpg'
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found!")
        return
        
    # Load image and convert to grayscale
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Display edges
    plt.figure()
    plt.imshow(edges, cmap='gray')
    plt.title("Edge Image")
    plt.axis("off")
    plt.show()

    # Detect circles - parameters tuned for coins image
    print("\nDetecting circles...")
    min_radius = 10
    max_radius = 30
    threshold = 15
    min_dist = 20
    r_ssz = 2
    theta_ssz = 5

    # Detect circles
    detected_circles, accumulator = myHoughCircles(edges, min_radius, max_radius, threshold, min_dist, r_ssz, theta_ssz)

    # Visualize detected circles
    for (x_c, y_c, r, v) in detected_circles:
        cv2.circle(img, (x_c, y_c), r, (0, 255, 0), 2)

    # Visualize accumulator slices
    for r_idx, r in enumerate(np.arange(10, 31, 2)):
        plt.subplot(3, 4, r_idx + 1)
        plt.imshow(accumulator[r_idx], cmap='hot', interpolation='nearest')
        plt.title(f"Radius: {r}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Visualize peak radius
    if detected_circles:
        peak_radius = max(detected_circles, key=lambda x: x[3])[2]
        print(f"Peak radius: {peak_radius}")

    # Visualize the accumulator slice at the radius with maximum votes
    peak_r_idx = np.where(np.arange(min_radius, max_radius + 1, r_ssz) == peak_radius)[0][0]
    plt.figure()
    plt.imshow(accumulator[peak_r_idx], cmap='hot')
    plt.title(f"Accumulator Slice at Peak Radius = {peak_radius}")
    plt.axis('off')
    plt.show()
    
    print("\n" + "=" * 70)
    print("Parameter Analysis:")
    print("  - Canny thresholds affect edge quality and thus detection")
    # ...more analysis can be added here
    print("=" * 70)
    print("Task 2 complete!")
    print("=" * 70)


    # =============================================================
    print("=" * 70)
    print("Task 3: Mean Shift for Peak Detection in Hough Accumulator")
    print("=" * 70)

    print("Applying mean shift to find peaks...")
    # peaks = myMeanShift # TODO
    
    # Visualize corresponding circles on original image    
    # TODO
    
    print("\n" + "=" * 70)
    print("Bandwidth Parameter Analysis:")
    # ...more analysis can be added here
    print("=" * 70)
    print("Task 3 complete!")
    

if __name__ == "__main__":
    main()