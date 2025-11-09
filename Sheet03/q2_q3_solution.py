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

    '''
    Voting process:
    For every edge pixel, we assume it lies on the circumference of
    potential circles with different radii. For each radius, we compute
    all possible center coordinates (a, b) using:
        a = x - r*cos(theta)
        b = y - r*sin(theta)
    Each valid (a, b, r) is incremented in the accumulator.
    '''

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

    # If no threshold provided, use half of the max accumulator value
    if threshold is None:
        threshold = 0.5 * np.max(accumulator)

    # Collect all candidate points above threshold
    candidates = np.argwhere(accumulator > threshold)
    values = accumulator[accumulator > threshold]

    # Combine coordinates with their accumulator votes
    points = np.hstack((candidates, values[:, None]))  # shape: (N, 4)

    '''
    Each candidate point is iteratively shifted towards the mean position
    of its neighbors within a sphere of given bandwidth. This converges
    to a local maximum of the density (the mode).
    '''
    shifted_points = points[:, :3].astype(np.float64)

    for it in range(10):
        for i, p in enumerate(shifted_points):
            distances = np.linalg.norm(shifted_points - p, axis=1)
            neighbors = shifted_points[distances < bandwidth]

            if len(neighbors) > 0:
                shifted_points[i] = np.mean(neighbors, axis=0)

    '''
    Merge close-by points (modes) to avoid duplicates.
    '''
    peaks = []
    for i, p in enumerate(shifted_points):
        if not any(np.linalg.norm(p - q[:3]) < bandwidth/2 for q in peaks):
            r_i, y, x = p.astype(int)
            value = accumulator[r_i, y, x]
            peaks.append((x, y, r_i, value))

    return peaks
    

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
    min_radius = 20
    max_radius = 60
    threshold = 120
    min_dist = 20
    r_ssz = 1
    theta_ssz = 5

    # Detect circles
    detected_circles, accumulator = myHoughCircles(edges, min_radius, max_radius, threshold, min_dist, r_ssz, theta_ssz)

    # Visualize detected circles
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title("Detected Circles (Hough Transform)")

    for (x, y, r, v) in detected_circles:
        circ = Circle((x, y), r, color='lime', fill=False, linewidth=2)
        ax.add_patch(circ)
        ax.plot(x, y, 'r.', markersize=5)

    plt.show()

    # Visualize accumulator slices for some radii
    n_r, h, w = accumulator.shape
    plt.figure(figsize=(12, 4))
    for i, r in enumerate(range(min_radius, min_radius + 3)):
        plt.subplot(1, 3, i+1)
        plt.imshow(accumulator[i, :, :], cmap='hot')
        plt.title(f'Accumulator slice r={r}')
        plt.axis('off')
    plt.show()

    # Visualize the accumulator slice at the radius with maximum votes
    max_r_idx = np.argmax(np.sum(accumulator, axis=(1, 2)))
    plt.imshow(accumulator[max_r_idx, :, :], cmap='hot')
    plt.title(f"Accumulator slice with max votes (r={min_radius + max_r_idx})")
    plt.axis('off')
    plt.show()

    # Analysis of parameters
    print("\n" + "=" * 70)
    print("Parameter Analysis:")
    print(" - Higher Canny thresholds = fewer edges, fewer detections.")
    print(" - Lower Canny thresholds = more noise, false detections.")
    print(" - Increasing theta step = faster but less accurate.")
    print(" - Reducing radius range = faster, but may miss larger coins.")
    print("=" * 70)
    print("Task 2 complete!")
    print("=" * 70)


    # =============================================================
    print("=" * 70)
    print("Task 3: Mean Shift for Peak Detection in Hough Accumulator")
    print("=" * 70)

    print("Applying mean shift to find peaks...")
    bandwidth = 25
    peaks = myMeanShift(accumulator, bandwidth)
    
    # Visualize corresponding circles on original image    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Circles after Mean Shift (bandwidth={bandwidth})")

    for (x, y, r_i, val) in peaks:
        radius = min_radius + r_i
        circ = Circle((x, y), radius, color='yellow', fill=False, linewidth=2)
        ax.add_patch(circ)
        ax.plot(x, y, 'r.', markersize=5)

    plt.show()

    # Bandwidth parameter analysis
    print("\n" + "=" * 70)
    print("Bandwidth Parameter Analysis:")
    print(" - Small bandwidth → detects many small noisy peaks.")
    print(" - Medium bandwidth (~25) → good clustering, clean results.")
    print(" - Large bandwidth → merges close peaks, might miss smaller circles.")
    print("=" * 70)
    print("Task 3 complete!")
    print("=" * 70)
    

if __name__ == "__main__":
    main()