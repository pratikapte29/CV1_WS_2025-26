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
    '''
    Custom implementation of the Hough Transform for circle detection.
    The function takes the edge map and votes for possible circle centers
    in a 3D accumulator (radius, y_center, x_center).

    Args:
        edges       : Single-channel binary edge image (result from Canny)
        min_radius  : Minimum radius to search
        max_radius  : Maximum radius to search
        threshold   : Minimum number of votes to keep a circle
        min_dist    : Minimum allowed distance between two detected centers
        r_ssz       : Step size for radius (e.g., 1)
        theta_ssz   : Step size for theta in degrees (smaller = more accurate but slower)
    Returns:
        detected_circles : List of (x_center, y_center, radius, votes)
        accumulator      : 3D array (n_radii, height, width)
    '''

    # Get image dimensions
    h, w = edges.shape

    # Limit maximum radius to the image diagonal length
    max_radius = min(max_radius, int(np.linalg.norm([h, w])))

    # Create list of all edge point coordinates
    edge_points = np.array(np.nonzero(edges))   # shape (2, N)
    num_points = edge_points.shape[1]

    # Create radius range
    radii = np.arange(min_radius, max_radius + 1, r_ssz)
    n_r = len(radii)

    # Initialize accumulator with zeros
    accumulator = np.zeros((n_r, h, w), dtype=np.uint64)

    '''
    Voting process:
    For every edge pixel, we assume it lies on the circumference of
    potential circles with different radii. For each radius, we compute
    all possible center coordinates (a, b) using:
        a = x - r*cos(theta)
        b = y - r*sin(theta)
    Each valid (a, b, r) is incremented in the accumulator.
    '''
    for idx in range(num_points):
        y, x = edge_points[:, idx]

        for r_i, r in enumerate(radii):
            for theta in range(0, 360, theta_ssz):
                a = int(x - r * np.cos(np.deg2rad(theta)))
                b = int(y - r * np.sin(np.deg2rad(theta)))

                # Only vote if the center is within image bounds
                if 0 <= a < w and 0 <= b < h:
                    accumulator[r_i, b, a] += 1

    '''
    Now we look for positions in the accumulator where the votes
    exceed the threshold value. These correspond to likely circle centers.
    '''
    detected_circles = []
    for r_i, r in enumerate(radii):
        layer = accumulator[r_i, :, :]
        y_peaks, x_peaks = np.where(layer > threshold)

        for (xc, yc) in zip(x_peaks, y_peaks):
            votes = layer[yc, xc]

            # Check minimum distance constraint to avoid duplicate detections
            too_close = False
            for (x_prev, y_prev, r_prev, _) in detected_circles:
                if np.sqrt((x_prev - xc)**2 + (y_prev - yc)**2) < min_dist:
                    too_close = True
                    break

            if not too_close:
                detected_circles.append((xc, yc, r, votes))

    return detected_circles, accumulator


def myMeanShift(accumulator, bandwidth, threshold=None):
    '''
    Mean Shift algorithm to find peaks (modes) in the Hough accumulator.

    Args:
        accumulator : 3D Hough accumulator (n_r, h, w)
        bandwidth   : Spatial bandwidth (how far to consider neighbors)
        threshold   : Minimum vote value to consider (if None, use 0.5 * max)
    Returns:
        peaks : List of (x, y, r_index, votes)
    '''

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

    '''
    Step 1: Load the image and convert it to grayscale.
    '''
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    '''
    Step 2: Detect edges using the Canny operator.
    The thresholds here are tuned for coins.jpg.
    '''
    edges = cv2.Canny(gray, 100, 200)

    '''
    Step 3: Define parameters for Hough Transform.
    '''
    min_radius = 20
    max_radius = 60
    threshold = 120
    min_dist = 20
    r_ssz = 1
    theta_ssz = 5

    '''
    Step 4: Run our Hough Circle detector.
    '''
    detected_circles, accumulator = myHoughCircles(edges, min_radius, max_radius,
                                                  threshold, min_dist, r_ssz, theta_ssz)

    '''
    Step 5: Visualize detected circles.
    '''
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title("Detected Circles (Hough Transform)")

    for (x, y, r, v) in detected_circles:
        circ = Circle((x, y), r, color='lime', fill=False, linewidth=2)
        ax.add_patch(circ)
        ax.plot(x, y, 'r.', markersize=5)

    plt.show()

    '''
    Step 6: Visualize accumulator slices for some radii.
    '''
    n_r, h, w = accumulator.shape
    plt.figure(figsize=(12, 4))
    for i, r in enumerate(range(min_radius, min_radius + 3)):
        plt.subplot(1, 3, i+1)
        plt.imshow(accumulator[i, :, :], cmap='hot')
        plt.title(f'Accumulator slice r={r}')
        plt.axis('off')
    plt.show()

    '''
    Step 7: Find the radius index with the maximum overall votes.
    '''
    max_r_idx = np.argmax(np.sum(accumulator, axis=(1, 2)))
    plt.imshow(accumulator[max_r_idx, :, :], cmap='hot')
    plt.title(f"Accumulator slice with max votes (r={min_radius + max_r_idx})")
    plt.axis('off')
    plt.show()

    print("\n" + "=" * 70)
    print("Parameter Analysis:")
    print(" - Higher Canny thresholds = fewer edges, fewer detections.")
    print(" - Lower Canny thresholds = more noise, false detections.")
    print(" - Increasing theta step = faster but less accurate.")
    print(" - Reducing radius range = faster, but may miss larger coins.")
    print("=" * 70)
    print("Task 2 complete!")
    print("=" * 70)


    '''
    Task 3: Mean Shift for Peak Detection
    ---------------------------------------------------------------
    We apply mean shift on the accumulator to find strong peaks that
    correspond to actual circle centers. This helps remove redundant
    detections from Task 2.
    '''
    print("=" * 70)
    print("Task 3: Mean Shift for Peak Detection in Hough Accumulator")
    print("=" * 70)

    bandwidth = 25
    peaks = myMeanShift(accumulator, bandwidth)

    '''
    Step 8: Visualize circles corresponding to mean-shift peaks.
    '''
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Circles after Mean Shift (bandwidth={bandwidth})")

    for (x, y, r_i, val) in peaks:
        radius = min_radius + r_i
        circ = Circle((x, y), radius, color='yellow', fill=False, linewidth=2)
        ax.add_patch(circ)
        ax.plot(x, y, 'r.', markersize=5)

    plt.show()

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
