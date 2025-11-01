# Template for Exercise 4 – NCC Stereo Matching

from turtle import right
import cv2
import numpy as np
import matplotlib.pyplot as plt


WINDOW_SIZE = 11       # NCC patch size
MAX_DISPARITY = 64     # Maximum search range


def compute_manual_ncc_map(left_image, right_image, window_size, max_disparity):
    """
    Compute a dense disparity map using Normalized Cross-Correlation (NCC).
    
    Arguments:
        left_image, right_image : input grayscale stereo pair
        window_size             : size of the correlation window
        max_disparity           : maximum horizontal shift to consider

    Returns:
        disparity_map : computed disparity for each pixel (float32)
    """
    # Convert to float for math operations
    left = left_image.astype(np.float32) 
    right = right_image.astype(np.float32) 

    # Half window size for indexing
    half_w = window_size // 2
    h, w = left.shape

    # Output disparity map
    disparity_map = np.zeros((h, w), dtype=np.float32)

    # Small epsilon to avoid division by zero
    eps = 1e-8

    # Loop through all pixels (excluding borders)
    for y in range(half_w, h - half_w):
        for x in range(half_w, w - half_w):

            # Extract left patch
            left_patch = left[y - half_w:y + half_w + 1, x - half_w:x + half_w + 1]

            meanL = np.mean(left_patch)
            left_centered = left_patch - meanL
            sum_sq_L = np.sum(left_centered ** 2)
            
            if sum_sq_L < eps:
                continue

            best_ncc = -1.0
            best_disp = 0

            # Loop over all possible disparities
            for d in range(max_disparity):
                xr = x - d
                if xr - half_w < 0:
                    continue  # out of right image bounds

                # Extract right patch
                if xr - half_w < 0 or xr + half_w >= w:
                    continue
                right_patch = right[y - half_w:y + half_w + 1, xr - half_w:xr + half_w + 1]

                meanR = np.mean(right_patch)
                right_centered = right_patch - meanR
                sum_sq_R = np.sum(right_centered ** 2)
                
                if sum_sq_R < eps:
                    continue

                # Compute NCC
                num = np.sum(left_centered * right_centered)
                den = np.sqrt(sum_sq_L * sum_sq_R)
                
                if den < eps:
                    continue
                    
                ncc = num / den

                # Update best match
                if ncc > best_ncc:
                    best_ncc = ncc
                    best_disp = d

            disparity_map[y, x] = best_disp

    return disparity_map


def compute_mae(a, b, mask=None):
    """
    Compute Mean Absolute Error (MAE) between two disparity maps.
    Optionally, use a mask to exclude invalid pixels.
    """
    diff = np.abs(a - b)

    if mask is not None:
        diff = diff[mask]

    return np.mean(diff)


# ==========================================================


# TODO: 1. Load the stereo image pair (left.png, right.png) in grayscale
left_img = cv2.imread(r'./data/left.jpg', cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread(r'./data/right.jpg', cv2.IMREAD_GRAYSCALE)

# cv2.imshow('Left Image', left_img)
# cv2.imshow('Right Image', right_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# TODO: 2. Call your NCC function to compute the manual disparity map

manual_map = compute_manual_ncc_map(left_img, right_img, WINDOW_SIZE, MAX_DISPARITY)

# TODO: 3. Compute a benchmark map using cv2.StereoBM_create with the same parameters

stereo = cv2.StereoBM_create(numDisparities=MAX_DISPARITY, blockSize=WINDOW_SIZE)

benchmark_map = stereo.compute(left_img, right_img)
benchmark_map = benchmark_map.astype(np.float32) / 16.0  # scale back

# TODO: 4. Visualize both maps and compare them qualitatively

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Manual NCC Disparity Map")
plt.imshow(manual_map, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("OpenCV StereoBM Disparity Map")
plt.imshow(benchmark_map, cmap='gray')
plt.axis("off")
plt.tight_layout()
plt.show()

# TODO: 5. Quantitatively compare both maps by computing MAE (Mean Absolute Error)

mae = compute_mae(manual_map, benchmark_map, mask=(benchmark_map > 0))
print(f"Mean Absolute Error (MAE) between manual NCC and OpenCV StereoBM: {mae:.3f} pixels")

# TODO: 6. Ensure your manual implementation achieves MAE < 0.7 pixels

if mae < 0.7:
    print("MAE requirement satisfied (< 0.7)")
else:
    print("MAE too high — check parameters or NCC implementation.")
