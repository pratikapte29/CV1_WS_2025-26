import cv2
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
# Function: compute_ncc
# Purpose : Compute the Normalized Cross-Correlation (NCC) score between
#           two image patches of identical size.
# ------------------------------------------------------------------------------
def compute_ncc(patch_left, patch_right):
    patch_left = patch_left - np.mean(patch_left)
    patch_right = patch_right - np.mean(patch_right)

    numerator = np.sum(patch_left * patch_right)
    denominator = np.sqrt(np.sum(patch_left ** 2) * np.sum(patch_right ** 2))

    if denominator == 0:
        return 0
    return numerator / denominator


# ------------------------------------------------------------------------------
# Function: ncc_disparity
# Purpose : Compute a disparity map using NCC-based block matching.
# ------------------------------------------------------------------------------
def ncc_disparity(left_img, right_img, max_disp, block_size):
    height, width = left_img.shape
    half = block_size // 2
    disparity_map = np.zeros((height, width), dtype=np.float64)

    # Use reflective padding to minimize border artifacts
    left_padded = np.pad(left_img, half, mode='reflect')
    right_padded = np.pad(right_img, half, mode='reflect')

    # Iterate over all valid pixel positions
    for y in range(half, height - half):
        for x in range(half, width - half):
            best_ncc = -1
            best_disparity = 0

            # Extract the local block from the left image
            block_left = left_padded[y - half:y + half + 1, x - half:x + half + 1]

            # Search along the epipolar line (left-right direction)
            for d in range(max_disp):
                x_right = x - d  # right image shifted left by disparity d
                if x_right - half < 0:
                    continue

                block_right = right_padded[y - half:y + half + 1,
                                           x_right - half:x_right + half + 1]
                ncc_score = compute_ncc(block_left, block_right)

                if ncc_score > best_ncc:
                    best_ncc = ncc_score
                    best_disparity = d

            disparity_map[y, x] = best_disparity

    return disparity_map


# ------------------------------------------------------------------------------
# Function: compute_mae
# Purpose : Compute Mean Absolute Error (MAE) between two disparity maps.
# ------------------------------------------------------------------------------
def compute_mae(map_a, map_b):
    return np.mean(np.abs(map_a - map_b))


# ==============================================================================
# Main Program
# ==============================================================================
if __name__ == "__main__":

    # 1. Load the rectified stereo image pair
    left = cv2.imread("data/left.jpg", cv2.IMREAD_GRAYSCALE)
    right = cv2.imread("data/right.jpg", cv2.IMREAD_GRAYSCALE)

    if left is None or right is None:
        raise FileNotFoundError("Required images 'left.png' and/or 'right.png' not found.")

    left = left.astype(np.float64) / 255.0
    right = right.astype(np.float64) / 255.0

    # 2. Define stereo matching parameters
    block_size = 9
    max_disparity = 64  # should be divisible by 16 (as in StereoBM)

    # 3. Compute manual NCC-based disparity map
    print("Computing manual NCC disparity map...")
    disparity_ncc = ncc_disparity(left, right, max_disparity, block_size)

    # 4. Compute reference disparity using OpenCV StereoBM
    stereo_bm = cv2.StereoBM_create(numDisparities=max_disparity, blockSize=block_size)
    disparity_bm = stereo_bm.compute((left * 255).astype(np.uint8),
                                     (right * 255).astype(np.uint8)).astype(np.float64)
    disparity_bm /= 16.0  # convert fixed-point to real disparity values

    # 5. Normalize disparity maps for quantitative comparison
    disparity_bm_norm = cv2.normalize(disparity_bm, None, 0, 1, cv2.NORM_MINMAX)
    disparity_ncc_norm = cv2.normalize(disparity_ncc, None, 0, 1, cv2.NORM_MINMAX)

    # 6. Compute Mean Absolute Error (MAE)
    valid_mask = disparity_bm > 0
    mae_value = compute_mae(disparity_bm_norm[valid_mask], disparity_ncc_norm[valid_mask])
    print(f"Mean Absolute Error (MAE): {mae_value:.4f}")

    # 7. Visualization of results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(disparity_ncc, cmap="jet")
    plt.title("Manual NCC Disparity Map")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(disparity_bm, cmap="jet")
    plt.title("OpenCV StereoBM Disparity Map")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # 8. Validation
    assert mae_value < 0.7, f"MAE requirement not met (must be < 0.7, got {mae_value:.3f})"
    print("Manual NCC disparity map verified (MAE < 0.7).")