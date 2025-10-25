# ==============================================================================
# Assignment: Computer Vision WS25/26 – Denoising and Optimization (Q2–Q4)
# Author: [Your Name]
# Environment: Python 3.12, NumPy 2.3.3, OpenCV 4.11.0.86, scikit-image, matplotlib
# Note: Only allowed libraries used as per assignment requirements.
# ==============================================================================

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio

# ==============================================================================
# 0. Setup and Image Loading
# ==============================================================================
print("--- 0. Setup: Loading Images ---")

'''
Load the clean (bonn.jpg) and noisy (bonn_noisy.jpg) images.
Convert them to grayscale, normalize to [0,1], and compute baseline PSNR.
'''

# --- Load Images ---
original_img_color = cv2.imread("bonn.jpg")                              # Load original
original_img_gray = cv2.cvtColor(original_img_color, cv2.COLOR_BGR2GRAY)  # Convert to gray

noisy_img_color = cv2.imread("bonn_noisy.jpg")                            # Load noisy
noisy_img = cv2.cvtColor(noisy_img_color, cv2.COLOR_BGR2GRAY)             # Convert to gray

# Normalize to float range [0,1]
original_img_float_01 = original_img_gray.astype(np.float32) / 255.0
noisy_img_float_01 = noisy_img.astype(np.float32) / 255.0

# --- Compute PSNR of noisy image ---
psnr_noisy = peak_signal_noise_ratio(original_img_gray, noisy_img, data_range=255)
print(f"PSNR of noisy image: {psnr_noisy:.2f} dB")

# Display Original vs Noisy
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(original_img_gray, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(noisy_img, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")
plt.show()

# ==============================================================================
# Custom Filter Definitions (for parts a, b, c)
# ==============================================================================

def custom_gaussian_filter(image, kernel_size, sigma):
    """
    Custom Gaussian Filter
    Creates a Gaussian kernel manually and performs convolution pixel by pixel.
    """
    k = kernel_size // 2

    # Step 1: Create Gaussian kernel
    ax = np.arange(-k, k + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)  # Normalize so sum = 1

    # Step 2: Reflect padding for border handling
    padded = np.pad(image, k, mode="reflect")
    filtered = np.zeros_like(image)

    # Step 3: Manual convolution
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i + kernel_size, j:j + kernel_size]
            filtered[i, j] = np.sum(region * kernel)
    return filtered


def custom_median_filter(image, kernel_size):
    """
    Custom Median Filter
    Replaces each pixel with the median of its neighborhood.
    """
    k = kernel_size // 2
    padded = np.pad(image, k, mode="reflect")
    filtered = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i + kernel_size, j:j + kernel_size]
            filtered[i, j] = np.median(region)
    return filtered


def custom_bilateral_filter(image, d, sigma_color, sigma_space):
    """
    Custom Bilateral Filter
    Edge-preserving filter using spatial + intensity weighting.
    """
    pad = d // 2
    padded = np.pad(image, pad, mode="reflect")
    filtered = np.zeros_like(image)

    # Precompute spatial Gaussian for distance
    x, y = np.mgrid[-pad:pad + 1, -pad:pad + 1]
    spatial = np.exp(-(x**2 + y**2) / (2 * sigma_space**2))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i + d, j:j + d]
            center_val = padded[i + pad, j + pad]
            intensity = np.exp(-((region - center_val) ** 2) / (2 * sigma_color**2))
            weights = spatial * intensity
            weights /= np.sum(weights)
            filtered[i, j] = np.sum(region * weights)
    return filtered


# ==============================================================================
# 1. Filter Application (Parts a, b, c)
# ==============================================================================
print("\n--- 1. Filter Application (Parts a, b, c) ---")

# Default parameters
K_DEFAULT = 7
S_DEFAULT = 2.0
D_DEFAULT = 9
SC_DEFAULT = 75
SS_DEFAULT = 75

# -------------------------- a) Gaussian Filter --------------------------
print("a) Applying Gaussian Filter...")

# OpenCV Gaussian filter
denoised_gaussian_cv2 = cv2.GaussianBlur(noisy_img_float_01, (K_DEFAULT, K_DEFAULT), S_DEFAULT)
psnr_gaussian_cv2 = peak_signal_noise_ratio(original_img_float_01, denoised_gaussian_cv2, data_range=1)

# Custom Gaussian
denoised_gaussian_custom = custom_gaussian_filter(noisy_img_float_01, K_DEFAULT, S_DEFAULT)
psnr_gaussian_custom = peak_signal_noise_ratio(original_img_float_01, denoised_gaussian_custom, data_range=1)

print(f"PSNR Gaussian (cv2): {psnr_gaussian_cv2:.2f} dB")
print(f"PSNR Gaussian (custom): {psnr_gaussian_custom:.2f} dB")

# Visualization
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.imshow(noisy_img_float_01, cmap='gray'); plt.title("Noisy"); plt.axis("off")
plt.subplot(1, 3, 2); plt.imshow(denoised_gaussian_cv2, cmap='gray'); plt.title("Gaussian (cv2)"); plt.axis("off")
plt.subplot(1, 3, 3); plt.imshow(denoised_gaussian_custom, cmap='gray'); plt.title("Gaussian (Custom)"); plt.axis("off")
plt.show()

# -------------------------- b) Median Filter --------------------------
print("\nb) Applying Median Filter...")

# OpenCV Median filter
denoised_median_cv2 = cv2.medianBlur((noisy_img_float_01 * 255).astype(np.uint8), K_DEFAULT)
denoised_median_cv2 = denoised_median_cv2.astype(np.float32) / 255.0
psnr_median_cv2 = peak_signal_noise_ratio(original_img_float_01, denoised_median_cv2, data_range=1)

# Custom Median
denoised_median_custom = custom_median_filter(noisy_img_float_01, K_DEFAULT)
psnr_median_custom = peak_signal_noise_ratio(original_img_float_01, denoised_median_custom, data_range=1)

print(f"PSNR Median (cv2): {psnr_median_cv2:.2f} dB")
print(f"PSNR Median (custom): {psnr_median_custom:.2f} dB")

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.imshow(noisy_img_float_01, cmap='gray'); plt.title("Noisy"); plt.axis("off")
plt.subplot(1, 3, 2); plt.imshow(denoised_median_cv2, cmap='gray'); plt.title("Median (cv2)"); plt.axis("off")
plt.subplot(1, 3, 3); plt.imshow(denoised_median_custom, cmap='gray'); plt.title("Median (Custom)"); plt.axis("off")
plt.show()

# -------------------------- c) Bilateral Filter --------------------------
print("\nc) Applying Bilateral Filter...")

# OpenCV Bilateral
denoised_bilateral_cv2 = cv2.bilateralFilter(noisy_img_float_01, D_DEFAULT, SC_DEFAULT / 255.0, SS_DEFAULT)
psnr_bilateral_cv2 = peak_signal_noise_ratio(original_img_float_01, denoised_bilateral_cv2, data_range=1)

# Custom Bilateral (slower)
denoised_bilateral_custom = custom_bilateral_filter(noisy_img_float_01, 7, 0.1, 3)
psnr_bilateral_custom = peak_signal_noise_ratio(original_img_float_01, denoised_bilateral_custom, data_range=1)

print(f"PSNR Bilateral (cv2): {psnr_bilateral_cv2:.2f} dB")
print(f"PSNR Bilateral (custom): {psnr_bilateral_custom:.2f} dB")

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.imshow(noisy_img_float_01, cmap='gray'); plt.title("Noisy"); plt.axis("off")
plt.subplot(1, 3, 2); plt.imshow(denoised_bilateral_cv2, cmap='gray'); plt.title("Bilateral (cv2)"); plt.axis("off")
plt.subplot(1, 3, 3); plt.imshow(denoised_bilateral_custom, cmap='gray'); plt.title("Bilateral (Custom)"); plt.axis("off")
plt.show()

# ==============================================================================
# 2. Performance Comparison (Part d)
# ==============================================================================
print("\n--- d) Performance Comparison ---")

# Collect PSNR values
psnr_results = {
    "Gaussian": psnr_gaussian_cv2,
    "Median": psnr_median_cv2,
    "Bilateral": psnr_bilateral_cv2
}

best_filter = max(psnr_results, key=psnr_results.get)
print("\nPSNR Results:")
for k, v in psnr_results.items():
    print(f"{k:10s}: {v:.2f} dB")
print(f"\nBest performing filter: {best_filter}")

# Display side-by-side comparison
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.imshow(denoised_gaussian_cv2, cmap='gray'); plt.title("Gaussian"); plt.axis("off")
plt.subplot(1, 3, 2); plt.imshow(denoised_median_cv2, cmap='gray'); plt.title("Median"); plt.axis("off")
plt.subplot(1, 3, 3); plt.imshow(denoised_bilateral_cv2, cmap='gray'); plt.title("Bilateral"); plt.axis("off")
plt.show()

# ==============================================================================
# 3. Parameter Optimization (Part e)
# ==============================================================================
print("\n--- e) Parameter Optimization ---")

def run_optimization(original_img, noisy_img):
    """
    Simple brute-force search for best PSNR by varying filter parameters.
    """
    best = {}

    # Gaussian
    best_psnr, best_k, best_sigma = 0, None, None
    for k in [3, 5, 7, 9]:
        for sigma in [0.5, 1, 1.5, 2]:
            out = cv2.GaussianBlur(noisy_img, (k, k), sigma)
            p = peak_signal_noise_ratio(original_img, out, data_range=1)
            if p > best_psnr:
                best_psnr, best_k, best_sigma = p, k, sigma
    best["Gaussian"] = (best_psnr, best_k, best_sigma)

    # Median
    best_psnr, best_k = 0, None
    for k in [3, 5, 7, 9]:
        out = cv2.medianBlur((noisy_img * 255).astype(np.uint8), k)
        out = out.astype(np.float32) / 255.0
        p = peak_signal_noise_ratio(original_img, out, data_range=1)
        if p > best_psnr:
            best_psnr, best_k = p, k
    best["Median"] = (best_psnr, best_k)

    # Bilateral
    best_psnr, best_d, best_sc, best_ss = 0, None, None, None
    for d in [5, 9]:
        for sc in [25, 50, 75]:
            for ss in [25, 50, 75]:
                out = cv2.bilateralFilter(noisy_img, d, sc / 255.0, ss)
                p = peak_signal_noise_ratio(original_img, out, data_range=1)
                if p > best_psnr:
                    best_psnr, best_d, best_sc, best_ss = p, d, sc, ss
    best["Bilateral"] = (best_psnr, best_d, best_sc, best_ss)

    return best

# Run optimization
opt_results = run_optimization(original_img_float_01, noisy_img_float_01)

print("\nOptimal Parameters and PSNR:")
for k, v in opt_results.items():
    print(f"{k:10s} -> PSNR: {v[0]:.2f} | Params: {v[1:]}")

# Apply optimal filters
opt_g = cv2.GaussianBlur(noisy_img_float_01, (opt_results["Gaussian"][1], opt_results["Gaussian"][1]), opt_results["Gaussian"][2])
opt_m = cv2.medianBlur((noisy_img_float_01 * 255).astype(np.uint8), opt_results["Median"][1]).astype(np.float32) / 255.0
opt_b = cv2.bilateralFilter(noisy_img_float_01, opt_results["Bilateral"][1], opt_results["Bilateral"][2] / 255.0, opt_results["Bilateral"][3])

# Display optimized outputs
plt.figure(figsize=(12, 6))
titles = ["Noisy", "Gaussian Opt", "Median Opt", "Bilateral Opt"]
images = [noisy_img_float_01, opt_g, opt_m, opt_b]
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis("off")
plt.show()
