# ==============================================================================
# Exercise 3 – Spatial and Frequency Domain Filtering
# Author: Danny Ronikar Franklin
# ==============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_box_kernel(k):
    """
    Create a normalized k×k box filter kernel.
    """
    h = np.ones((k, k), dtype=np.float64)
    h /= np.sum(h)
    return h


def make_gauss_kernel(k, sigma):
    """
    Create a normalized 2D Gaussian filter kernel of size k×k.
    """
    ax = np.arange(-k // 2 + 1., k // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    h /= np.sum(h)
    return h


def conv2_same_zero(img, h):
    """
    Perform 2D spatial convolution using zero padding.
    Output should have the same size as the input image.
    (Do NOT use cv2.filter2D)
    """
    kh, kw = h.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    out = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i + kh, j:j + kw]
            out[i, j] = np.sum(region * h)
    return out


def freq_linear_conv(img, h):
    """
    Perform linear convolution in the frequency domain.
    (You can use numpy.fft)
    """
    H = np.zeros_like(img)
    kh, kw = h.shape
    H[:kh, :kw] = h
    H = np.fft.ifftshift(H)

    F_img = np.fft.fft2(img)
    F_h = np.fft.fft2(H)

    F_res = F_img * F_h
    res = np.real(np.fft.ifft2(F_res))
    return res


def compute_mad(a, b):
    """
    Compute Mean Absolute Difference (MAD) between two images.
    """
    return np.mean(np.abs(a - b))


# ==========================================================
# Main Execution
# ==========================================================

# 1. Load grayscale image
img = cv2.imread(r"E:\Bonn\Mobile Robotics\1 Semester\Computer Vsion\Exercises\Sheet02\Sheet02\data\lena.png", cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0

# 2. Construct 9×9 box and Gaussian kernels
k_size = 9
sigma = 2
box_kernel = make_box_kernel(k_size)
gauss_kernel = make_gauss_kernel(k_size, sigma)

# 3. Apply both filters in spatial domain
box_spatial = conv2_same_zero(img, box_kernel)
gauss_spatial = conv2_same_zero(img, gauss_kernel)

# 4. Apply both filters in frequency domain
box_freq = freq_linear_conv(img, box_kernel)
gauss_freq = freq_linear_conv(img, gauss_kernel)

# 5. Compute MAD
mad_box = compute_mad(box_spatial, box_freq)
mad_gauss = compute_mad(gauss_spatial, gauss_freq)

print(f"MAD (Box Filter): {mad_box:.10f}")
print(f"MAD (Gaussian Filter): {mad_gauss:.10f}")

# 6. Visualization
titles = [
    "Original Image",
    "Box Filter (Spatial)",
    "Box Filter (Frequency)",
    "Gaussian Filter (Spatial)",
    "Gaussian Filter (Frequency)"
]
images = [img, box_spatial, box_freq, gauss_spatial, gauss_freq]

plt.figure(figsize=(12, 8))
for i, (title, im) in enumerate(zip(titles, images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(im, cmap="gray")
    plt.title(title)
    plt.axis("off")
plt.tight_layout()
plt.show()

# 7. Verify condition
assert mad_box < 1e-7 and mad_gauss < 1e-7, "❌ MAD requirement not met!"
print("✅ Spatial and Frequency filtering results match (MAD < 1e-7)")
