import cv2
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
# Function: make_box_kernel
# Purpose : Create a normalized k×k box filter kernel.
# ------------------------------------------------------------------------------
def make_box_kernel(k):
    kernel = np.ones((k, k), dtype=np.float64)
    kernel /= np.sum(kernel)
    return kernel


# ------------------------------------------------------------------------------
# Function: make_gauss_kernel
# Purpose : Create a normalized 2D Gaussian filter kernel of size k×k.
# ------------------------------------------------------------------------------
def make_gauss_kernel(k, sigma):
    ax = np.arange(-k // 2 + 1., k // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel


# ------------------------------------------------------------------------------
# Function: conv2_same_zero
# Purpose : Perform 2D convolution with zero padding (same output size).
# ------------------------------------------------------------------------------
def conv2_same_zero(img, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # Zero padding
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)),
                    mode='constant', constant_values=0)

    # Convolution output
    output = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * kernel)

    return output


# ------------------------------------------------------------------------------
# Function: freq_linear_conv
# Purpose : Perform *linear* convolution in the frequency domain using FFT.
# ------------------------------------------------------------------------------
def freq_linear_conv(img, kernel):
    H, W = img.shape
    kh, kw = kernel.shape

    # Linear convolution requires padding to (H + kh - 1, W + kw - 1)
    pad_h, pad_w = H + kh - 1, W + kw - 1

    # Zero-pad image and kernel to common size
    F_img = np.fft.fft2(img, s=(pad_h, pad_w))
    F_kernel = np.fft.fft2(kernel, s=(pad_h, pad_w))

    # Multiply in frequency domain
    F_res = F_img * F_kernel

    # Inverse FFT and take real part
    conv_result = np.real(np.fft.ifft2(F_res))

    # Crop center region to original image size
    start_h = kh // 2
    start_w = kw // 2
    result = conv_result[start_h:start_h + H, start_w:start_w + W]

    return result


# ------------------------------------------------------------------------------
# Function: compute_mad
# Purpose : Compute the Mean Absolute Difference (MAD) between two arrays.
# ------------------------------------------------------------------------------
def compute_mad(a, b):
    return np.mean(np.abs(a - b))


# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":

    # 1. Load grayscale image
    img = cv2.imread("data/lena.png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image 'lena.png' not found in the working directory.")

    img = img.astype(np.float64) / 255.0

    # 2. Construct 9×9 box and Gaussian kernels
    k_size = 9
    sigma = 2.0
    box_kernel = make_box_kernel(k_size)
    gauss_kernel = make_gauss_kernel(k_size, sigma)

    # 3. Apply filters in spatial domain
    box_spatial = conv2_same_zero(img, box_kernel)
    gauss_spatial = conv2_same_zero(img, gauss_kernel)

    # 4. Apply filters in frequency domain
    box_freq = freq_linear_conv(img, box_kernel)
    gauss_freq = freq_linear_conv(img, gauss_kernel)

    # 5. Compute Mean Absolute Difference (MAD)
    mad_box = compute_mad(box_spatial, box_freq)
    mad_gauss = compute_mad(gauss_spatial, gauss_freq)

    print(f"MAD (Box Filter):     {mad_box:.10e}")
    print(f"MAD (Gaussian Filter): {mad_gauss:.10e}")

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

    # 7. Verification of equivalence between spatial and frequency results
    if mad_box >= 1e-7 or mad_gauss >= 1e-7:
        raise AssertionError("MAD requirement not met.")
    print("Spatial and frequency domain filtering results match (MAD < 1×10⁻⁷).")
