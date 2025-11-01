# Template for Exercise 2 –  Fourier Transform and Image Reconstruction
import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_fft(img):
    """
    Compute the Fourier Transform of an image and return:
    - The shifted spectrum
    - The magnitude
    - The phase
    """
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    phase = np.angle(fshift)
    return fshift, magnitude, phase


def reconstruct_from_mag_phase(mag, phase):
    """
    Reconstruct an image from given magnitude and phase.
    """
    # Combine magnitude and phase into complex numbers
    complex_spectrum = mag * np.exp(1j * phase)
    # Inverse shift
    f_ishift = np.fft.ifftshift(complex_spectrum)
    # Inverse FFT
    img_recon = np.fft.ifft2(f_ishift)
    # Take real part and clip to valid range [0,255]
    img_recon = np.real(img_recon)
    img_recon = np.clip(img_recon, 0, 255)
    return img_recon.astype(np.uint8)


def compute_mad(a, b):
    """
    Compute the Mean Absolute Difference (MAD) between two images.
    """
    # Convert to float32 to avoid overflow during subtraction
    a_int = a.astype(np.float32)
    b_int = b.astype(np.float32)
    mad = np.mean(np.abs(a_int - b_int))
    return mad

# ==========================================================

# TODO: 1. Load the two grayscale images (1.png and 2.png)
img1 = cv2.imread('data/1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('data/2.png', cv2.IMREAD_GRAYSCALE)

# TODO: 2. Compute magnitude and phase of both images
f1, mag1, phase1 = compute_fft(img1)
f2, mag2, phase2 = compute_fft(img2)

# Visualize magnitude and phase
plt.figure(figsize=(10,4))
plt.subplot(2,2,1); plt.imshow(np.log1p(mag1), cmap='gray'); plt.title('Magnitude 1')
plt.subplot(2,2,2); plt.imshow(phase1, cmap='gray'); plt.title('Phase 1')
plt.subplot(2,2,3); plt.imshow(np.log1p(mag2), cmap='gray'); plt.title('Magnitude 2')
plt.subplot(2,2,4); plt.imshow(phase2, cmap='gray'); plt.title('Phase 2')
plt.show()

# TODO: 3. Swap magnitude and phase between the two images
# mag1 + phase2
recon_mag1_phase2 = reconstruct_from_mag_phase(mag1, phase2)
# mag2 + phase1
recon_mag2_phase1 = reconstruct_from_mag_phase(mag2, phase1)

# TODO: 4. Reconstruct and save the swapped results
cv2.imwrite('reconstructed_mag1_phase2.png', recon_mag1_phase2)
cv2.imwrite('reconstructed_mag2_phase1.png', recon_mag2_phase1)

# TODO: 5. Compute and print the MAD values between originals and reconstructions
mad1 = compute_mad(img1, recon_mag1_phase2)
mad2 = compute_mad(img2, recon_mag2_phase1)
print("MAD between img1 and reconstructed_mag1_phase2:", mad1)
print("MAD between img2 and reconstructed_mag2_phase1:", mad2)

# TODO: 6. Visualize all images (originals, magnitude, phase, reconstructions)
plt.figure(figsize=(12,6))
plt.subplot(2,3,1); plt.imshow(img1, cmap='gray'); plt.title('Original Image 1')
plt.subplot(2,3,2); plt.imshow(img2, cmap='gray'); plt.title('Original Image 2')
plt.subplot(2,3,3); plt.imshow(recon_mag1_phase2, cmap='gray'); plt.title('Mag1 + Phase2')
plt.subplot(2,3,4); plt.imshow(recon_mag2_phase1, cmap='gray'); plt.title('Mag2 + Phase1')
plt.subplot(2,3,5); plt.imshow(np.log1p(mag1), cmap='gray'); plt.title('Magnitude 1')
plt.subplot(2,3,6); plt.imshow(np.log1p(mag2), cmap='gray'); plt.title('Magnitude 2')
plt.show()
