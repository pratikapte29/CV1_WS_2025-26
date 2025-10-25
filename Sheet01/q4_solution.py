import cv2
import numpy as np

# ==============================================================================
# Load the image and convert it to gray scale
# ==============================================================================

img = cv2.imread('bonn.jpg')

# Visualize if needed:
# cv2.imshow('Original Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Visualize if needed:
# cv2.imshow('Grayscale Image', gray_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ==============================================================================
# (a) Module to decompose the input kernel using SVD
# ==============================================================================

def decompose_kernel_svd(kernel):
    """
    Decompose the input kernel using Singular Value Decomposition Class in cv2
    
    Args:
        kernel: 2D numpy array representing the kernel to be decomposed.
    
    Returns:
        u: Left singular vectors.
        s: Singular values (as a 1D array).
        vt: Right singular vectors (transposed).
    """
    u = np.zeros((kernel.shape[0], kernel.shape[0]), dtype=np.float32)
    s = np.zeros((min(kernel.shape), 1), dtype=np.float32)
    vt = np.zeros((kernel.shape[1], kernel.shape[1]), dtype=np.float32)

    # Ensure kernel is float32
    kernel = np.array(kernel, dtype=np.float32)
    
    # Perform SVD using OpenCV
    cv2.SVDecomp(kernel, s, u, vt)

    # Flatten singular values to 1D array
    s = s.flatten()

    return u, s, vt

# Example kernels
K1 = np.array([[0.0113, 0.0838, 0.0113],
               [0.0838, 0.6193, 0.0838],
               [0.0113, 0.0838, 0.0113]], dtype=np.float32)

K2 = np.array([[-0.8984, 0.1472, 1.1410],
               [-1.9075, 0.1566, 2.1359],
               [-0.8659, 0.0573, 1.0337]], dtype=np.float32)

# Uncomment to test with another kernel (more of a test case for my better understanding)
# K3 = np.array([[1, 2, 1],
#                [2, 4, 2],
#                [1, 2, 1]], dtype=np.float32)

# Decompose the kernel using SVD
u1, s1, vt1 = decompose_kernel_svd(K1)
print("Decomposed Kernel using SVD:")
print("U Matrix:\n", u1)
print("Singular Values:\n", s1)
print("VT Matrix:\n", vt1)

# Decompose the second kernel using SVD
u2, s2, vt2 = decompose_kernel_svd(K2)
print("\nDecomposed Second Kernel using SVD:")
print("U Matrix:\n", u2)
print("Singular Values:\n", s2)
print("VT Matrix:\n", vt2)

# ==============================================================================
# Check if kernel is separable and if not, make an approximation
# ==============================================================================

def check_separability_and_approximate(u, s, vt, gray_img, K):
    """
    Check if the kernel is separable using its singular values.
    If not separable, approximate it using the largest singular value.
    Then, filter the image using both the original and approximated kernels,
    and compute the maximum absolute error between the two filtered images.
    Args:
        u: Left singular vectors.
        s: Singular values (1D array).
        vt: Right singular vectors (transposed).
        gray_img: Grayscale image to be filtered.
        K: Original kernel.
    """

    is_separable = np.sum(s > 1e-6) == 1
    if not is_separable:
        print("Kernel is not separable. Approximating using the largest singular value.")
        sigma = s[0]               # largest singular value
        u_vec = u[:, 0].reshape(-1, 1)  # first column of U
        v_vec = vt[0, :].reshape(1, -1) # first row of VT
        K_approx = sigma * np.dot(u_vec, v_vec)  # rank-1 approximation
        
        # Filter the image using both kernels
        filtered_full = cv2.filter2D(gray_img, -1, K)
        filtered_approx = cv2.filter2D(gray_img, -1, K_approx)

        # If needed, we can also compare visually:
        combined = np.hstack((filtered_full, filtered_approx))
        cv2.imshow('Filtered Images: Full Kernel (Left) | Approximated Kernel (Right)', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Compute absolute pixel error
        diff = np.abs(filtered_full.astype(np.float32) - filtered_approx.astype(np.float32))
        max_error = np.max(diff)
        print(f"Maximum absolute error between full and approximated filtering: {max_error:.4f}")

    else:
        print("Kernel is separable.")

# Check separability and approximate for both kernels

print("\n--- Checking separability and approximating Kernel 1 ---")       
check_separability_and_approximate(u1, s1, vt1, gray_img, K1)
print("\n--- Checking separability and approximating Kernel 2 ---")
check_separability_and_approximate(u2, s2, vt2, gray_img, K2)
