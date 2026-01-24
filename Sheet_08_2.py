import numpy as np
import matplotlib.pyplot as plt

# 2.1: Generalized Procrustes Analysis (GPA)

def procrustes_align(shapes, max_iter=100, tol=1e-7):
    """Aligns multiple shapes to a common mean."""
    M, N, D = shapes.shape
    # 1. Translation: Center each shape at the origin
    centered = shapes - shapes.mean(axis=1, keepdims=True)
    
    # 2. Scaling: Normalize to unit Frobenius norm
    norms = np.linalg.norm(centered, axis=(1, 2), keepdims=True)
    scaled = centered / norms
    
    # 3. Iterative Rotation Alignment
    current_mean = scaled[0].copy()
    aligned_shapes = scaled.copy()
    
    for i in range(max_iter):
        new_aligned = []
        for j in range(M):
            # SVD to find optimal rotation matrix R
            A = scaled[j]
            B = current_mean
            U, _, Vt = np.linalg.svd(A.T @ B)
            R = U @ Vt
            new_aligned.append(A @ R)
            
        new_aligned = np.array(new_aligned)
        new_mean = new_aligned.mean(axis=0)
        new_mean /= np.linalg.norm(new_mean) # Normalize mean to prevent drift
        
        # Convergence check
        if np.linalg.norm(new_mean - current_mean) < tol:
            break
        current_mean = new_mean
        aligned_shapes = new_aligned
        
    return aligned_shapes, current_mean

# Load Data
train_data = np.load('hands_train.npy') # (38, 56, 2)
aligned_train, mean_shape = procrustes_align(train_data)

# Visualization 2.1
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); plt.title("Original Shapes")
for s in train_data: plt.plot(s[:, 0], s[:, 1], 'b-', alpha=0.2)
plt.subplot(1, 2, 2); plt.title("GPA Aligned Shapes")
for s in aligned_train: plt.plot(s[:, 0], s[:, 1], 'r-', alpha=0.2)
plt.plot(mean_shape[:, 0], mean_shape[:, 1], 'k-', lw=2, label="Mean")
plt.legend(); plt.show()

# 2.2.1: Statistical Shape Model (PCA)

# Flatten shapes for PCA (38 subjects, 112 features)
data_flat = aligned_train.reshape(len(aligned_train), -1)
mean_vec = data_flat.mean(axis=0)
data_centered = data_flat - mean_vec

# Covariance Matrix & Eigen-decomposition (Manual Implementation)
C = (data_centered.T @ data_centered) / (len(data_flat) - 1)
evals, evecs = np.linalg.eigh(C)

# Sort eigenvalues/vectors descending
idx = np.argsort(evals)[::-1]
evals, evecs = evals[idx], evecs[:, idx]

# Find N for 90% energy
cumulative_variance = np.cumsum(evals) / np.sum(evals)
N = np.argmax(cumulative_variance >= 0.90) + 1
print(f"Components for 90% energy (N): {N}")

# 2.2.2: Probabilistic PCA (PPCA)

# Sigma^2 is the average of discarded eigenvalues
sigma2 = np.mean(evals[N:])
print(f"PPCA Noise Variance (sigma^2): {sigma2}")

# 2.3: Inference & Reconstruction
test_shape = np.load('hands_test.npy')

# 1. Align test shape to the training mean
test_centered = test_shape - test_shape.mean(axis=0)
test_scaled = test_centered / np.linalg.norm(test_centered)
U_t, _, Vt_t = np.linalg.svd(test_scaled.T @ mean_shape)
test_aligned = test_scaled @ (U_t @ Vt_t)
test_vec = test_aligned.flatten()

# PCA Projection & Reconstruction
V_N = evecs[:, :N]
b_pca = V_N.T @ (test_vec - mean_vec)
pca_rec = mean_vec + (V_N @ b_pca)

# PPCA Reconstruction (Posterior Mean)
L_N = np.diag(evals[:N])
W = V_N @ np.sqrt(np.maximum(L_N - sigma2 * np.eye(N), 0))
M_mat = W.T @ W + sigma2 * np.eye(N)
z_ppca = np.linalg.inv(M_mat) @ W.T @ (test_vec - mean_vec)
ppca_rec = mean_vec + (W @ z_ppca)

# MSE Calculation
mse_pca = np.mean((test_vec - pca_rec)**2)
mse_ppca = np.mean((test_vec - ppca_rec)**2)

print(f"PCA MSE: {mse_pca:.6e}")
print(f"PPCA MSE: {mse_ppca:.6e}")

# Visualization 2.3
plt.figure(figsize=(10, 5))
plt.plot(test_aligned[:, 0], test_aligned[:, 1], 'k.', label="Original Test")
plt.plot(pca_rec.reshape(-1, 2)[:, 0], pca_rec.reshape(-1, 2)[:, 1], 'r-', label="PCA Rec")
plt.plot(ppca_rec.reshape(-1, 2)[:, 0], ppca_rec.reshape(-1, 2)[:, 1], 'g--', label="PPCA Rec")
plt.title(f"Reconstruction (N={N})")
plt.legend(); plt.axis('equal'); plt.show()