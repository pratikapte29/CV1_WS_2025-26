import numpy as np
import cv2


def load_landmarks(path: str):
    """
    Load the landmark locations from the txt file
    """
    return np.loadtxt(path)


def transform_shape(shape, theta):
    """
    Transforms the landmark points by parameters from theta

    :param shape: np.ndarray of points
    :param theta: array of tx, ty, scale, alpha
    """
    tx, ty, scale, alpha = theta

    R = np.array([
        [np.cos(alpha), -np.sin(alpha)],
        [np.sin(alpha),  np.cos(alpha)]
    ])

    return scale * (shape @ R.T) + np.array([tx, ty])


def closest_edge_points(pts, edge_points):
    """
    For each point, find closest edge pixel
    """
    correspondences = []

    for p in pts:
        dists = np.sum((edge_points - p) ** 2, axis=1)
        idx = np.argmin(dists)
        correspondences.append(edge_points[idx])

    return np.array(correspondences)


def estimate_transform(src, dst):
    """
    Estimate tx, ty, scale, rotation from src → dst
    """
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)

    src_c = src - src_mean
    dst_c = dst - dst_mean

    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    scale = np.trace(R.T @ H) / np.sum(src_c ** 2)

    t = dst_mean - scale * (R @ src_mean)

    alpha = np.arctan2(R[1, 0], R[0, 0])

    return np.array([t[0], t[1], scale, alpha])


def main():
    shape = load_landmarks("rat.txt")

    img = cv2.imread("rat.webp", cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.5)

    # Canny edge detection
    edges = cv2.Canny(img, 20, 150)

    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # compute distance transform
    dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)

    dist_vis = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow("Distance Transform", dist_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Collect edge points
    edge_points = np.column_stack(np.where(edges > 0))[:, ::-1]

    # initialization 
    theta = np.array([0.0, 0.0, 1.0, 0.0])

    # Iteratice Closest Point loop
    for it in range(15):
        pts = transform_shape(shape, theta)

        corr = closest_edge_points(pts, edge_points)

        theta = estimate_transform(shape, corr)

        # Visualising after every iteration
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Draw closed loop for model points (rubber band effect)
        pts_int = pts.astype(int)
        cv2.polylines(vis, [pts_int], True, (0, 0, 255), 2)
        
        # Draw closed loop for correspondence points
        corr_int = corr.astype(int)
        cv2.polylines(vis, [corr_int], True, (255, 0, 0), 2)
        
        # Draw connecting lines between corresponding points
        for p, c in zip(pts_int, corr_int):
            cv2.line(vis, tuple(p), tuple(c), (0, 255, 0), 1)

        cv2.imshow("ICP fitting", vis)
        cv2.waitKey(100)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
