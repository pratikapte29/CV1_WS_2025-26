import argparse
import cv2
import numpy as np
import math

np.set_printoptions(suppress=True, precision=5)

# global storage for mouse clicks
CLICKED_PTS = []


def click_event(event, x, y, flags, param):
    # --- YOUR CODE HERE ---#
    # TODO extract x, y coordinates from a mouse event (click)
    global CLICKED_PTS
    if event == cv2.EVENT_LBUTTONDOWN:
        CLICKED_PTS.append([float(x), float(y)])


def pick_points(image, window_name='image'):
    """
    Generic point picker.
    """
    global CLICKED_PTS
    CLICKED_PTS = []

    img = image.copy()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, click_event)

    # --- YOUR CODE HERE ---#
    # TODO get the coordinates from a mouse event (click) for n_points and return the results in an array [n_points, 2]
    # TODO mark the selected points on the image

    print(f"[{window_name}] Left-click to add points. Press 'q' to finish, 'u' to undo last point.")
    while True:
        vis = img.copy()
        for k, (px, py) in enumerate(CLICKED_PTS):
            cv2.circle(vis, (int(px), int(py)), 6, (0, 0, 255), -1)
            cv2.putText(vis, str(k + 1), (int(px) + 8, int(py) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, vis)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('u') and len(CLICKED_PTS) > 0:
            CLICKED_PTS.pop()
        elif key == ord('q'):
            break

    cv2.destroyWindow(window_name)
    pts_xy = np.array(CLICKED_PTS, dtype=np.float64)

    return pts_xy


def _normalize_points(pts_xy):
    pts = np.asarray(pts_xy, dtype=np.float64)
    mean = np.mean(pts, axis=0)
    d = pts - mean
    dist = np.sqrt(np.sum(d * d, axis=1))
    mean_dist = np.mean(dist) if len(dist) > 0 else 1.0
    if mean_dist < 1e-12:
        s = 1.0
    else:
        s = math.sqrt(2.0) / mean_dist

    T = np.array([
        [s, 0.0, -s * mean[0]],
        [0.0, s, -s * mean[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])
    pts_n = (T @ pts_h.T).T
    return pts_n, T


def compute_Perspective(pts_src, pts_target):
    # --- YOUR CODE HERE ---#
    # TODO compute Perspective transformation from a set of corresponding points
    # Do not use RANSAC here

    pts_src = np.asarray(pts_src, dtype=np.float64)
    pts_target = np.asarray(pts_target, dtype=np.float64)

    if pts_src.shape[0] < 4 or pts_target.shape[0] < 4:
        raise ValueError("Need at least 4 point correspondences to compute homography.")
    if pts_src.shape != pts_target.shape:
        raise ValueError("Source and target point arrays must have the same shape.")

    src_n, T_src = _normalize_points(pts_src)
    dst_n, T_dst = _normalize_points(pts_target)

    N = src_n.shape[0]
    A = np.zeros((2 * N, 9), dtype=np.float64)

    for i in range(N):
        x, y, w = src_n[i]
        u, v, s = dst_n[i]
        A[2 * i + 0, :] = [0, 0, 0, -w * x, -w * y, -w * w, v * x, v * y, v * w]
        A[2 * i + 1, :] = [w * x, w * y, w * w, 0, 0, 0, -u * x, -u * y, -u * w]

    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    Hn = h.reshape(3, 3)

    H = np.linalg.inv(T_dst) @ Hn @ T_src
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]

    return H


def compute_error(H, pts_src, pts_dst):
    # --- YOUR CODE HERE ---#
    # TODO Compute the errors between the transferred and measured points and print the alignment errors.

    pts_src = np.asarray(pts_src, dtype=np.float64)
    pts_dst = np.asarray(pts_dst, dtype=np.float64)

    src_h = np.hstack([pts_src, np.ones((pts_src.shape[0], 1), dtype=np.float64)])
    proj = (H @ src_h.T).T
    proj_xy = proj[:, :2] / proj[:, 2:3]

    dif = proj_xy - pts_dst
    e = np.sqrt(np.sum(dif * dif, axis=1))
    print("Alignment errors (pixels) per point:\n", e)
    print("Mean error (px): {:.6f}".format(float(np.mean(e))))
    print("Max  error (px): {:.6f}".format(float(np.max(e))))

    return e


def _warp_to_canvas(img, H, canvas_shape, offset_xy=(0, 0)):
    hC, wC = canvas_shape[:2]
    T = np.array([[1, 0, offset_xy[0]],
                  [0, 1, offset_xy[1]],
                  [0, 0, 1]], dtype=np.float64)
    Ht = T @ H
    warped = cv2.warpPerspective(img, Ht, (wC, hC))
    return warped


def _compute_panorama_canvas(images, H_to_ref):
    # Determine bounds by warping image corners
    all_xy = []
    for img, H in zip(images, H_to_ref):
        h, w = img.shape[:2]
        corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)
        corners_h = np.hstack([corners, np.ones((4, 1), dtype=np.float64)])
        proj = (H @ corners_h.T).T
        proj_xy = proj[:, :2] / proj[:, 2:3]
        all_xy.append(proj_xy)

    all_xy = np.vstack(all_xy)
    min_xy = np.min(all_xy, axis=0)
    max_xy = np.max(all_xy, axis=0)

    min_x, min_y = float(min_xy[0]), float(min_xy[1])
    max_x, max_y = float(max_xy[0]), float(max_xy[1])

    offset_x = int(math.floor(-min_x)) if min_x < 0 else 0
    offset_y = int(math.floor(-min_y)) if min_y < 0 else 0

    W = int(math.ceil(max_x + offset_x))
    H = int(math.ceil(max_y + offset_y))

    W = max(W, 1)
    H = max(H, 1)

    return (H, W), (offset_x, offset_y)


def _blend_simple(warped_list):
    acc = None
    wsum = None
    for w in warped_list:
        mask = (np.sum(w, axis=2) > 0).astype(np.float32)
        if acc is None:
            acc = w.astype(np.float32)
            wsum = mask
        else:
            acc += w.astype(np.float32)
            wsum += mask
    wsum = np.clip(wsum, 1.0, None)
    out = (acc / wsum[..., None]).astype(np.uint8)
    return out


def task_02(args):
    # Load images
    imgA = cv2.imread(args.a)
    imgB = cv2.imread(args.b)
    imgC = cv2.imread(args.c)

    if imgA is None or imgB is None or imgC is None:
        raise RuntimeError("Could not read one or more images (A/B/C). Check the paths.")

    # --- YOUR CODE HERE ---#
    # TODO Step 1: pick points correspondences for image pairs AB and BC
    # For the submission actually provide your selected points as hard-coded arrays and comment out the interactive selection.

    # Interactive picking (use this to obtain your points once):
    # ptsA = pick_points(imgA, "Pick points in A (for A->B), press q to finish")
    # ptsB_for_AB = pick_points(imgB, "Pick corresponding points in B (for A->B), press q to finish")
    # ptsB_for_BC = pick_points(imgB, "Pick points in B (for B->C), press q to finish")
    # ptsC = pick_points(imgC, "Pick corresponding points in C (for B->C), press q to finish")

    # IMPORTANT FOR SUBMISSION:
    # Replace the arrays below with YOUR selected coordinates, and keep the interactive code commented out.
    ptsA = np.array([
        # [x, y],
    ], dtype=np.float64)

    ptsB_for_AB = np.array([
        # [x, y],
    ], dtype=np.float64)

    ptsB_for_BC = np.array([
        # [x, y],
    ], dtype=np.float64)

    ptsC = np.array([
        # [x, y],
    ], dtype=np.float64)

    if ptsA.shape[0] == 0 or ptsB_for_AB.shape[0] == 0 or ptsB_for_BC.shape[0] == 0 or ptsC.shape[0] == 0:
        print("No hard-coded points found. Falling back to interactive point picking...")
        ptsA = pick_points(imgA, "Pick points in A (for A->B), press q to finish")
        ptsB_for_AB = pick_points(imgB, "Pick corresponding points in B (for A->B), press q to finish")
        ptsB_for_BC = pick_points(imgB, "Pick points in B (for B->C), press q to finish")
        ptsC = pick_points(imgC, "Pick corresponding points in C (for B->C), press q to finish")

    if ptsA.shape[0] < 6 or ptsB_for_AB.shape[0] < 6 or ptsB_for_BC.shape[0] < 6 or ptsC.shape[0] < 6:
        raise RuntimeError("Need at least 6 correspondences for AB and BC (per assignment).")

    # --- YOUR CODE HERE ---#
    # TODO Step 2: Compute perspective transformation between A and B, print errors and visualize warped image AB.
    H_A2B = compute_Perspective(ptsA, ptsB_for_AB)
    print("\nH_A->B:\n", H_A2B)
    print("\nErrors for A->B:")
    compute_error(H_A2B, ptsA, ptsB_for_AB)

    warpedA_into_B = cv2.warpPerspective(imgA, H_A2B, (imgB.shape[1], imgB.shape[0]))
    visAB = imgB.copy()
    maskA = (np.sum(warpedA_into_B, axis=2) > 0)[:, :, None]
    visAB = np.where(maskA, warpedA_into_B, visAB)
    cv2.imwrite("task2_warp_A_into_B.png", visAB)

    # --- YOUR CODE HERE ---#
    # TODO Step 3: Compute perspective transformation between B and C.
    H_B2C = compute_Perspective(ptsB_for_BC, ptsC)
    print("\nH_B->C:\n", H_B2C)
    print("\nErrors for B->C:")
    compute_error(H_B2C, ptsB_for_BC, ptsC)

    # --- YOUR CODE HERE ---#
    # TODO Step 4: Derive A->C transformation and visualize stitched panorama ABC.
    H_A2C = H_B2C @ H_A2B
    if abs(H_A2C[2, 2]) > 1e-12:
        H_A2C = H_A2C / H_A2C[2, 2]
    print("\nH_A->C (derived):\n", H_A2C)

    # Warp everything into C coordinates (reference = C)
    images = [imgA, imgB, imgC]
    H_to_C = [H_A2C, H_B2C, np.eye(3, dtype=np.float64)]

    canvas_shape, offset_xy = _compute_panorama_canvas(images, H_to_C)

    warpedA = _warp_to_canvas(imgA, H_A2C, canvas_shape, offset_xy)
    warpedB = _warp_to_canvas(imgB, H_B2C, canvas_shape, offset_xy)
    warpedC = _warp_to_canvas(imgC, np.eye(3, dtype=np.float64), canvas_shape, offset_xy)

    pano = _blend_simple([warpedA, warpedB, warpedC])

    cv2.imwrite("task2_panorama_ABC.png", pano)
    cv2.imshow("Warp A into B (saved: task2_warp_A_into_B.png)", visAB)
    cv2.imshow("Panorama ABC (saved: task2_panorama_ABC.png)", pano)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=str, default="./data/A.png", help="Image A path")
    parser.add_argument("--b", type=str, default="./data/B.png", help="Image B path")
    parser.add_argument("--c", type=str, default="./data/C.png", help="Image C path")
    args = parser.parse_args()
    task_02(args)
