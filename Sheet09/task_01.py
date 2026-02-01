import numpy as np
import cv2
import glob

np.set_printoptions(suppress=True)

def task_01():

    # --- YOUR CODE HERE ---#
    # TODO read the images
    img_paths = []
    img_paths += sorted(glob.glob("./camera_calibration/*.png"))
    img_paths += sorted(glob.glob("./camera_calibration/*.jpg"))
    img_paths += sorted(glob.glob("./camera_calibration/*.jpeg"))

    if len(img_paths) == 0:
        raise RuntimeError("No calibration images found in ./camera_calibration/ (png/jpg/jpeg).")

    images = []
    for p in img_paths:
        im = cv2.imread(p)
        if im is None:
            continue
        images.append((p, im))

    if len(images) == 0:
        raise RuntimeError("Calibration images could not be read.")

    # --- YOUR CODE HERE ---#
    # TODO calibrate the camera with checkerboard patterns
    # NOTE: pattern_size is the number of INNER corners (cols, rows).
    # Common choices: (9,6) or (8,6). If detection fails, try changing this.
    pattern_size = (8, 6)   # (cols_inner, rows_inner)



    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    img_size = None
    used = 0

    for path, img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags)


        if not found:
            continue

        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp.copy())
        imgpoints.append(corners_refined)
        used += 1

    if used < 3:
        raise RuntimeError(
            f"Not enough valid checkerboard detections ({used}). "
            f"Try adding more images or changing pattern_size={pattern_size}."
        )

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )

    # --- YOUR CODE HERE ---#
    # TODO print the results of the calibration in a readable way
    print("\n=== Camera Calibration Results ===")
    print(f"Images used: {used} / {len(images)}")
    print(f"Image size (w,h): {img_size}")
    print("\nCamera matrix K:\n", K)
    print("\nDistortion coefficients (k1,k2,p1,p2,k3[,...]):\n", dist.ravel())
    print("\nFocal lengths (pixels): fx = {:.4f}, fy = {:.4f}".format(K[0, 0], K[1, 1]))
    print("Principal point (pixels): cx = {:.4f}, cy = {:.4f}".format(K[0, 2], K[1, 2]))

    # --- YOUR CODE HERE ---#
    # TODO compute the re-projection error
    total_err = 0.0
    total_points = 0

    for i in range(len(objpoints)):
        imgpoints_proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], imgpoints_proj, cv2.NORM_L2)
        n = len(objpoints[i])
        total_err += err * err
        total_points += n

    rmse = np.sqrt(total_err / max(total_points, 1))
    print("\nMean reprojection RMSE (pixels): {:.6f}".format(rmse))
    print("=================================\n")


if __name__ == "__main__":
    task_01()
