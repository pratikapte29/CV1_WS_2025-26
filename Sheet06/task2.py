import cv2
import numpy as np
import os

from task1 import MOG

# Helper functions below:

def compute_centroid(box):
    # Compute centroid of a bounding box
    x, y, w, h = box
    return (x + w // 2, y + h // 2)

def clean_mask(mask):
    # Clean the foreground mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

    return mask_cleaned

def detect_and_count_people(img_files_path, num_gaussians=5, bg_thresh=0.5, lr=0.01,
                            min_area=40, min_aspect_ratio=0.5, dist_threshold=5):
        
    first_img = cv2.imread(img_files_path + "0001.jpg")
    height, width = first_img.shape[:2]

    # Initialize the MOG model class
    mog = MOG(
        height=height, 
        width=width, 
        number_of_gaussians=num_gaussians,
        background_thresh=bg_thresh, 
        lr=lr
        )
    
    # Create list to store all the valid i.e. centroids that are to be counted as people.
    valid_centroids = []
    img_files = sorted([f for f in os.listdir(img_files_path) if f.endswith('.jpg')])

    for i, img_file in enumerate(img_files):
        img = cv2.imread(img_files_path + img_file)

        fg_mask = mog.updateParam(img, np.ones((height, width)))  # get the foreground mask similar to task1 
        fg_mask = clean_mask(fg_mask)  # clean the above fg mask
        
        # Find contours in the cleaned foreground mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print("number of contours found: ", len(contours))
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            aspect_ratio = float(h / w)
            if aspect_ratio < min_aspect_ratio:
                continue
            centroid = compute_centroid((x, y, w, h))
            already_counted = False
            for c in valid_centroids:
                if np.linalg.norm(np.array(centroid) - np.array(c)) < dist_threshold:
                    already_counted = True
                    break
            if not already_counted:
                valid_centroids.append(centroid)

            vis_img = img.copy()
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(vis_img, centroid, 5, (0, 0, 255), -1)
            cv2.putText(vis_img, f"Area: {area:.0f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(vis_img, f"AR: {aspect_ratio:.2f}", (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.imshow(f"Frame {i+1}", vis_img)
            cv2.imshow(f"Mask {i+1}", fg_mask)
            cv2.waitKey(0)  # Press any key to move to next frame
            cv2.destroyAllWindows()

        print(f"Image {i + 1} people detected: {len(valid_centroids)}")
    return len(valid_centroids)

# temprorary test:

detect_and_count_people('imgs/')