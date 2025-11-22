import numpy as np
import cv2
import skimage
import matplotlib.pyplot as plt
import math

np.set_printoptions(suppress=True)

img1 = cv2.imread('img_mosaic_initial_segmentation.tif', 0)
cv2.imshow('Initial Segmentation', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
img1_smooth = cv2.GaussianBlur(img1, (3, 3), 0)
img1_smooth = cv2.normalize(img1_smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
blur = cv2.GaussianBlur(img1_smooth, (3, 3), 0)
sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
Energy_ext = -(sobelx**2 + sobely**2)
Energy_ext = cv2.normalize(Energy_ext, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) 
Roughness = cv2.Laplacian(img1, cv2.CV_64F, ksize=3)
Energy_int = np.abs(Roughness)
Energy_int = cv2.normalize(Energy_int, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# TODO your implementation
Roughness_penalty = 0.3
Total_Energy = cv2.addWeighted(Energy_ext, 0.7, Energy_int, Roughness_penalty, 0)
# cv2.imshow('Total Energy', Total_Energy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

_, refined_img = cv2.threshold(Total_Energy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

gradient_y, gradient_x = np.gradient(Total_Energy.astype(np.float32))   
labels = (refined_img > 0).astype(np.float32)          
h, w = Total_Energy.shape
step = 0.50                                              

for i in range(5):                                    
    new_labels = np.zeros_like(labels)
    # iterate only over foreground pixels
    row_indices, column_indices = np.where(labels > 0.5)
    for i in range(len(column_indices)):
        x, y = column_indices[i], row_indices[i]
        # clip the image
        new_pixel_x = int(np.clip(x - step * gradient_x[y, x], 0, w-1))  #New position = Old position - (step size * gradient)
        new_pixel_y = int(np.clip(y - step * gradient_y[y, x], 0, h-1))
        new_labels[new_pixel_y, new_pixel_x] = 1.0  #new location as foreground 
    labels = new_labels
refined_img = (labels*255).astype(np.uint8)
cv2.imshow('Refined Image', refined_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("img_mosaic_pred.tif", refined_img)

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# refined_image = cv2.morphologyEx(refined_img, cv2.MORPH_OPEN, kernel)
# refined_image = cv2.morphologyEx(refined_img, cv2.MORPH_CLOSE, kernel)
# cv2.imshow('Refined Image after Morphology', refined_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#########################################################
# IOU Computation
#########################################################

ground_truth  = cv2.imread('data/img_mosaic_label.tif', cv2.IMREAD_GRAYSCALE)
predicted_segment = cv2.imread('img_mosaic_pred.tif',  cv2.IMREAD_GRAYSCALE)
ground_truth = cv2.resize(ground_truth, (predicted_segment.shape[1], predicted_segment.shape[0]))

# if ground_truth is None or predicted_segment is None:
#     raise FileNotFoundError('image not found')

ground_truth   = cv2.threshold(ground_truth,   127, 255, cv2.THRESH_BINARY)[1]
predicted_segment = cv2.threshold(predicted_segment, 127, 255, cv2.THRESH_BINARY)[1]
if ground_truth[0,0] == 0:       
    ground_truth = cv2.bitwise_not(ground_truth)
if predicted_segment[0,0] == 0:     
    predicted_segment = cv2.bitwise_not(predicted_segment)

intersection = np.logical_and(ground_truth, predicted_segment).sum()
union        = np.logical_or(ground_truth, predicted_segment).sum()
if union>0:
        iou = intersection / union 
else:
     0.0

iou_rounded = round(iou, 4)
print("The IOU score is: ", iou_rounded)



#Plotting the outputs
Titles = ["Initial Segmentation", "Total Energy", "Refined Image"]
Images = [img1, Total_Energy, refined_img]
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(Images[i], 'gray')
    plt.title(Titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()    
