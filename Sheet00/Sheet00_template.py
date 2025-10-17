"""
Exercise 0 for MA-INF 2201 Computer Vision WS25/26
Introduction to OpenCV - Template
Python 3.12, OpenCV 4.11, NumPy 2.3.3
Image: bonn.jpeg
"""

import cv2
import numpy as np
import random
import time

# ============================================================================
# Exercise 1: Read and Display Image (0.5 Points)
# ============================================================================
def exercise1():
    """
    Read and display the image bonn.jpeg.
    Print the image dimensions and data type.
    """
    print("Exercise 1: Read and Display Image")
    
    # TODO: Read the image 'bonn.jpeg' using cv2.imread()
    img = None
    img = cv2.imread(r"./bonn.jpeg")
    
    # TODO: Check if image was loaded successfully
    if img is None:
        print("Error: Image could not be loaded properly")
        return -1
    else:    
        print("Image loaded successfully")
    
    # TODO: Display the image using cv2.imshow()
    cv2.imshow("Bonn Image", img)

    # TODO: Wait for a key press using cv2.waitKey(0)
    cv2.waitKey(0)
    
    # TODO: Close all windows using cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    
    # TODO: Print image dimensions (height, width, channels)
    img_height, img_width, img_channels = img.shape

    print("Image height: ", img_height)
    print("Image width: ", img_width)
    print("Image Channels: ", img_channels)

    # TODO: Print image data type
    img_dtype = img.dtype
    print("Image data type: ", img_dtype)
    
    print("Exercise 1 completed!\n")
    return img


# ============================================================================
# Exercise 2: HSV Color Space (0.5 Points)
# ============================================================================
def exercise2(img):
    """
    Convert image to HSV color space and display all three channels separately.
    """
    print("Exercise 2: HSV Color Space")
    
    # TODO: Convert to HSV using cv2.cvtColor() with cv2.COLOR_BGR2HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # TODO: Split HSV into H, S, V channels using cv2.split()
    h, s, v = cv2.split()
    
    # TODO: Display all three channels
    # Hint: You can concatenate them horizontally using cv2.hconcat()
    
    print("Exercise 2 completed!\n")
    return hsv


# ============================================================================
# Exercise 3: Brightness Adjustment with Loops (1 Point)
# ============================================================================
def exercise3(img):
    """
    Add 50 to all pixel values and clip to [0, 255] using nested for-loops.
    Display original and brightened images side by side.
    """
    print("Exercise 3: Brightness Adjustment with Loops")
    
    # TODO: Create a copy of the image
    result = None
    
    # TODO: Get image dimensions
    
    # TODO: Use nested for-loops to iterate through each pixel, add 50 to pixel value, and clip pixel value to [0, 255]
    
    # TODO: Display original and result side by side
    
    print("Exercise 3 completed!\n")
    return result


# ============================================================================
# Exercise 4: Vectorized Brightness Adjustment (1 Points)
# ============================================================================
def exercise4(img):
    """
    Perform the same brightness adjustment using NumPy in one line.
    Compare execution time with loop-based approach.
    """
    print("Exercise 4: Vectorized Brightness Adjustment")
    
    # TODO: Time the loop-based approach (from exercise 3)
    start_time_loop = time.time()
    # ... (implement or copy loop code)
    end_time_loop = time.time()
    
    # TODO: Time the vectorized approach
    start_time_vec = time.time()
    # TODO: Add 50 and clip in one line using np.clip()
    result = None
    end_time_vec = time.time()
    
    # TODO: Print execution times
    print(f"Loop-based approach: {end_time_loop - start_time_loop:.4f} seconds")
    print(f"Vectorized approach: {end_time_vec - start_time_vec:.4f} seconds")
    
    # TODO: Display the result
    
    print("Exercise 4 completed!\n")
    return result


# ============================================================================
# Exercise 5: Extract and Paste Patch (0.5 Points)
# ============================================================================
def exercise5(img):
    """
    Extract a 32×32 patch from top-left corner and paste at 3 random locations.
    """
    print("Exercise 5: Extract and Paste Patch")
    
    # TODO: Extract 32x32 patch from top-left corner (starting at 0,0)
    patch_size = 32
    patch = None
    
    # TODO: Create a copy of the image
    img_copy = None
    
    # TODO: Get image dimensions
    
    # TODO: Generate 3 random locations and paste the patch
    # Use random.randint() and ensure patch fits within boundaries
    for i in range(3):
        pass  # TODO: Generate random coordinates and paste
    
    # TODO: Display the result
    
    print("Exercise 5 completed!\n")


# ============================================================================
# Exercise 6: Binary Masking (0.5 Points)
# ============================================================================
def exercise6(img):
    """
    Create masked version showing only bright regions.
    Convert to grayscale, threshold at 128, use as mask.
    """
    print("Exercise 6: Binary Masking")
    
    # TODO: Convert to grayscale using cv2.cvtColor() with cv2.COLOR_BGR2GRAY
    gray = None
    
    # TODO: Apply binary threshold at value 128
    # Use cv2.threshold() with cv2.THRESH_BINARY
    _, mask = None, None
    
    # TODO: Apply mask to original color image
    # Hint: Use cv2.bitwise_and() with the mask
    masked = None
    
    # TODO: Display original, mask, and masked result
    
    print("Exercise 6 completed!\n")


# ============================================================================
# Exercise 7: Border and Annotations (1 Points)
# ============================================================================
def exercise7(img):
    """
    Add 20-pixel border and draw 5 circles and 5 text labels at random positions.
    """
    print("Exercise 7: Border and Annotations")
    
    # TODO: Add 20-pixel border using cv2.copyMakeBorder()
    # Use cv2.BORDER_CONSTANT with a color of your choice
    bordered = None
    
    # TODO: Get dimensions of bordered image
    
    # TODO: Draw 5 random circles
    # Use random.randint() and cv2.circle(img, center, radius, color, thickness)
    for i in range(5):
        pass  # TODO: Implement circle drawing
    
    # TODO: Add 5 random text labels
    # Use random.randint() and cv2.putText(img, text, org, font, fontScale, color, thickness)
    for i in range(5):
        pass  # TODO: Implement text drawing
    
    # TODO: Display the result
    
    print("Exercise 7 completed!\n")


# ============================================================================
# Main function
# ============================================================================
def main():
    """
    Run all exercises.
    """
    print("=" * 60)
    print("Exercise 0: Introduction to OpenCV")
    print("=" * 60 + "\n")
    
    # Uncomment the exercises you want to run:
    img = exercise1()
    # if img is None:
    #     return
    # exercise2(img)
    # exercise3(img)
    # exercise4(img)
    # exercise5(img)
    # exercise6(img)
    # exercise7(img)
    
    print("=" * 60)
    print("All exercises completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
