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
    h, s, v = cv2.split(img)
    
    # TODO: Display all three channels
    # Hint: You can concatenate them horizontally using cv2.hconcat()
    
    cv2.imshow("HSV Channels", cv2.hconcat([h, s, v]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
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
    
    result = img.copy()  # TODO: Create a copy of the image
    
    height, width, channels = img.shape  # TODO: Get image dimensions
    
    # TODO: Use nested for-loops to iterate through each pixel, add 50 to pixel value, and clip pixel value to [0, 255]
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                new_value = img[y, x, c] + 50
                result[y, x, c] = np.clip(new_value, 0, 255)
    
    combined = np.hstack((img, result))  # TODO: Display original and result side by side
    cv2.imshow("Original (Left) | Brightened (Right)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
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

    # NOTE: First convert to int16 to avoid overflow, then back to uint8 
    #       after clipping, or else overflow causes img to change color 
    result = np.clip(img.astype(np.int16) + 50, 0, 255).astype(np.uint8)
    end_time_vec = time.time()
    
    # TODO: Print execution times
    print(f"Loop-based approach: {end_time_loop - start_time_loop:.4f} seconds")
    print(f"Vectorized approach: {end_time_vec - start_time_vec:.4f} seconds")
    
    # TODO: Display the result
    display_imgs = cv2.hconcat([img, result])

    cv2.imshow("Original vs Brightness Adjusted", display_imgs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    
    patch_size = 32
    patch = img[0:patch_size, 0:patch_size].copy()  # TODO: Extract 32x32 patch from top-left corner (starting at 0,0)
    
    img_copy = img.copy()  # TODO: Create a copy of the image
    
    height, width, channels = img.shape  # TODO: Get image dimensions
    
    # TODO: Generate 3 random locations and paste the patch
    # Use random.randint() and ensure patch fits within boundaries
    for i in range(3):
        x = random.randint(0, width - patch_size)
        y = random.randint(0, height - patch_size)
        img_copy[y:y+patch_size, x:x+patch_size] = patch
    
    cv2.imshow("Patched Image", img_copy)  # TODO: Display the result
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Exercise 5 completed!\n")
    return img_copy


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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # TODO: Apply binary threshold at value 128
    # Use cv2.threshold() with cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    
    # TODO: Apply mask to original color image
    # Hint: Use cv2.bitwise_and() with the mask
    masked = cv2.bitwise_and(img, img, mask=mask)

    # TODO: Display original, mask, and masked result
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convert mask to 3-channel for displaying together

    # print(img.dtype, mask.dtype, masked.dtype)  .... just to debug
    
    display_imgs = cv2.hconcat([img, mask, masked])

    cv2.imshow("Original vs Mask vs Masked Result", display_imgs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Exercise 6 completed!\n")


# ============================================================================
# Exercise 7: Border and Annotations (1 Points)
# ============================================================================
def exercise7(img):
    """
    Add 20-pixel border and draw 5 circles and 5 text labels at random positions.
    """
    print("Exercise 7: Border and Annotations")
    
    bordered = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(0, 0, 255))  # TODO: Add 20-pixel border using cv2.copyMakeBorder()
    # Use cv2.BORDER_CONSTANT with a color of your choice
    
    height, width, channels = bordered.shape  # TODO: Get dimensions of bordered image
    
    # TODO: Draw 5 random circles
    # Use random.randint() and cv2.circle(img, center, radius, color, thickness)
    for i in range(5):
        center_x = random.randint(20, width - 20)
        center_y = random.randint(20, height - 20)
        radius = random.randint(10, 40)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.circle(bordered, (center_x, center_y), radius, color, -1)
    
    # TODO: Add 5 random text labels
    # Use random.randint() and cv2.putText(img, text, org, font, fontScale, color, thickness)
    for i in range(5):
        x = random.randint(30, width - 100)
        y = random.randint(30, height - 30)
        text = f"Text{i+1}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.putText(bordered, text, (x, y), font, font_scale, color, 2, cv2.LINE_AA)
    
    cv2.imshow("Border and Annotations", bordered)  # TODO: Display the result
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Exercise 7 completed!\n")
    return bordered


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
    if img is None:
        return
    exercise2(img)
    exercise3(img)
    exercise4(img)
    exercise5(img)
    exercise6(img)
    exercise7(img)
    
    print("=" * 60)
    print("All exercises completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
