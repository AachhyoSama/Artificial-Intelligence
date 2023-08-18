import cv2
import numpy as np
from matplotlib import pyplot as plt

# We are going through the process of pattern recognition of a digital image.

# Load the noisy image
image = cv2.imread("images/original_image.jpeg", cv2.IMREAD_GRAYSCALE)

# Step1: Let's apply Gaussian Blur or also called noise reduction by blurring
# Specify the kernel size (odd number for better results)
kernel_size = 5
# Apply Gaussian blur
blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Save the blurred image
cv2.imwrite("images/blurred_image.png", blurred_image)

# Step 2: Prepare histogram from the blurred image
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Blurred Image")
plt.imshow(blurred_image, cmap="gray")

plt.tight_layout()
plt.show()


# Step 3: Let's apply thresholding for image segmentation (Otsu Thresholding)
_, thresholded_image = cv2.threshold(
    blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Display the thresholded image
plt.figure(figsize=(7, 5))
plt.title("Otsu Thresholded Image")
plt.imshow(thresholded_image, cmap="gray")
plt.tight_layout()
plt.show()

# Save the threshold image
cv2.imwrite("images/threshold.png", thresholded_image)


# Step 4: Connectivity analysis to find contours in an image
# Find connected components in the thresholded image
_, labels, stats, _ = cv2.connectedComponentsWithStats(thresholded_image)

# Generate random colors for each object (excluding background)
object_colors = np.random.randint(0, 256, size=(len(stats), 3), dtype=np.uint8)

# Create a copy of the original image for drawing bounding boxes
image_with_colored_boxes = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Loop through the connected components
for label in range(1, len(stats)):
    left = stats[label, cv2.CC_STAT_LEFT]
    top = stats[label, cv2.CC_STAT_TOP]
    width = stats[label, cv2.CC_STAT_WIDTH]
    height = stats[label, cv2.CC_STAT_HEIGHT]

    # Get the color for this object
    color = tuple(map(int, object_colors[label]))

    # Draw bounding boxes around the objects with the assigned color
    cv2.rectangle(
        image_with_colored_boxes, (left, top), (left + width, top + height), color, 2
    )

# Display the image with colored bounding boxes
plt.figure(figsize=(7, 5))
plt.title("Objects with Colored Bounding Boxes")
plt.imshow(cv2.cvtColor(image_with_colored_boxes, cv2.COLOR_BGR2RGB))
plt.tight_layout()
plt.show()

# Save the image after analysis
cv2.imwrite("images/image_after_analysis.png", image_with_colored_boxes)


# Step 5: Pattern Recognition
# Create dynamic arrays for Area and Perimeter
num_objects = len(stats)
Area = np.zeros(num_objects)
Perimeter = np.zeros(num_objects)

# Scan the image to calculate Area and Perimeter
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        label = labels[y, x]
        if label > 0:
            Area[label] += 1
            if (
                x == 0
                or x == image.shape[1] - 1
                or y == 0
                or y == image.shape[0] - 1
                or labels[y - 1, x] != label
                or labels[y + 1, x] != label
                or labels[y, x - 1] != label
                or labels[y, x + 1] != label
            ):
                Perimeter[label] += 1

# Adjust Area
for i in range(1, num_objects):
    Area[i] -= 1

# Create a copy of the original image for drawing recognized shapes
image_with_recognized_shapes = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Recognize shapes
for i in range(1, num_objects):
    ratio = 16.0 * Area[i] / (Perimeter[i] * Perimeter[i])
    left = stats[i, cv2.CC_STAT_LEFT]
    top = stats[i, cv2.CC_STAT_TOP]
    width = stats[i, cv2.CC_STAT_WIDTH]
    height = stats[i, cv2.CC_STAT_HEIGHT]
    color = tuple(map(int, object_colors[i]))

    if ratio < 1.1:
        shape = "Square"
    else:
        shape = "Circle"

    # Draw bounding boxes around the recognized shapes
    cv2.rectangle(
        image_with_recognized_shapes,
        (left, top),
        (left + width, top + height),
        color,
        2,
    )
    cv2.putText(
        image_with_recognized_shapes,
        shape,
        (left, top - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
    )

# Display the image with recognized shapes
plt.figure(figsize=(7, 5))
plt.title("Objects with Recognized Shapes")
plt.imshow(image_with_recognized_shapes)
plt.tight_layout()
plt.show()

# Save the image with recognized shapes
cv2.imwrite("images/recognized_image.png", image_with_recognized_shapes)

# Free Area and Perimeter arrays
del Area
del Perimeter
