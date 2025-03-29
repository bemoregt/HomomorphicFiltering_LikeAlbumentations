import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from homomorphic_filter import HomomorphicFilter

# Load an image
image_path = 'input_image.jpg'  # Replace with your image path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Create a transform pipeline with HomomorphicFilter
transform = A.Compose([
    HomomorphicFilter(p=1.0),
    # Add other transformations as needed
    # A.RandomRotate90(),
])

# Apply the transformation
transformed = transform(image=image)
transformed_image = transformed["image"]

# Display the original and transformed images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Homomorphic Filtered Image')
plt.imshow(transformed_image)
plt.axis('off')

plt.tight_layout()
plt.show()

# Save the transformed image
cv2.imwrite('output_image.jpg', cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
