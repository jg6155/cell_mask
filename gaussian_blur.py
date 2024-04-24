import cv2
from matplotlib import pyplot as plt

def apply_gaussian_blur(image_path, kernel_size=(5, 5), sigma_x=0):
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image was successfully loaded
    if image is None:
        print("Error: Image not found.")
        return None

    # Convert image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply Gaussian Blur
    blurred_image = cv2.GaussianBlur(image_rgb, kernel_size, sigma_x)
    
    # Display the original and blurred images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(blurred_image)
    plt.title('Gaussian Blurred Image')
    plt.axis('off')

    plt.show()

    return blurred_image

# Usage
image_path = 'test_img.tif'  # Replace with your image path
blurred_image = apply_gaussian_blur(image_path, kernel_size=(15, 15), sigma_x=10)

plt.imsave('super_blur.tif',blurred_image, format='TIFF')
