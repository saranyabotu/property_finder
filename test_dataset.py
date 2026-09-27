import cv2

image_path = "dataset/images/image_001.png"

image = cv2.imread(image_path)

if image is not None:
    print("Image loaded successfully!")
    print("Image size:", image.shape)
else:
    print("Could not load the image.")