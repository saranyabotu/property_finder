from ultralytics import YOLO
from PIL import Image
from ocr_reader import read_text
import os

# Load our trained property advertisement detector
model = YOLO(
    "runs/detect/property_ad_detector-4/weights/best.pt"
)

# Image to analyze
image_path = "dataset/images/image_002.png"

# Minimum confidence for accepting a detection
CONF_THRESHOLD = 0.10

# Run YOLO detection
results = model.predict(
    source=image_path,
    conf=CONF_THRESHOLD,
    max_det=10
)

# Open original image
image = Image.open(image_path)

best_box = None
best_confidence = 0.0

for result in results:
    for box in result.boxes:
        confidence = float(box.conf[0])

        if confidence > best_confidence:
            best_confidence = confidence
            best_box = box

# Check whether an advertisement was detected
if best_box is None:
    print("No property advertisement detected.")
else:
    print("Property advertisement detected!")
    print(f"Confidence: {best_confidence:.4f}")

    # Get bounding box coordinates
    x1, y1, x2, y2 = best_box.xyxy[0].tolist()

    # Convert to integers
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    print(f"Bounding box: {x1}, {y1}, {x2}, {y2}")

    # Crop the detected advertisement
    cropped_image = image.crop((x1, y1, x2, y2))

    # Create output folder
    os.makedirs("dataset/crops", exist_ok=True)

    crop_path = "dataset/crops/detected_ad.png"

    # Save crop
    cropped_image.save(crop_path)

    print(f"Advertisement crop saved: {crop_path}")

    # Send crop to OCR
    ocr_output_path = "backend/ocr_detected_ad.txt"

    extracted_text = read_text(
        crop_path,
        ocr_output_path
    )

    print("\n===== OCR RESULT =====")
    print(extracted_text)
    print("======================")