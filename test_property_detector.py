from ultralytics import YOLO

# Load the latest trained model
model = YOLO(
    "runs/detect/property_ad_detector-4/weights/best.pt"
)

# Image to test
image_path = "dataset/images/image_014.png"

# Run detection
results = model.predict(
    source=image_path,
    save=True,
    conf=0.001,
    max_det=10
)

# Display results
for result in results:
    print("Detected objects:", len(result.boxes))

    if len(result.boxes) > 0:
        print("Detection details:")

        for i, box in enumerate(result.boxes):
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            print(
                f"  Detection {i + 1}: "
                f"class={class_id}, "
                f"confidence={confidence:.4f}"
            )

print("Detection test completed!")