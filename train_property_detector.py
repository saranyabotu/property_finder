from ultralytics import YOLO

print("Starting Property Advertisement Detector training...")

model = YOLO("yolo11n.pt")

model.train(
    data="dataset/data.yaml",
    epochs=30,
    imgsz=640,
    batch=2,
    name="property_ad_detector"
)

print("Training completed!")