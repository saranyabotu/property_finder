import os
import shutil

source_images = "dataset/images"
source_labels = "dataset/labels"

train_images = "dataset/train/images"
train_labels = "dataset/train/labels"

val_images = "dataset/val/images"
val_labels = "dataset/val/labels"

os.makedirs(train_images, exist_ok=True)
os.makedirs(train_labels, exist_ok=True)
os.makedirs(val_images, exist_ok=True)
os.makedirs(val_labels, exist_ok=True)

# Images 001–012 → training
for i in range(1, 13):
    image_name = f"image_{i:03d}.png"
    label_name = f"image_{i:03d}.txt"

    shutil.copy2(
        os.path.join(source_images, image_name),
        os.path.join(train_images, image_name)
    )

    shutil.copy2(
        os.path.join(source_labels, label_name),
        os.path.join(train_labels, label_name)
    )

# Images 013–014 → validation
for i in range(13, 15):
    image_name = f"image_{i:03d}.png"
    label_name = f"image_{i:03d}.txt"

    shutil.copy2(
        os.path.join(source_images, image_name),
        os.path.join(val_images, image_name)
    )

    shutil.copy2(
        os.path.join(source_labels, label_name),
        os.path.join(val_labels, label_name)
    )

print("YOLO dataset prepared successfully!")
print("Training images: 12")
print("Validation images: 2")