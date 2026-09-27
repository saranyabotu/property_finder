import os

train_images = "dataset/train/images"
train_labels = "dataset/train/labels"
val_images = "dataset/val/images"
val_labels = "dataset/val/labels"

print("===== YOLO DATASET CHECK =====")

print("\nTraining images:")
for file in os.listdir(train_images):
    print(file)

print("\nTraining labels:")
for file in os.listdir(train_labels):
    print(file)

print("\nValidation images:")
for file in os.listdir(val_images):
    print(file)

print("\nValidation labels:")
for file in os.listdir(val_labels):
    print(file)

print("\nDataset structure check completed.")