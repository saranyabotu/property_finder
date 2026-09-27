import os
import pandas as pd
from PIL import Image
from ultralytics import YOLO

from ocr_reader import read_text
from property_details import extract_property_details
from property_classifier import classify_property


# =========================================================
# 1. PATHS
# =========================================================

IMAGE_FOLDER = "dataset/images"
METADATA_FILE = "dataset/metadata.csv"
DATABASE_FILE = "dataset/property_database.csv"
CROP_FOLDER = "dataset/crops"

MODEL_PATH = "runs/detect/property_ad_detector-4/weights/best.pt"

CONF_THRESHOLD = 0.10
MAX_AGE_DAYS = 30


# =========================================================
# 2. CREATE REQUIRED FOLDER
# =========================================================

os.makedirs(CROP_FOLDER, exist_ok=True)


# =========================================================
# 3. LOAD YOLO MODEL
# =========================================================

print("\nLoading property advertisement detector...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully.")


# =========================================================
# 4. LOAD METADATA
# =========================================================

metadata = pd.read_csv(METADATA_FILE)

metadata.columns = metadata.columns.str.strip()


# =========================================================
# 5. PROCESS ONLY IMAGES HAVING METADATA
# =========================================================

image_files = (
    metadata["image_name"]
    .astype(str)
    .str.strip()
    .tolist()
)

if not image_files:

    print("No images with metadata found.")
    exit()


# =========================================================
# 6. DATABASE COLUMNS
# =========================================================

database_columns = [
    "image_name",
    "property_type",
    "price",
    "area",
    "bhk",
    "location",
    "capture_date",
    "latitude",
    "longitude"
]


# =========================================================
# 7. LOAD / MIGRATE DATABASE
# =========================================================

if os.path.exists(DATABASE_FILE):

    database = pd.read_csv(DATABASE_FILE)

    database.columns = database.columns.str.strip()

    # Add BHK column if old database doesn't contain it
    if "bhk" not in database.columns:

        database.insert(
            database.columns.get_loc("area") + 1,
            "bhk",
            "Not detected"
        )

    # Make sure all required columns exist
    for column in database_columns:

        if column not in database.columns:

            database[column] = "Not detected"

    # Keep columns in correct order
    database = database[database_columns]

    # Save migrated database
    database.to_csv(
        DATABASE_FILE,
        index=False
    )

else:

    database = pd.DataFrame(
        columns=database_columns
    )

    database.to_csv(
        DATABASE_FILE,
        index=False
    )


# =========================================================
# 8. PROCESS EACH IMAGE
# =========================================================

for image_name in sorted(image_files):

    print("\n==========================================")
    print("Processing:", image_name)
    print("==========================================")


    # =====================================================
    # METADATA
    # =====================================================

    image_data = metadata[
        metadata["image_name"]
        .astype(str)
        .str.strip()
        == image_name
    ]

    if image_data.empty:

        print("Metadata not found.")
        print("Skipping image.")

        continue


    row = image_data.iloc[0]


    # =====================================================
    # CAPTURE DATE
    # =====================================================

    capture_date = pd.to_datetime(
        row["capture_date"],
        errors="coerce"
    )

    if pd.isna(capture_date):

        print("Invalid capture date.")
        print("Skipping image.")

        continue


    today = pd.Timestamp.today().normalize()

    age_days = (
        today - capture_date.normalize()
    ).days


    print("Capture Date:", capture_date.date())
    print("Image Age:", age_days, "days")


    # =====================================================
    # RECENCY CHECK
    # =====================================================

    if age_days < 0:

        print("Capture date is in the future.")
        print("Skipping image.")

        continue


    if age_days > MAX_AGE_DAYS:

        print(
            "Image is older than",
            MAX_AGE_DAYS,
            "days."
        )

        print("Skipping image.")

        continue


    print("Image is recent. Processing...")


    # =====================================================
    # IMAGE PATH
    # =====================================================

    image_path = os.path.join(
        IMAGE_FOLDER,
        image_name
    )

    if not os.path.exists(image_path):

        print("Image file not found.")
        print("Skipping image.")

        continue


    # =====================================================
    # YOLO DETECTION
    # =====================================================

    results = model.predict(
        source=image_path,
        conf=CONF_THRESHOLD,
        max_det=10,
        verbose=False
    )


    best_box = None
    best_confidence = 0.0


    for result in results:

        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )

            if confidence > best_confidence:

                best_confidence = confidence
                best_box = box


    if best_box is None:

        print(
            "No property advertisement detected by YOLO."
        )

        continue


    print(
        "Property advertisement detected!"
    )

    print(
        "YOLO Confidence:",
        round(best_confidence, 4)
    )


    # =====================================================
    # GET BOUNDING BOX
    # =====================================================

    x1, y1, x2, y2 = (
        best_box.xyxy[0].tolist()
    )

    x1, y1, x2, y2 = map(
        int,
        [x1, y1, x2, y2]
    )


    print(
        f"Bounding Box: "
        f"{x1}, {y1}, {x2}, {y2}"
    )


    # =====================================================
    # CROP ADVERTISEMENT
    # =====================================================

    original_image = Image.open(
        image_path
    )

    crop = original_image.crop(
        (x1, y1, x2, y2)
    )


    base_name = os.path.splitext(
        image_name
    )[0]


    crop_path = os.path.join(
        CROP_FOLDER,
        f"{base_name}_ad.png"
    )


    crop.save(crop_path)


    print(
        "Advertisement crop saved:",
        crop_path
    )


    # =====================================================
    # OCR
    # =====================================================

    ocr_output_path = os.path.join(
        "backend",
        f"{base_name}_ocr.txt"
    )


    extracted_text = read_text(
        crop_path,
        ocr_output_path
    )


    print("\nOCR TEXT")
    print("------------------------------------------")
    print(extracted_text)
    print("------------------------------------------")


    # =====================================================
    # PROPERTY DETAILS
    # =====================================================

    details = extract_property_details(
        extracted_text
    )


    # =====================================================
    # PROPERTY TYPE
    # =====================================================

    property_type = classify_property(
        extracted_text
    )


    # =====================================================
    # VERIFIED GPS
    # =====================================================

    latitude = row["latitude"]
    longitude = row["longitude"]


    # =====================================================
    # COMPLETE PROPERTY RECORD
    # =====================================================

    print("\nPROPERTY RECORD")
    print("------------------------------------------")

    print(
        "Image Name    :",
        image_name
    )

    print(
        "Property Type :",
        property_type
    )

    print(
        "Price         :",
        details["price"]
    )

    print(
        "Area          :",
        details["area"]
    )

    print(
        "BHK           :",
        details["bhk"]
    )

    print(
        "Location      :",
        details["location"]
    )

    print(
        "Capture Date  :",
        capture_date.date()
    )

    print(
        "Latitude      :",
        latitude
    )

    print(
        "Longitude     :",
        longitude
    )

    print("------------------------------------------")


    # =====================================================
    # DUPLICATE CHECK
    # =====================================================

    database["image_name"] = (
        database["image_name"]
        .astype(str)
        .str.strip()
    )


    if image_name in database["image_name"].values:

        print(
            "Property already exists in database."
        )

        print(
            "Skipping duplicate record."
        )

        continue


    # =====================================================
    # CREATE NEW DATABASE RECORD
    # =====================================================

    new_record = {
        "image_name": image_name,
        "property_type": property_type,
        "price": details["price"],
        "area": details["area"],
        "bhk": details["bhk"],
        "location": details["location"],
        "capture_date": capture_date.strftime("%Y-%m-%d"),
        "latitude": latitude,
        "longitude": longitude
    }


    new_row = pd.DataFrame(
        [new_record],
        columns=database_columns
    )


    database = pd.concat(
        [database, new_row],
        ignore_index=True
    )


    database.to_csv(
        DATABASE_FILE,
        index=False
    )


    print(
        "Property saved to database successfully!"
    )


# =========================================================
# COMPLETION
# =========================================================

print("\n==========================================")
print("PROPERTY PROCESSING COMPLETED")
print("==========================================")