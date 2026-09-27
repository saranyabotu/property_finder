from pathlib import Path
import pandas as pd
from PIL import Image
from ultralytics import YOLO

from ocr_reader import read_text
from property_details import extract_property_details
from property_classifier import classify_property


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_FOLDER = BASE_DIR / "dataset" / "images"
CROP_FOLDER = BASE_DIR / "dataset" / "crops"

BATCH_RESULTS_FILE = (
    BASE_DIR
    / "dataset"
    / "batch_results.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "detect"
    / "property_ad_detector-4"
    / "weights"
    / "best.pt"
)

CONF_THRESHOLD = 0.10


# =========================================================
# CREATE CROP FOLDER
# =========================================================

CROP_FOLDER.mkdir(exist_ok=True)


# =========================================================
# LOAD YOLO MODEL
# =========================================================

print()
print("==============================================")
print(" PROPERTY DISCOVERY SYSTEM - BATCH PROCESSOR")
print("==============================================")
print()

print("Loading YOLO model...")

model = YOLO(str(MODEL_PATH))

print("YOLO model loaded successfully.")
print()


# =========================================================
# PROCESS ONE IMAGE
# =========================================================

def process_image(image_path):

    print("----------------------------------------------")
    print(f"Processing: {image_path.name}")

    record = {

        "image_name":
            image_path.name,

        "detection_status":
            "Not detected",

        "confidence":
            "",

        "property_type":
            "Not detected",

        "price":
            "Not detected",

        "area":
            "Not detected",

        "bhk":
            "Not detected",

        "location":
            "Not detected",

        "ocr_text":
            "",

        "capture_date":
            "",

        "latitude":
            "",

        "longitude":
            ""
    }


    # =====================================================
    # YOLO
    # =====================================================

    try:

        detection_results = model.predict(
            source=str(image_path),
            conf=CONF_THRESHOLD,
            max_det=10,
            verbose=False
        )

    except Exception as error:

        print(
            f"YOLO error: {error}"
        )

        record["detection_status"] = (
            "Processing error"
        )

        return record


    # =====================================================
    # FIND BEST DETECTION
    # =====================================================

    best_box = None
    best_confidence = 0.0


    for result in detection_results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )


            if confidence > best_confidence:

                best_confidence = confidence
                best_box = box


    # =====================================================
    # NO DETECTION
    # =====================================================

    if best_box is None:

        print(
            "No property advertisement detected."
        )

        return record


    record["detection_status"] = (
        "Advertisement detected"
    )

    record["confidence"] = round(
        best_confidence,
        4
    )


    print(
        f"Advertisement detected "
        f"(confidence: {best_confidence:.4f})"
    )


    # =====================================================
    # OPEN IMAGE
    # =====================================================

    try:

        original_image = Image.open(
            image_path
        )

    except Exception as error:

        print(
            f"Image error: {error}"
        )

        record["detection_status"] = (
            "Image processing error"
        )

        return record


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


    # =====================================================
    # CROP ADVERTISEMENT
    # =====================================================

    crop_filename = (
        f"{image_path.stem}_ad.png"
    )

    crop_path = (
        CROP_FOLDER
        / crop_filename
    )


    try:

        crop = original_image.crop(
            (x1, y1, x2, y2)
        )

        crop.save(
            crop_path
        )

        print(
            f"Advertisement crop saved: "
            f"{crop_filename}"
        )

    except Exception as error:

        print(
            f"Crop error: {error}"
        )

        record["detection_status"] = (
            "Crop error"
        )

        return record


    # =====================================================
    # OCR
    # =====================================================

    try:

        ocr_text = read_text(
            str(crop_path)
        )

    except Exception as error:

        print(
            f"OCR error: {error}"
        )

        ocr_text = ""


    record["ocr_text"] = (
        str(ocr_text).strip()
    )


    print("OCR completed.")


    # =====================================================
    # PROPERTY DETAILS
    # =====================================================

    try:

        details = extract_property_details(
            ocr_text
        )

    except Exception as error:

        print(
            f"Property-detail extraction error: "
            f"{error}"
        )

        details = {}


    # =====================================================
    # PROPERTY TYPE
    # =====================================================

    try:

        property_type = classify_property(
            ocr_text
        )

    except Exception as error:

        print(
            f"Classification error: {error}"
        )

        property_type = (
            "Not detected"
        )


    # =====================================================
    # STORE DETECTED DETAILS
    # =====================================================

    record["property_type"] = (
        details.get(
            "property_type",
            property_type
        )
        if isinstance(details, dict)
        else property_type
    )


    if (
        not record["property_type"]
        or str(record["property_type"]).lower()
        in ["none", "nan"]
    ):

        record["property_type"] = (
            property_type
        )


    record["price"] = (
        details.get(
            "price",
            "Not detected"
        )
        if isinstance(details, dict)
        else "Not detected"
    )


    record["area"] = (
        details.get(
            "area",
            "Not detected"
        )
        if isinstance(details, dict)
        else "Not detected"
    )


    record["bhk"] = (
        details.get(
            "bhk",
            "Not detected"
        )
        if isinstance(details, dict)
        else "Not detected"
    )


    record["location"] = (
        details.get(
            "location",
            "Not detected"
        )
        if isinstance(details, dict)
        else "Not detected"
    )


    print(
        f"Property type: "
        f"{record['property_type']}"
    )

    print(
        f"Price: "
        f"{record['price']}"
    )

    print(
        f"Area: "
        f"{record['area']}"
    )

    print(
        f"BHK: "
        f"{record['bhk']}"
    )

    print(
        f"Location: "
        f"{record['location']}"
    )


    return record


# =========================================================
# MAIN BATCH PROCESS
# =========================================================

def main():

    image_files = sorted(
        [
            file
            for file in IMAGE_FOLDER.iterdir()
            if file.is_file()
            and file.suffix.lower()
            in [".png", ".jpg", ".jpeg"]
        ]
    )


    if not image_files:

        print(
            "No images found in dataset/images."
        )

        return


    print(
        f"Found {len(image_files)} images."
    )

    print()


    records = []


    # =====================================================
    # PROCESS ALL IMAGES
    # =====================================================

    for number, image_path in enumerate(
        image_files,
        start=1
    ):

        print(
            f"[{number}/{len(image_files)}]"
        )

        record = process_image(
            image_path
        )

        records.append(
            record
        )

        print()


    # =====================================================
    # CREATE DATAFRAME
    # =====================================================

    results = pd.DataFrame(
        records
    )


    # =====================================================
    # SAVE CSV
    # =====================================================

    results.to_csv(
        BATCH_RESULTS_FILE,
        index=False
    )


    # =====================================================
    # SUMMARY
    # =====================================================

    detected_count = (
        results[
            results["detection_status"]
            == "Advertisement detected"
        ].shape[0]
    )

    not_detected_count = (
        results[
            results["detection_status"]
            == "Not detected"
        ].shape[0]
    )


    print()
    print("==============================================")
    print(" BATCH PROCESSING COMPLETED")
    print("==============================================")
    print()

    print(
        f"Total images      : "
        f"{len(results)}"
    )

    print(
        f"Advertisements    : "
        f"{detected_count}"
    )

    print(
        f"Not detected      : "
        f"{not_detected_count}"
    )

    print()

    print(
        f"Results saved to:"
    )

    print(
        BATCH_RESULTS_FILE
    )

    print()


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()