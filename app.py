from pathlib import Path
from datetime import datetime
import hashlib
import re

import pandas as pd
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO

from property_classifier import classify_property
from property_details import extract_property_details


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
IMAGE_FOLDER = DATASET_DIR / "images"
UPLOAD_FOLDER = BASE_DIR / "uploads"

METADATA_FILE = DATASET_DIR / "metadata.csv"
DATABASE_FILE = DATASET_DIR / "property_database.csv"

MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "detect"
    / "property_ad_detector-4"
    / "weights"
    / "best.pt"
)

MAX_AGE_DAYS = 30

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp"
}


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

model = YOLO(str(MODEL_PATH))


# ============================================================
# OCR
# ============================================================

def read_text(image_path):
    """
    Reads text from the uploaded property advertisement
    using EasyOCR.
    """

    try:
        import easyocr

        reader = easyocr.Reader(
            ["en"],
            gpu=False,
            verbose=False
        )

        results = reader.readtext(str(image_path))

        text_parts = []

        for item in results:
            if len(item) >= 2:
                text_parts.append(str(item[1]))

        return "\n".join(text_parts)

    except Exception as e:
        print("OCR Error:", e)
        return ""


# ============================================================
# FILE VALIDATION
# ============================================================

def allowed_file(filename):
    """
    Checks whether the uploaded file has a supported
    image extension.
    """

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# IMAGE HASH
# ============================================================

def calculate_image_hash(file_path):
    """
    Creates SHA-256 hash for duplicate-image detection.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        for chunk in iter(
            lambda: file.read(8192),
            b""
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# DATABASE
# ============================================================

def load_database():
    """
    Loads property_database.csv.

    Expected columns:
    image_name
    property_type
    price
    area
    bhk
    location
    capture_date
    latitude
    longitude
    """

    columns = [
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

    if not DATABASE_FILE.exists():

        return pd.DataFrame(
            columns=columns
        )

    try:

        df = pd.read_csv(
            DATABASE_FILE
        )

        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
        )

        for column in columns:

            if column not in df.columns:
                df[column] = ""

        return df[columns].copy()

    except Exception as e:

        print(
            "Database loading error:",
            e
        )

        return pd.DataFrame(
            columns=columns
        )


def save_database(df):
    """
    Saves the property database.
    """

    df.to_csv(
        DATABASE_FILE,
        index=False
    )


# ============================================================
# PRICE CONVERSION
# ============================================================

def price_to_number(value):
    """
    Converts prices such as:

    47 LAKH
    75 LAKH
    1.2 CRORE

    into numeric values.
    """

    if pd.isna(value):
        return None

    text = (
        str(value)
        .strip()
        .upper()
    )

    if text in {
        "",
        "NAN",
        "NONE",
        "NOT DETECTED",
        "UNKNOWN"
    }:
        return None

    text = text.replace(
        ",",
        ""
    )

    # LAKHS

    match = re.search(
        r"₹?\s*(\d+(?:\.\d+)?)\s*LAKHS?",
        text
    )

    if match:

        try:

            return (
                float(match.group(1))
                * 100000
            )

        except ValueError:

            return None

    # CRORES

    match = re.search(
        r"₹?\s*(\d+(?:\.\d+)?)\s*CRORES?",
        text
    )

    if match:

        try:

            return (
                float(match.group(1))
                * 10000000
            )

        except ValueError:

            return None

    # Plain number

    match = re.search(
        r"₹?\s*(\d+(?:\.\d+)?)",
        text
    )

    if match:

        try:

            return float(
                match.group(1)
            )

        except ValueError:

            return None

    return None


# ============================================================
# METADATA
# ============================================================

def get_metadata(image_name):
    """
    Gets capture date and GPS from metadata.csv if available.

    IMPORTANT:
    A new image does NOT have to exist in metadata.csv.

    If there is no metadata entry:
    - today's date is used
    - GPS is left empty initially
    - GPS can later be obtained from the detected location
    """

    # --------------------------------------------------------
    # Try metadata.csv first
    # --------------------------------------------------------

    if METADATA_FILE.exists():

        try:

            metadata = pd.read_csv(
                METADATA_FILE
            )

            metadata.columns = (
                metadata.columns
                .astype(str)
                .str.strip()
            )

            if "image_name" in metadata.columns:

                metadata["image_name"] = (
                    metadata["image_name"]
                    .astype(str)
                    .str.strip()
                )

                matches = metadata[
                    metadata["image_name"]
                    .str.lower()
                    ==
                    image_name.lower()
                ]

                if not matches.empty:

                    row = matches.iloc[0]

                    return {
                        "capture_date": row.get(
                            "capture_date",
                            ""
                        ),
                        "latitude": row.get(
                            "latitude",
                            ""
                        ),
                        "longitude": row.get(
                            "longitude",
                            ""
                        )
                    }

        except Exception as e:

            print(
                "Metadata CSV error:",
                e
            )

    # --------------------------------------------------------
    # No metadata found
    # --------------------------------------------------------

    return {
        "capture_date":
            datetime.now().strftime(
                "%Y-%m-%d"
            ),
        "latitude": "",
        "longitude": ""
    }


# ============================================================
# LOCATION → DEMO GPS
# ============================================================

def get_location_coordinates(location):
    """
    Provides prototype/demo coordinates for recognized locations.

    In a real deployment, these coordinates would come from
    the imagery/map source.

    Returns:
        latitude, longitude
    """

    if not location:

        return "", ""

    location_text = (
        str(location)
        .lower()
        .strip()
    )

    location_coordinates = {

        "manikonda":
            (17.4100, 78.4200),

        "alkapoor colony":
            (17.4100, 78.4200),

        "nelamangala":
            (13.1020, 77.3930),

        "puppalaguda":
            (17.4040, 78.3830),

        "vizianagaram":
            (18.1067, 83.3956),

        "karelibaug":
            (22.3290, 73.2080),

        "vijayawada":
            (16.5062, 80.6480),

        "hyderabad":
            (17.3850, 78.4867),

        "mookandapalli":
            (12.7400, 77.8300),

        "rk beach":
            (17.7140, 83.3230)
    }

    for name, coordinates in (
        location_coordinates.items()
    ):

        if name in location_text:

            return coordinates

    return "", ""


# ============================================================
# RECENT IMAGE CHECK
# ============================================================

def check_recent_image(capture_date):
    """
    Checks whether imagery is within MAX_AGE_DAYS.
    """

    parsed_date = pd.to_datetime(
        capture_date,
        errors="coerce"
    )

    if pd.isna(parsed_date):

        return False, None

    today = pd.Timestamp(
        datetime.now().date()
    )

    age_days = (
        today
        -
        parsed_date.normalize()
    ).days

    # Future dates are rejected

    if age_days < 0:

        return False, age_days

    # Too old

    if age_days > MAX_AGE_DAYS:

        return False, age_days

    return True, age_days


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "home.html"
    )


# ============================================================
# DISCOVER
# ============================================================

@app.route("/discover")
def discover():

    return render_template(
        "discover.html"
    )


# ============================================================
# SEARCH PAGE
# ============================================================

@app.route("/search")
def search():

    return render_template(
        "search.html"
    )


# Endpoint aliases for old templates

app.add_url_rule(
    "/search",
    endpoint="search_page",
    view_func=search
)

app.add_url_rule(
    "/search",
    endpoint="search_properties_page",
    view_func=search
)


# ============================================================
# SEARCH RESULTS
# ============================================================

@app.route("/results")
def results():

    location = (
        request.args
        .get(
            "location",
            ""
        )
        .strip()
    )

    property_type = (
        request.args
        .get(
            "property_type",
            ""
        )
        .strip()
        .upper()
    )

    max_price = (
        request.args
        .get(
            "max_price",
            ""
        )
        .strip()
    )

    max_age = (
        request.args
        .get(
            "max_age",
            "30"
        )
        .strip()
    )

    df = load_database()

    print()
    print(
        "========================================"
    )
    print(
        "PROPERTY SEARCH"
    )
    print(
        "========================================"
    )

    print(
        "Location      :",
        location
    )

    print(
        "Property Type :",
        property_type
    )

    print(
        "Maximum Price :",
        max_price
    )

    print(
        "Maximum Age   :",
        max_age
    )

    print(
        "Database Rows :",
        len(df)
    )

    # --------------------------------------------------------
    # LOCATION FILTER
    # --------------------------------------------------------

    if location:

        location_search = (
            location
            .lower()
            .strip()
        )

        df = df[
            df["location"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
            .str.contains(
                location_search,
                regex=False,
                na=False
            )
        ]

    # --------------------------------------------------------
    # PROPERTY TYPE FILTER
    # --------------------------------------------------------

    if property_type:

        df = df[
            df["property_type"]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.strip()
            ==
            property_type
        ]

    # --------------------------------------------------------
    # PRICE FILTER
    # --------------------------------------------------------

    if max_price:

        try:

            maximum_price = float(
                max_price
            )

            numeric_prices = (
                df["price"]
                .apply(price_to_number)
            )

            # IMPORTANT:
            #
            # If price is detected:
            #     compare it with maximum price.
            #
            # If price is NOT detected:
            #     keep the property.
            #
            df = df[
                numeric_prices.isna()
                |
                (
                    numeric_prices
                    <= maximum_price
                )
            ]

        except ValueError:

            print(
                "Invalid maximum price:",
                max_price
            )

    # --------------------------------------------------------
    # RECENCY FILTER
    # --------------------------------------------------------

    if max_age:

        try:

            age_limit = int(
                max_age
            )

            today = pd.Timestamp(
                datetime.now().date()
            )

            dates = pd.to_datetime(
                df["capture_date"],
                errors="coerce"
            )

            age_days = (
                today
                -
                dates
            ).dt.days

            df = df[
                dates.notna()
                &
                (age_days >= 0)
                &
                (age_days <= age_limit)
            ]

        except ValueError:

            print(
                "Invalid maximum age:",
                max_age
            )

    # --------------------------------------------------------
    # SEARCH OUTPUT
    # --------------------------------------------------------

    print(
        "----------------------------------------"
    )

    print(
        "Matching Properties:",
        len(df)
    )

    if not df.empty:

        print(
            df.to_string(
                index=False
            )
        )

    else:

        print(
            "No matching properties found."
        )

    print(
        "========================================"
    )

    print()

    properties = (
        df.to_dict(
            "records"
        )
    )

    return render_template(
        "results.html",
        properties=properties,
        location=location,
        property_type=property_type,
        max_price=max_price,
        max_age=max_age
    )


# Endpoint alias

app.add_url_rule(
    "/results",
    endpoint="results_page",
    view_func=results
)


# ============================================================
# PROPERTY DETAILS
# ============================================================

@app.route(
    "/property/<image_name>"
)
def property_details_page(
    image_name
):

    df = load_database()

    if df.empty:

        return (
            "Property database is empty.",
            404
        )

    image_name = (
        image_name
        .strip()
        .lower()
    )

    matches = df[
        df["image_name"]
        .astype(str)
        .str.strip()
        .str.lower()
        ==
        image_name
    ]

    if matches.empty:

        return (
            "Property not found.",
            404
        )

    property_data = (
        matches
        .iloc[0]
        .to_dict()
    )

    return render_template(
        "property.html",
        property=property_data
    )


# Endpoint alias

app.add_url_rule(
    "/property/<image_name>",
    endpoint="property_details",
    view_func=property_details_page
)


# ============================================================
# RECENT PROPERTIES
# ============================================================

@app.route("/recent")
def recent():

    df = load_database()

    if df.empty:

        properties = []

    else:

        dates = pd.to_datetime(
            df["capture_date"],
            errors="coerce"
        )

        today = pd.Timestamp(
            datetime.now().date()
        )

        age_days = (
            today
            -
            dates
        ).dt.days

        recent_df = df[
            dates.notna()
            &
            (age_days >= 0)
            &
            (age_days <= MAX_AGE_DAYS)
        ].copy()

        properties = (
            recent_df
            .to_dict("records")
        )

    return render_template(
        "recent.html",
        properties=properties
    )


# Endpoint alias

app.add_url_rule(
    "/recent",
    endpoint="recent_properties",
    view_func=recent
)


# ============================================================
# UPLOAD / DISCOVER
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    # --------------------------------------------------------
    # Check image field
    # --------------------------------------------------------

    if "image" not in request.files:

        return render_template(
            "discover.html",
            error="Please select an image."
        )

    file = request.files["image"]

    if file.filename == "":

        return render_template(
            "discover.html",
            error="Please select an image."
        )

    # --------------------------------------------------------
    # Secure filename
    # --------------------------------------------------------

    filename = secure_filename(
        file.filename
    )

    if not allowed_file(filename):

        return render_template(
            "discover.html",
            error="Unsupported image format."
        )

    # --------------------------------------------------------
    # Save uploaded image
    # --------------------------------------------------------

    upload_path = (
        UPLOAD_FOLDER
        /
        filename
    )

    file.save(
        upload_path
    )

    # ========================================================
    # DUPLICATE IMAGE CHECK
    # ========================================================

    try:

        new_hash = calculate_image_hash(
            upload_path
        )

        database = load_database()

        duplicate = False

        if not database.empty:

            for existing_name in (
                database["image_name"]
                .astype(str)
                .str.strip()
            ):

                dataset_path = (
                    IMAGE_FOLDER
                    /
                    existing_name
                )

                upload_existing_path = (
                    UPLOAD_FOLDER
                    /
                    existing_name
                )

                comparison_path = None

                if dataset_path.exists():

                    comparison_path = (
                        dataset_path
                    )

                elif upload_existing_path.exists():

                    comparison_path = (
                        upload_existing_path
                    )

                if comparison_path is not None:

                    try:

                        existing_hash = (
                            calculate_image_hash(
                                comparison_path
                            )
                        )

                        if (
                            existing_hash
                            ==
                            new_hash
                        ):

                            duplicate = True

                            break

                    except Exception:

                        pass

        if duplicate:

            upload_path.unlink(
                missing_ok=True
            )

            return render_template(
                "discover.html",
                error=(
                    "This image has already "
                    "been uploaded. The "
                    "property was not added again."
                )
            )

    except Exception as e:

        print(
            "Duplicate check error:",
            e
        )

    # ========================================================
    # GET METADATA
    # ========================================================

    metadata = get_metadata(
        filename
    )

    capture_date = metadata.get(
        "capture_date",
        ""
    )

    latitude = metadata.get(
        "latitude",
        ""
    )

    longitude = metadata.get(
        "longitude",
        ""
    )

    # ========================================================
    # RECENT IMAGE CHECK
    # ========================================================

    recent_image, age_days = (
        check_recent_image(
            capture_date
        )
    )

    if not recent_image:

        upload_path.unlink(
            missing_ok=True
        )

        if age_days is not None:

            return render_template(
                "discover.html",
                error=(
                    f"This imagery is "
                    f"{age_days} days old. "
                    f"Only imagery from the "
                    f"last {MAX_AGE_DAYS} days "
                    f"is accepted."
                )
            )

        return render_template(
            "discover.html",
            error="Invalid capture date."
        )

    # ========================================================
    # AI + OCR PROCESSING
    # ========================================================

    try:

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        yolo_results = model.predict(
            source=str(upload_path),
            conf=0.10,
            verbose=False
        )

        detection = yolo_results[0]

        if (
            detection.boxes is None
            or
            len(detection.boxes) == 0
        ):

            upload_path.unlink(
                missing_ok=True
            )

            return render_template(
                "discover.html",
                error=(
                    "No property advertisement "
                    "was detected in the image."
                )
            )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidences = (
            detection.boxes
            .conf
            .tolist()
        )

        best_confidence = max(
            confidences
        )

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        ocr_text = read_text(
            str(upload_path)
        )

        print()
        print(
            "OCR TEXT"
        )
        print(
            "----------------------------------------"
        )
        print(
            ocr_text
        )
        print(
            "----------------------------------------"
        )

        # ----------------------------------------------------
        # Extract property details
        # ----------------------------------------------------

        details = (
            extract_property_details(
                ocr_text
            )
        )

        property_type = (
            classify_property(
                ocr_text
            )
        )

        # ----------------------------------------------------
        # Values from extraction
        # ----------------------------------------------------

        detected_property_type = (
            details.get(
                "property_type",
                property_type
            )
        )

        if (
            not detected_property_type
            or
            detected_property_type
            == "UNKNOWN"
        ):

            detected_property_type = (
                property_type
            )

        price = details.get(
            "price",
            "Not detected"
        )

        area = details.get(
            "area",
            "Not detected"
        )

        bhk = details.get(
            "bhk",
            "Not detected"
        )

        location = details.get(
            "location",
            "Not detected"
        )

        # ----------------------------------------------------
        # GPS
        # ----------------------------------------------------

        # If metadata did not contain GPS,
        # use recognized OCR location
        # for prototype/demo coordinates.

        if (
            not latitude
            or
            not longitude
        ):

            detected_latitude, detected_longitude = (
                get_location_coordinates(
                    location
                )
            )

            if detected_latitude:

                latitude = (
                    detected_latitude
                )

            if detected_longitude:

                longitude = (
                    detected_longitude
                )

        # ----------------------------------------------------
        # Final GPS fallback
        # ----------------------------------------------------

        if not latitude:

            latitude = "Not available"

        if not longitude:

            longitude = "Not available"

        # ====================================================
        # FINAL RESULT
        # ====================================================

        result_data = {

            "image_name":
                filename,

            "property_type":
                detected_property_type,

            "price":
                price,

            "area":
                area,

            "bhk":
                bhk,

            "location":
                location,

            "capture_date":
                capture_date,

            "latitude":
                latitude,

            "longitude":
                longitude,

            "confidence":
                round(
                    float(
                        best_confidence
                    ),
                    4
                )
        }

        # ====================================================
        # SAVE TO PROPERTY DATABASE
        # ====================================================

        database = load_database()

        existing = database[
            database["image_name"]
            .astype(str)
            .str.lower()
            .str.strip()
            ==
            filename.lower()
        ]

        if existing.empty:

            new_property = pd.DataFrame(
                [{
                    "image_name":
                        result_data[
                            "image_name"
                        ],

                    "property_type":
                        result_data[
                            "property_type"
                        ],

                    "price":
                        result_data[
                            "price"
                        ],

                    "area":
                        result_data[
                            "area"
                        ],

                    "bhk":
                        result_data[
                            "bhk"
                        ],

                    "location":
                        result_data[
                            "location"
                        ],

                    "capture_date":
                        result_data[
                            "capture_date"
                        ],

                    "latitude":
                        result_data[
                            "latitude"
                        ],

                    "longitude":
                        result_data[
                            "longitude"
                        ]
                }]
            )

            database = pd.concat(
                [
                    database,
                    new_property
                ],
                ignore_index=True
            )

            database = (
                database
                .drop_duplicates(
                    subset=[
                        "image_name"
                    ],
                    keep="first"
                )
            )

            save_database(
                database
            )

        # ====================================================
        # TERMINAL OUTPUT
        # ====================================================

        print()
        print(
            "========================================"
        )
        print(
            "NEW PROPERTY ANALYZED"
        )
        print(
            "========================================"
        )

        print(
            "Image       :",
            result_data["image_name"]
        )

        print(
            "Type        :",
            result_data["property_type"]
        )

        print(
            "Price       :",
            result_data["price"]
        )

        print(
            "Area        :",
            result_data["area"]
        )

        print(
            "BHK         :",
            result_data["bhk"]
        )

        print(
            "Location    :",
            result_data["location"]
        )

        print(
            "Date        :",
            result_data["capture_date"]
        )

        print(
            "Latitude    :",
            result_data["latitude"]
        )

        print(
            "Longitude   :",
            result_data["longitude"]
        )

        print(
            "Confidence  :",
            result_data["confidence"]
        )

        print(
            "========================================"
        )

        print()

        # ----------------------------------------------------
        # Show result on Discover page
        # ----------------------------------------------------

        return render_template(
            "discover.html",
            result=result_data
        )

    except Exception as e:

        print(
            "Processing error:",
            e
        )

        return render_template(
            "discover.html",
            error=(
                f"Processing failed: "
                f"{str(e)}"
            )
        )


# ============================================================
# PROPERTY IMAGE
# ============================================================

@app.route(
    "/property_image/<filename>"
)
def property_image(filename):

    safe_filename = secure_filename(
        filename
    )

    # First check dataset images

    dataset_file = (
        IMAGE_FOLDER
        /
        safe_filename
    )

    if dataset_file.exists():

        return send_from_directory(
            IMAGE_FOLDER,
            safe_filename
        )

    # Then check uploaded images

    upload_file = (
        UPLOAD_FOLDER
        /
        safe_filename
    )

    if upload_file.exists():

        return send_from_directory(
            UPLOAD_FOLDER,
            safe_filename
        )

    return (
        "Image not found.",
        404
    )


# ============================================================
# UPLOADED IMAGE
# ============================================================

@app.route(
    "/uploaded_image/<filename>"
)
def uploaded_image(filename):

    safe_filename = secure_filename(
        filename
    )

    return send_from_directory(
        UPLOAD_FOLDER,
        safe_filename
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "========================================"
    )
    print(
        "PROPERTY DISCOVERY SYSTEM"
    )
    print(
        "========================================"
    )
    print(
        "Server: http://127.0.0.1:5000"
    )
    print(
        "========================================"
    )
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )