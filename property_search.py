import pandas as pd
import re
from datetime import datetime


# =========================================================
# LOAD PROPERTY DATABASE
# =========================================================

DATABASE_FILE = "dataset/property_database.csv"

data = pd.read_csv(DATABASE_FILE)

data.columns = data.columns.str.strip()


# =========================================================
# CLEAN TEXT FIELDS
# =========================================================

data["location"] = (
    data["location"]
    .astype(str)
    .str.strip()
    .str.lower()
)

data["property_type"] = (
    data["property_type"]
    .astype(str)
    .str.strip()
    .str.lower()
)

data["price"] = (
    data["price"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# =========================================================
# CONVERT PRICE TO LAKHS
# =========================================================

def extract_price(price_text):

    match = re.search(
        r"\d+(?:\.\d+)?",
        price_text
    )

    if not match:
        return None

    return float(match.group())


data["price_lakhs"] = data["price"].apply(
    extract_price
)


# =========================================================
# CONVERT CAPTURE DATE
# =========================================================

data["capture_date"] = pd.to_datetime(
    data["capture_date"],
    errors="coerce"
)


# =========================================================
# CALCULATE IMAGE AGE
# =========================================================

today = pd.Timestamp(
    datetime.now().date()
)

data["age_days"] = (
    today - data["capture_date"]
).dt.days


# =========================================================
# USER SEARCH INPUT
# =========================================================

search_location = input(
    "Enter location to search: "
).strip().lower()

search_type = input(
    "Enter property type (FOR SALE / FOR RENT): "
).strip().lower()

max_price = float(
    input(
        "Enter maximum price in lakhs: "
    )
)

max_age = int(
    input(
        "Enter maximum image age in days: "
    )
)


# =========================================================
# FILTER PROPERTIES
# =========================================================

results = data[
    (data["location"] == search_location) &
    (data["property_type"] == search_type) &
    (data["price_lakhs"].notna()) &
    (data["price_lakhs"] <= max_price) &
    (data["age_days"].notna()) &
    (data["age_days"] >= 0) &
    (data["age_days"] <= max_age)
]


# =========================================================
# DISPLAY SEARCH RESULTS
# =========================================================

print("\nSearch Results:")
print("============================")


if results.empty:

    print("No properties found.")

else:

    display_columns = [
        "image_name",
        "property_type",
        "price",
        "area",
        "bhk",
        "location",
        "capture_date",
        "age_days",
        "latitude",
        "longitude"
    ]

    print(
        results[display_columns]
        .to_string(index=False)
    )

    print(
        "\nTotal properties found:",
        len(results)
    )