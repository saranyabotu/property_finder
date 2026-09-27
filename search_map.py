import pandas as pd
import folium
import re
import os
from datetime import datetime


# =========================================================
# 1. LOAD DATABASE
# =========================================================

DATABASE_FILE = "dataset/property_database.csv"

data = pd.read_csv(DATABASE_FILE)

data.columns = data.columns.str.strip()


# =========================================================
# 2. CLEAN DATA
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
# 3. EXTRACT NUMERIC PRICE
# =========================================================

def extract_price(price_text):

    match = re.search(
        r"\d+(?:\.\d+)?",
        price_text
    )

    if match:
        return float(match.group())

    return None


data["price_lakhs"] = data["price"].apply(
    extract_price
)


# =========================================================
# 4. CAPTURE DATE
# =========================================================

data["capture_date"] = pd.to_datetime(
    data["capture_date"],
    errors="coerce"
)


# =========================================================
# 5. IMAGE AGE
# =========================================================

today = pd.Timestamp(
    datetime.now().date()
)

data["age_days"] = (
    today - data["capture_date"]
).dt.days


# =========================================================
# 6. USER INPUT
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
# 7. FILTER PROPERTIES
# =========================================================

results = data[
    (data["location"] == search_location)
    &
    (data["property_type"] == search_type)
    &
    (data["price_lakhs"].notna())
    &
    (data["price_lakhs"] <= max_price)
    &
    (data["age_days"].notna())
    &
    (data["age_days"] >= 0)
    &
    (data["age_days"] <= max_age)
]


# =========================================================
# 8. SHOW RESULTS
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


    # =====================================================
    # 9. CREATE MAP
    # =====================================================

    first = results.iloc[0]

    property_map = folium.Map(
        location=[
            float(first["latitude"]),
            float(first["longitude"])
        ],
        zoom_start=14
    )


    marker_locations = []


    # =====================================================
    # 10. ADD PROPERTY MARKERS
    # =====================================================

    for _, row in results.iterrows():

        latitude = float(row["latitude"])
        longitude = float(row["longitude"])

        marker_locations.append(
            [latitude, longitude]
        )


        # -------------------------------------------------
        # Property image
        # -------------------------------------------------

        image_file = os.path.abspath(
            os.path.join(
                "dataset",
                "images",
                row["image_name"]
            )
        )

        image_url = (
            "file:///"
            + image_file.replace("\\", "/")
        )


        # -------------------------------------------------
        # Property information
        # -------------------------------------------------

        popup_html = f"""
        <div style="width:280px;">

            <img src="{image_url}"
                 width="250"
                 style="
                 display:block;
                 margin-bottom:10px;
                 ">

            <b>Property Type:</b>
            {row['property_type'].upper()}<br>

            <b>Price:</b>
            {row['price']}<br>

            <b>Area:</b>
            {row['area']}<br>

            <b>BHK:</b>
            {row['bhk']}<br>

            <b>Location:</b>
            {row['location'].title()}<br>

            <b>Capture Date:</b>
            {pd.to_datetime(row['capture_date']).date()}<br>

            <b>Image Age:</b>
            {int(row['age_days'])} days<br>

            <b>Latitude:</b>
            {latitude}<br>

            <b>Longitude:</b>
            {longitude}<br>

        </div>
        """


        # -------------------------------------------------
        # Add marker
        # -------------------------------------------------

        folium.Marker(
            location=[
                latitude,
                longitude
            ],
            popup=folium.Popup(
                popup_html,
                max_width=320
            ),
            tooltip=(
                f"{row['property_type'].upper()} - "
                f"{row['price']}"
            )
        ).add_to(property_map)


    # =====================================================
    # 11. FIT MAP TO ALL PROPERTIES
    # =====================================================

    if marker_locations:

        property_map.fit_bounds(
            marker_locations
        )


    # =====================================================
    # 12. SAVE MAP
    # =====================================================

    map_file = "search_results_map.html"

    property_map.save(map_file)


    print(
        "\nMap created successfully!"
    )

    print(
        "Map file:",
        os.path.abspath(map_file)
    )

    print(
        "Total properties found:",
        len(results)
    )