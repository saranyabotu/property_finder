import folium
import pandas as pd
import os

# Read property database
data = pd.read_csv("dataset/property_database.csv")

# Create map centered around the first property
first_latitude = data.iloc[0]["latitude"]
first_longitude = data.iloc[0]["longitude"]

property_map = folium.Map(
    location=[first_latitude, first_longitude],
    zoom_start=15
)

# Add each property to the map
for _, row in data.iterrows():

    # Get the full path of the advertisement image
    image_file = os.path.abspath(
        os.path.join("dataset", "images", row["image_name"])
    )

    # Convert local path to browser file URL
    image_url = "file:///" + image_file.replace("\\", "/")

    popup_html = f"""
    <div style="width:250px;">
        <img src="{image_url}"
             width="230"
             style="display:block; margin-bottom:10px;">
        
        <b>Property:</b> {row['property_type']}<br>
        <b>Price:</b> {row['price']}<br>
        <b>Area:</b> {row['area']}<br>
        <b>Location:</b> {row['location']}<br>
        <b>Capture Date:</b> {row['capture_date']}
    </div>
    """

    folium.Marker(
        location=[row["latitude"], row["longitude"]],
        popup=folium.Popup(popup_html, max_width=300)
    ).add_to(property_map)

# Save the map
property_map.save("property_map.html")

print("Property map with images created successfully!")