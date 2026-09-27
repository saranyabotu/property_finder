import pandas as pd
from datetime import datetime

# Load metadata
data = pd.read_csv("dataset/metadata.csv")

# Convert capture_date into date format
data["capture_date"] = pd.to_datetime(data["capture_date"])

# Today's date
today = pd.Timestamp(datetime.now().date())

# Number of days allowed
max_days = 30

# Calculate how old each image is
data["age_days"] = (today - data["capture_date"]).dt.days

# Keep only recent images
recent_images = data[data["age_days"] <= max_days]

print("Recent images:")
print(recent_images)