from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

BATCH_FILE = BASE_DIR / "dataset" / "batch_results.csv"
METADATA_FILE = BASE_DIR / "dataset" / "metadata.csv"
DATABASE_FILE = BASE_DIR / "dataset" / "property_database.csv"

# Read files
batch = pd.read_csv(BATCH_FILE)
metadata = pd.read_csv(METADATA_FILE)

# Clean column names
batch.columns = batch.columns.str.strip()
metadata.columns = metadata.columns.str.strip()

# Clean image names
batch["image_name"] = batch["image_name"].astype(str).str.strip()
metadata["image_name"] = metadata["image_name"].astype(str).str.strip()

# Metadata contains the final date and GPS information,
# so remove these columns from batch before merging.
for col in ["capture_date", "latitude", "longitude"]:
    if col in batch.columns:
        batch = batch.drop(columns=[col])

# Keep only required metadata columns
metadata = metadata[
    ["image_name", "capture_date", "latitude", "longitude"]
]

# Merge AI results with metadata
merged = pd.merge(
    batch,
    metadata,
    on="image_name",
    how="left"
)

# Create final database
database = pd.DataFrame()

database["image_name"] = merged["image_name"]
database["property_type"] = merged["property_type"].fillna("UNKNOWN")
database["price"] = merged["price"].fillna("Not detected")
database["area"] = merged["area"].fillna("Not detected")
database["bhk"] = merged["bhk"].fillna("Not detected")
database["location"] = merged["location"].fillna("Not detected")
database["capture_date"] = merged["capture_date"]
database["latitude"] = merged["latitude"]
database["longitude"] = merged["longitude"]

# Clean text values
for col in [
    "property_type",
    "price",
    "area",
    "bhk",
    "location"
]:
    database[col] = database[col].astype(str).str.strip()

# Remove duplicate images
database = database.drop_duplicates(
    subset=["image_name"],
    keep="first"
)

# Save final database
database.to_csv(DATABASE_FILE, index=False)

print("\n========================================")
print("PROPERTY DATABASE CREATED SUCCESSFULLY")
print("========================================\n")

print(database.to_string(index=False))

print("\nSaved to:")
print(DATABASE_FILE)