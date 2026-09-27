import pandas as pd

database_file = "dataset/property_database.csv"

# Read the property database
data = pd.read_csv(database_file)

print("Property Database:")
print(data)