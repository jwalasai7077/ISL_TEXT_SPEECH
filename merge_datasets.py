import pandas as pd
import os

alphabet_path = "alphabet_data_v1"
gesture_path = "gesture_data"

merged_data = []

# Read alphabet csv files
for file in os.listdir(alphabet_path):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(alphabet_path, file))
        df["label"] = file.replace(".csv", "")
        merged_data.append(df)

# Read gesture csv files
for file in os.listdir(gesture_path):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(gesture_path, file))
        df["label"] = file.replace(".csv", "")
        merged_data.append(df)

# Combine everything
merged_df = pd.concat(merged_data, ignore_index=True)

# Save
merged_df.to_csv("merged_dataset.csv", index=False)

print("Merged successfully!")
print("Total samples:", len(merged_df))
print("Total classes:", merged_df["label"].nunique())