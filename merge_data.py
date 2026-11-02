import os
import shutil

source_folders = ["alphabet_data_v1", "gesture_data"]
destination = "merged_dataset"

os.makedirs(destination, exist_ok=True)

for folder in source_folders:
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            src = os.path.join(folder, file)
            dst = os.path.join(destination, file)

            # Handle duplicate names
            if os.path.exists(dst):
                base, ext = os.path.splitext(file)
                count = 1
                while os.path.exists(dst):
                    dst = os.path.join(destination, f"{base}_{count}{ext}")
                    count += 1

            shutil.copy2(src, dst)

print("All CSV files copied successfully!")