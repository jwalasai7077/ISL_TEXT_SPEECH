import os
import math
from PIL import Image, ImageDraw, ImageFont

# ===== Dataset Folder Path =====
DATASET_PATH = r"C:\Users\jwala\OneDrive\Desktop\words\Indian"
OUTPUT_IMAGE = "isl_dataset_grid.png"

# ===== Grid Settings =====
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 600

LABEL_HEIGHT = 140
PADDING = 30

COLUMNS = 7       # Reduced columns so images appear larger
MAX_ROWS = 5

BACKGROUND_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)

# ===== Get Class Folders =====
classes = sorted([
    folder for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
])

# Limit classes
classes = classes[:COLUMNS * MAX_ROWS]

# ===== Create Canvas =====
rows = math.ceil(len(classes) / COLUMNS)

canvas_width = COLUMNS * (IMAGE_WIDTH + PADDING) + PADDING
canvas_height = rows * (IMAGE_HEIGHT + LABEL_HEIGHT + PADDING) + PADDING

canvas = Image.new("RGB", (canvas_width, canvas_height), BACKGROUND_COLOR)
draw = ImageDraw.Draw(canvas)

# ===== Large Bold Font =====
try:
    font = ImageFont.truetype("arialbd.ttf", 70)   # Bold large font
except:
    font = ImageFont.truetype("arial.ttf", 70)

# ===== Add Images =====
for idx, cls in enumerate(classes):

    class_path = os.path.join(DATASET_PATH, cls)

    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if len(images) == 0:
        continue

    # ===== Open Image =====
    img_path = os.path.join(class_path, images[0])

    img = Image.open(img_path).convert("RGB")

    # ===== Resize with High Quality =====
    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)

    row = idx // COLUMNS
    col = idx % COLUMNS

    x = PADDING + col * (IMAGE_WIDTH + PADDING)
    y = PADDING + row * (IMAGE_HEIGHT + LABEL_HEIGHT + PADDING)

    # ===== Paste Image =====
    canvas.paste(img, (x, y))

    # ===== Label Background =====
    label_y = y + IMAGE_HEIGHT

    draw.rectangle(
        [x, label_y, x + IMAGE_WIDTH, label_y + LABEL_HEIGHT],
        fill=(230, 230, 230)
    )

    # ===== Center Label =====
    bbox = draw.textbbox((0, 0), cls, font=font)

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    text_x = x + (IMAGE_WIDTH - text_width) // 2
    text_y = label_y + (LABEL_HEIGHT - text_height) // 2 - 10

    draw.text(
        (text_x, text_y),
        cls,
        fill=TEXT_COLOR,
        font=font
    )

# ===== Save Output =====
canvas.save(OUTPUT_IMAGE, quality=100)

print(f"Grid image saved as: {OUTPUT_IMAGE}")