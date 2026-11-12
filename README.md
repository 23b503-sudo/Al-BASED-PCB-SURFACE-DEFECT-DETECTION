# Al-BASED-PCB-SURFACE-DEFECT-DETECTION
!pip install -q ultralytics opencv-python pandas numpy scikit-learn matplotlib seaborn joblib
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from ultralytics import YOLO

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from google.colab import files

uploaded = files.upload()
print(list(uploaded.keys()))
import zipfile
import os

# Your actual uploaded dataset filename
zip_file = "/content/DeepPCB_without_Mousebite.zip"

# Extract the dataset
extract_path = "/content/PCB_Dataset"

with zipfile.ZipFile(zip_file, "r") as zip_ref:
    zip_ref.extractall(extract_path)

print("Dataset extracted successfully!")
import os

for root, dirs, files in os.walk("/content/PCB_Dataset"):
    level = root.replace("/content/PCB_Dataset", "").count(os.sep)

    if level <= 2:
        print("  " * level + os.path.basename(root) + "/")

        for file in files[:5]:
            print("  " * (level + 1) + file)
import os

root_path = "/content/PCB_Dataset"

image_count = 0
txt_count = 0
xml_count = 0

for root, dirs, files in os.walk(root_path):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            image_count += 1
        elif file.lower().endswith(".txt"):
            txt_count += 1
        elif file.lower().endswith(".xml"):
            xml_count += 1

print("Images:", image_count)
print("TXT annotation files:", txt_count)
print("XML annotation files:", xml_count)
import os

root_path = "/content/PCB_Dataset"

for root, dirs, files in os.walk(root_path):
    level = root.replace(root_path, "").count(os.sep)

    if level <= 3:
        print("  " * level + os.path.basename(root) + "/")

        for file in files[:3]:
            print("  " * (level + 1) + file)
import os

root_path = "/content/PCB_Dataset"

for root, dirs, files in os.walk(root_path):
    txt_files = [f for f in files if f.endswith(".txt")]

    if txt_files:
        first_file = os.path.join(root, txt_files[0])

        print("Annotation file:", first_file)

        with open(first_file, "r") as f:
            print("Annotation contents:")
            print(f.read()[:1000])

        break
import os
from collections import Counter

DATASET_PATH = "/content/PCB_Dataset"

# Show the main folders
print("Dataset folders:")
print(os.listdir(DATASET_PATH))

# Count images and labels in each split
for split in ["train", "valid", "val", "test"]:
    image_dir = os.path.join(DATASET_PATH, split, "images")
    label_dir = os.path.join(DATASET_PATH, split, "labels")

    if os.path.isdir(image_dir):
        image_count = len([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])
        print(f"{split}: {image_count} images")

    if os.path.isdir(label_dir):
        label_count = len([
            f for f in os.listdir(label_dir)
            if f.lower().endswith(".txt")
        ])
        print(f"{split}: {label_count} label files")

# Count class IDs in all annotation files
class_counts = Counter()

for root, dirs, files in os.walk(DATASET_PATH):
    for filename in files:
        if filename.endswith(".txt"):
            filepath = os.path.join(root, filename)

            with open(filepath, "r") as f:
                for line in f:
                    parts = line.strip().split()

                    if len(parts) == 5:
                        class_counts[int(parts[0])] += 1

print("\nClass IDs found:")
print(dict(sorted(class_counts.items())))
import os

for root, dirs, files in os.walk("/content/PCB_Dataset"):
    for filename in files:
        if filename.lower() == "data.yaml":
            path = os.path.join(root, filename)

            print("Found configuration:", path)

            with open(path, "r") as f:
                print(f.read())
import yaml
import os

DATASET_PATH = "/content/PCB_Dataset"

data_config = {
    "path": DATASET_PATH,
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "names": {
        0: "copper",
        1: "open",
        2: "pinhole",
        3: "short",
        4: "spur"
    }
}

yaml_path = "/content/pcb_data.yaml"

with open(yaml_path, "w") as f:
    yaml.safe_dump(data_config, f, sort_keys=False)

print(open(yaml_path).read())
import os

DATASET_PATH = "/content/PCB_Dataset"

for split in ["train", "valid", "test"]:
    image_dir = os.path.join(DATASET_PATH, split, "images")
    label_dir = os.path.join(DATASET_PATH, split, "labels")

    if not os.path.isdir(image_dir):
        print(f"{split}: IMAGE FOLDER MISSING")
        continue

    if not os.path.isdir(label_dir):
        print(f"{split}: LABEL FOLDER MISSING")
        continue

    images = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    labels = [
        f for f in os.listdir(label_dir)
        if f.lower().endswith(".txt")
    ]

    image_stems = {os.path.splitext(f)[0] for f in images}
    label_stems = {os.path.splitext(f)[0] for f in labels}

    print(f"\n{split.upper()}")
    print("Images:", len(images))
    print("Labels:", len(labels))
    print("Images without matching labels:", len(image_stems - label_stems))
    print("Labels without matching images:", len(label_stems - image_stems))
import torch

print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("GPU not available. Select Runtime > Change runtime type > GPU.")
import torch

print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("GPU is not enabled.")
!pip install -q ultralytics
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data="/content/pcb_data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    workers=2,
    seed=42,
    device=0,
    project="/content/pcb_yolo",
    name="yolo11n_pcb"
)
import os
import yaml

DATASET_PATH = "/content/PCB_Dataset"

# Check that your dataset is available
print("Dataset exists:", os.path.exists(DATASET_PATH))
print("Dataset folders:", os.listdir(DATASET_PATH))

# Create the YOLO configuration
data_config = {
    "path": DATASET_PATH,
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "names": {
        0: "copper",
        1: "open",
        2: "pinhole",
        3: "short",
        4: "spur"
    }
}

yaml_path = "/content/pcb_data.yaml"

with open(yaml_path, "w") as f:
    yaml.safe_dump(data_config, f, sort_keys=False)

print("\nConfiguration created:", os.path.exists(yaml_path))
print(open(yaml_path).read())
from google.colab import files

uploaded = files.upload()
import zipfile
import os

# Your actual uploaded dataset filename
zip_file = "/content/DeepPCB_without_Mousebite.zip"

# Extract the dataset
extract_path = "/content/PCB_Dataset"

with zipfile.ZipFile(zip_file, "r") as zip_ref:
    zip_ref.extractall(extract_path)

print("Dataset extracted successfully!")
import os

DATASET_PATH = "/content/PCB_Dataset"

print("Dataset exists:", os.path.exists(DATASET_PATH))
print("Folders:", os.listdir(DATASET_PATH))

for split in ["train", "valid", "test"]:
    image_dir = os.path.join(DATASET_PATH, split, "images")
    label_dir = os.path.join(DATASET_PATH, split, "labels")

    print(f"\n{split.upper()}")
    print("Images folder exists:", os.path.isdir(image_dir))
    print("Labels folder exists:", os.path.isdir(label_dir))
import yaml
import os

data_config = {
    "path": "/content/PCB_Dataset",
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "names": {
        0: "copper",
        1: "open",
        2: "pinhole",
        3: "short",
        4: "spur"
    }
}

yaml_path = "/content/pcb_data.yaml"

with open(yaml_path, "w") as f:
    yaml.safe_dump(data_config, f, sort_keys=False)

print("Configuration created:", os.path.exists(yaml_path))
print(open(yaml_path).read())
import torch

print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
!pip install -q ultralytics
from ultralytics import YOLO
import torch

model = YOLO("yolo11n.pt")

results = model.train(
    data="/content/pcb_data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    patience=20,
    workers=2,
    seed=42,
    device=0 if torch.cuda.is_available() else "cpu",
    project="/content/pcb_yolo",
    name="yolo11n_pcb"
)
from ultralytics import YOLO
import os

MODEL_PATH = "/content/pcb_yolo/yolo11n_pcb-2/weights/best.pt"

print("Model exists:", os.path.exists(MODEL_PATH))

best_model = YOLO(MODEL_PATH)

print("Model loaded successfully!")
print("Classes:", best_model.names)
metrics = best_model.val(
    data="/content/pcb_data.yaml",
    split="test",
    imgsz=640,
    batch=8
)

print("\n========== PCB TEST RESULTS ==========")
print(f"Precision:  {metrics.box.mp:.4f}")
print(f"Recall:     {metrics.box.mr:.4f}")
print(f"mAP@50:     {metrics.box.map50:.4f}")
print(f"mAP@50-95:  {metrics.box.map:.4f}")
import glob
from IPython.display import display, Image

test_images = glob.glob(
    "/content/PCB_Dataset/test/images/*"
)

if test_images:
    test_image = test_images[0]

    predictions = best_model.predict(
        source=test_image,
        conf=0.25,
        imgsz=640,
        save=True
    )

    output_path = predictions[0].save_dir

    print("Input image:", test_image)
    print("Output folder:", output_path)

    display(Image(filename=predictions[0].save_dir + "/" + test_image.split("/")[-1]))
else:
    print("No test images found.")
from ultralytics import YOLO

MODEL_PATH = "/content/pcb_yolo/yolo11n_pcb-2/weights/best.pt"

best_model = YOLO(MODEL_PATH)

metrics = best_model.val(
    data="/content/pcb_data.yaml",
    split="test",
    imgsz=640,
    batch=8
)

print("\n===== PCB TEST RESULTS =====")
print(f"Precision:  {metrics.box.mp:.4f}")
print(f"Recall:     {metrics.box.mr:.4f}")
print(f"mAP50:      {metrics.box.map50:.4f}")
print(f"mAP50-95:   {metrics.box.map:.4f}")
!pip install -q opencv-python pandas scikit-learn joblib
import os
import cv2
import numpy as np
import pandas as pd

DATASET_PATH = "/content/PCB_Dataset"

CLASS_NAMES = {
    0: "copper",
    1: "open",
    2: "pinhole",
    3: "short",
    4: "spur"
}

def extract_features(crop):
    if crop is None or crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    x, y, w, h = cv2.boundingRect(contour)

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    return {
        "area": area,
        "perimeter": perimeter,
        "width": w,
        "height": h,
        "aspect_ratio": w / h if h else 0,
        "circularity": (
            4 * np.pi * area / (perimeter ** 2)
            if perimeter else 0
        ),
        "solidity": (
            area / hull_area if hull_area else 0
        ),
        "mean_intensity": float(np.mean(gray)),
        "std_intensity": float(np.std(gray)),
        "edge_density": float(
            np.mean(cv2.Canny(gray, 50, 150) > 0)
        )
    }

print("Feature extraction function ready!")
def yolo_to_pixels(parts, image_width, image_height):
    class_id = int(parts[0])

    xc, yc, bw, bh = map(float, parts[1:5])

    x1 = int((xc - bw / 2) * image_width)
    y1 = int((yc - bh / 2) * image_height)
    x2 = int((xc + bw / 2) * image_width)
    y2 = int((yc + bh / 2) * image_height)

    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))

    return class_id, x1, y1, x2, y2


def build_feature_dataset(split="train"):
    image_dir = os.path.join(DATASET_PATH, split, "images")
    label_dir = os.path.join(DATASET_PATH, split, "labels")

    rows = []

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):
            continue

        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(
            label_dir,
            os.path.splitext(filename)[0] + ".txt"
        )

        if not os.path.exists(label_path):
            continue

        image = cv2.imread(image_path)

        if image is None:
            continue

        H, W = image.shape[:2]

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                class_id, x1, y1, x2, y2 = yolo_to_pixels(
                    parts, W, H
                )

                if class_id not in CLASS_NAMES:
                    continue

                crop = image[y1:y2, x1:x2]

                features = extract_features(crop)

                if features is None:
                    continue

                features["class_id"] = class_id
                features["class_name"] = CLASS_NAMES[class_id]
                features["image_name"] = filename

                rows.append(features)

    return pd.DataFrame(rows)


train_features = build_feature_dataset("train")

print("Feature dataset shape:", train_features.shape)
print(train_features.head())
train_features.to_csv(
    "/content/pcb_train_features.csv",
    index=False
)

print("Feature dataset saved!")
print("Number of defect samples:", len(train_features))
print("\nClass distribution:")
print(train_features["class_name"].value_counts())

print("\nFeature columns:")
print(train_features.columns.tolist())
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

FEATURE_COLUMNS = [
    "area",
    "perimeter",
    "width",
    "height",
    "aspect_ratio",
    "circularity",
    "solidity",
    "mean_intensity",
    "std_intensity",
    "edge_density"
]

df = train_features.dropna(
    subset=FEATURE_COLUMNS + ["class_id"]
).copy()

X = df[FEATURE_COLUMNS]
y = df["class_id"]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_val)

print("Validation accuracy:", accuracy_score(y_val, y_pred))

print("\nClassification report:")
print(classification_report(
    y_val,
    y_pred,
    labels=[0, 1, 2, 3, 4],
    target_names=[
        "copper",
        "open",
        "pinhole",
        "short",
        "spur"
    ],
    zero_division=0
))
joblib.dump(
    rf_model,
    "/content/pcb_random_forest.pkl"
)

joblib.dump(
    FEATURE_COLUMNS,
    "/content/pcb_feature_columns.pkl"
)

print("Random Forest saved successfully!")
from google.colab import files

files.download(
    "/content/pcb_random_forest.pkl"
)
files.download(
    "/content/pcb_feature_columns.pkl"
)
files.download(
    "/content/pcb_yolo/yolo11n_pcb-2/weights/best.pt"
)
import joblib
import pandas as pd

rf_model = joblib.load(
    "/content/pcb_random_forest.pkl"
)

FEATURE_COLUMNS = joblib.load(
    "/content/pcb_feature_columns.pkl"
)

# Select one extracted defect
sample = train_features.iloc[[0]]

# Prepare its features
X_sample = sample[FEATURE_COLUMNS]

# Predict its class
predicted_class = rf_model.predict(X_sample)[0]

class_names = {
    0: "copper",
    1: "open",
    2: "pinhole",
    3: "short",
    4: "spur"
}

print("Actual class:", sample["class_name"].iloc[0])
print("Predicted class:", class_names[predicted_class])
from ultralytics import YOLO
import joblib
import os

# Paths to your trained models
YOLO_PATH = "/content/pcb_yolo/yolo11n_pcb-2/weights/best.pt"
RF_PATH = "/content/pcb_random_forest.pkl"
FEATURE_PATH = "/content/pcb_feature_columns.pkl"

# Check that files exist
for path in [YOLO_PATH, RF_PATH, FEATURE_PATH]:
    print(path, "->", os.path.exists(path))

# Load models
yolo_model = YOLO(YOLO_PATH)
rf_model = joblib.load(RF_PATH)
FEATURE_COLUMNS = joblib.load(FEATURE_PATH)

CLASS_NAMES = {
    0: "copper",
    1: "open",
    2: "pinhole",
    3: "short",
    4: "spur"
}

print("\nBoth models loaded successfully!")
print("YOLO classes:", yolo_model.names)
!pip install ultralytics joblib opencv-python-headless
from ultralytics import YOLO
import joblib
import os
import cv2
import numpy as np
import pandas as pd

print("All libraries imported successfully!")
import os

paths = [
    "/content/pcb_yolo/yolo11n_pcb-2/weights/best.pt",
    "/content/pcb_random_forest.pkl",
    "/content/pcb_feature_columns.pkl"
]

for path in paths:
    print(path, "->", os.path.exists(path))
import os

print("Searching for trained models...\n")

for root, dirs, files in os.walk("/content"):
    for file in files:
        if file in [
            "best.pt",
            "last.pt",
            "pcb_random_forest.pkl",
            "pcb_feature_columns.pkl"
        ]:
            print(os.path.join(root, file))
from google.colab import drive
drive.mount('/content/drive')
import os

print("Checking trained models...")

for root, dirs, files in os.walk("/content"):
    for file in files:
        if file in [
            "best.pt",
            "last.pt",
            "pcb_random_forest.pkl",
            "pcb_feature_columns.pkl"
        ]:
            print(os.path.join(root, file))
import os

print("Checking dataset and model folders...")

for path in [
    "/content/PCB_Dataset",
    "/content/pcb_yolo",
    "/content/pcb_random_forest.pkl",
    "/content/pcb_feature_columns.pkl"
]:
    print(path, "EXISTS:", os.path.exists(path))
from google.colab import drive
drive.mount('/content/drive')
from google.colab import drive
drive.mount('/content/drive')
from google.colab import drive
drive.mount('/content/drive')
from google.colab import drive
drive.mount('/content/drive')
from google.colab import drive
drive.mount('/content/drive')
import os

print("Checking Google Drive for trained models...\n")

for root, dirs, files in os.walk("/content/drive/MyDrive"):
    for file in files:
        if file in [
            "best.pt",
            "last.pt",
            "pcb_random_forest.pkl",
            "pcb_feature_columns.pkl"
        ]:
            print(os.path.join(root, file))
import os

print("Checking Colab files...")

for root, dirs, files in os.walk("/content"):
    for file in files:
        if file in [
            "best.pt",
            "last.pt",
            "pcb_random_forest.pkl",
            "pcb_feature_columns.pkl"
        ]:
            print(os.path.join(root, file))
import os

folder = "/content/drive/MyDrive"

print("Checking Google Drive folders...\n")

for item in os.listdir(folder):
    path = os.path.join(folder, item)
    if os.path.isdir(path):
        print("Folder:", item)
    else:
        print("File:", item)
import os

base = "/content/drive/MyDrive"

print("Searching for PCB datasets and trained models...\n")

keywords = [
    "pcb", "yolo", "best.pt", "last.pt",
    "random_forest", "feature_columns"
]

for root, dirs, files in os.walk(base):
    for name in dirs + files:
        if any(word in name.lower() for word in keywords):
            print(os.path.join(root, name))
import os

base = "/content/drive/MyDrive"

print("Top-level folders and files:\n")

for item in os.listdir(base):
    print(item)
import os

base = "/content/drive/MyDrive"

print("Checking for PCB project folders:\n")

for item in os.listdir(base):
    path = os.path.join(base, item)

    if os.path.isdir(path):
        print("FOLDER:", item)
import os

folder = "/content/drive/MyDrive/Muhsina"

print("Checking Muhsina folder...\n")

for root, dirs, files in os.walk(folder):
    for name in files:
        print(os.path.join(root, name))
import os

base = "/content/drive/MyDrive"

print("Searching Google Drive for PCB model files...\n")

keywords = [
    "best.pt",
    "last.pt",
    "pcb_random_forest.pkl",
    "pcb_feature_columns.pkl"
]

for root, dirs, files in os.walk(base):
    for name in files:
        if name in keywords:
            print("FOUND:", os.path.join(root, name))

print("\nSearch completed!")
import os

paths = [
    "/content/pcb_random_forest.pkl",
    "/content/pcb_feature_columns.pkl",
    "/content/pcb_yolo/yolo11n_pcb-2/weights/best.pt"
]

for path in paths:
    if os.path.exists(path):
        print("FOUND:", path)
    else:
        print("NOT FOUND:", path)
import os

print("Checking model files...")

for file in [
    "/content/best.pt",
    "/content/pcb_feature_columns.pkl",
    "/content/pcb_random_forest.pkl"
]:
    if os.path.exists(file):
        print("FOUND:", file)
    else:
        print("NOT FOUND:", file)
import joblib
import os
import torch

# Load the Random Forest model
rf_model = joblib.load("/content/pcb_random_forest.pkl")

# Load the feature columns
with open("/content/pcb_feature_columns.pkl", "rb") as f:
    feature_columns = joblib.load(f)

# Check the YOLO model file
yolo_path = "/content/best.pt"

print("Random Forest model loaded successfully!")
print("Feature columns loaded successfully!")
print("YOLO model exists:", os.path.exists(yolo_path))
print("PyTorch version:", torch.__version__)
print("Number of features:", len(feature_columns))
from ultralytics import YOLO

# Load the trained YOLO model
yolo_model = YOLO("/content/best.pt")

print("YOLO model loaded successfully!")

# Display model information
yolo_model.info()
!pip install ultralytics
from ultralytics import YOLO

# Load the trained YOLO model
yolo_model = YOLO("/content/best.pt")

print("YOLO model loaded successfully!")

# Display model information
yolo_model.info()
import os
from google.colab import files
from IPython.display import display, Image

# Upload a PCB image from your computer
uploaded = files.upload()

# Get the uploaded image filename
image_path = next(iter(uploaded))

# Run YOLO detection
results = yolo_model.predict(
    source=image_path,
    conf=0.25,
    save=True
)

# Display the detection result
output_path = results[0].save_dir
print("Detection completed!")
print("Results saved in:", output_path)

display(Image(filename=results[0].path))
# Check the YOLO detection results

result = results[0]

print("Number of detections:", len(result.boxes))

if len(result.boxes) > 0:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        print("Defect:", yolo_model.names[class_id])
        print("Confidence:", round(confidence * 100, 2), "%")
else:
    print("No defects detected in this image.")
import matplotlib.pyplot as plt

# Display PCB image with detected defects
result = results[0]

annotated_image = result.plot()

plt.figure(figsize=(12, 8))
plt.imshow(annotated_image[:, :, ::-1])
plt.axis("off")
plt.title("PCB Defect Detection Results")
plt.show()
import pandas as pd

result = results[0]

defect_data = []

for box in result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])

    x1, y1, x2, y2 = box.xyxy[0].tolist()

    defect_data.append({
        "defect": yolo_model.names[class_id],
        "confidence": confidence,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2
    })

df = pd.DataFrame(defect_data)

print("PCB Defect Detection Report")
display(df)
df.to_csv("/content/pcb_defect_report.csv", index=False)

print("Report saved successfully!")

from google.colab import files
files.download("/content/pcb_defect_report.csv")
# PCB Defect Classification using Random Forest

import joblib
import pandas as pd
import os

# Load the trained Random Forest model
rf_model = joblib.load("/content/pcb_random_forest.pkl")

# Load feature columns
feature_columns = joblib.load("/content/pcb_feature_columns.pkl")

print("Random Forest model loaded successfully!")
print("Number of features:", len(feature_columns))
print("Random Forest Expected Features:")
print(feature_columns)

print("\nNumber of features:", len(feature_columns))

print("\nRandom Forest Classes:")
print(rf_model.classes_)
# CELL 74 - CORRECTED FEATURE EXTRACTION
# Use the same feature extraction method as Random Forest training.

import cv2
import numpy as np
import pandas as pd

# Get the YOLO detection result
# FIX: Retrieve result from 'results' instead of 'result'
result = results[0]

# Original image in BGR format
image = result.orig_img

height, width = image.shape[:2]

feature_data = []
valid_detections = []

# Process every YOLO-detected defect
for detection_index, box in enumerate(result.boxes):

    x1, y1, x2, y2 = map(
        int, box.xyxy[0].tolist()
    )

    # Keep coordinates within the image
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    # Extract the original colour crop
    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        print(f"Detection {detection_index + 1}: Empty crop skipped")
        continue

    # IMPORTANT:
    # Use the SAME function used during RF training.
    features = extract_features(crop)

    if features is None:
        print(
            f"Detection {detection_index + 1}: "
            "Features could not be extracted"
        )
        continue

    feature_data.append(features)
    valid_detections.append({
        "Detection_ID": detection_index + 1,
        "YOLO_Class_ID": int(box.cls[0]),
        "YOLO_Defect": yolo_model.names[int(box.cls[0])],
        "YOLO_Confidence": round(
            float(box.conf[0]) * 100, 2
        ),
        "X1": x1,
        "Y1": y1,
        "X2": x2,
        "Y2": y2
    })

# Create feature dataframe in the correct order
features_df = pd.DataFrame(
    feature_data,
    columns=FEATURE_COLUMNS
)

print("Number of detected defects:", len(result.boxes))
print("Number of valid feature sets:", len(features_df))

print("\nExtracted features:")
display(features_df)

# Check that the input features match the model
assert list(features_df.columns) == list(FEATURE_COLUMNS)

if not features_df.empty:
    print("\nFeature statistics:")
    display(features_df.describe().round(3))
import os

# 1. Ensure the image is uploaded or locate your uploaded image
uploaded_images = [f for f in os.listdir('/content') if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

if uploaded_images:
    image_path = os.path.join('/content', uploaded_images[0])
    print(f"Using image: {image_path}")

    # 2. Re-run YOLO prediction to define 'results'
    results = yolo_model.predict(
        source=image_path,
        conf=0.25,
        save=True
    )
    print("Successfully defined 'results'. You can now run the feature extraction cells!")
else:
    print("No image found in /content directory. Please upload an image first!")
from google.colab import files
import cv2

uploaded = files.upload()          # a file picker opens
img_path = list(uploaded.keys())[0]
img = cv2.imread(img_path)         # loads as BGR, same as training
print(img_path, img.shape)


# Run YOLO prediction on the uploaded image to define 'results'
results = yolo_model.predict(
    source=img_path,
    conf=0.25,
    save=True
)

print("Successfully ran YOLO prediction and defined 'results'!")
# CELL 74 - CORRECTED FEATURE EXTRACTION
# Use the same feature extraction method as Random Forest training.

import cv2
import numpy as np
import pandas as pd

# Get the YOLO detection result
result = results[0]

# Original image in BGR format
image = result.orig_img

height, width = image.shape[:2]

feature_data = []
valid_detections = []

# Process every YOLO-detected defect
for detection_index, box in enumerate(result.boxes):

    x1, y1, x2, y2 = map(
        int, box.xyxy[0].tolist()
    )

    # Keep coordinates within the image
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    # Extract the original colour crop
    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        print(f"Detection {detection_index + 1}: Empty crop skipped")
        continue

    # IMPORTANT:
    # Use the SAME function used during RF training.
    features = extract_features(crop)

    if features is None:
        print(
            f"Detection {detection_index + 1}: "
            "Features could not be extracted"
        )
        continue

    feature_data.append(features)
    valid_detections.append({
        "Detection_ID": detection_index + 1,
        "YOLO_Class_ID": int(box.cls[0]),
        "YOLO_Defect": yolo_model.names[int(box.cls[0])],
        "YOLO_Confidence": round(
            float(box.conf[0]) * 100, 2
        ),
        "X1": x1,
        "Y1": y1,
        "X2": x2,
        "Y2": y2
    })

# Create feature dataframe in the correct order
features_df = pd.DataFrame(
    feature_data,
    columns=FEATURE_COLUMNS
)

print("Number of detected defects:", len(result.boxes))
print("Number of valid feature sets:", len(features_df))

print("\nExtracted features:")
display(features_df)

# Check that the input features match the model
assert list(features_df.columns) == list(FEATURE_COLUMNS)

if not features_df.empty:
    print("\nFeature statistics:")
    display(features_df.describe().round(3))
# Predict PCB defect classes using Random Forest

rf_predictions = rf_model.predict(features_df)

print("Random Forest Prediction Results")
print("-" * 40)

for i, prediction in enumerate(rf_predictions):
    print(f"Detection {i + 1}: Predicted class = {prediction}")

# Add predictions to the feature table
features_df["RF_predicted_class"] = rf_predictions

display(features_df)
# CELL 74 - CORRECTED FEATURE EXTRACTION
# Use the same feature extraction method as Random Forest training.

import cv2
import numpy as np
import pandas as pd

# Get the YOLO detection result
result = results[0]

# Original image in BGR format
image = result.orig_img

height, width = image.shape[:2]

feature_data = []
valid_detections = []

# Process every YOLO-detected defect
for detection_index, box in enumerate(result.boxes):

    x1, y1, x2, y2 = map(
        int, box.xyxy[0].tolist()
    )

    # Keep coordinates within the image
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    # Extract the original colour crop
    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        print(f"Detection {detection_index + 1}: Empty crop skipped")
        continue

    # IMPORTANT:
    # Use the SAME function used during RF training.
    features = extract_features(crop)

    if features is None:
        print(
            f"Detection {detection_index + 1}: "
            "Features could not be extracted"
        )
        continue

    feature_data.append(features)
    valid_detections.append({
        "Detection_ID": detection_index + 1,
        "YOLO_Class_ID": int(box.cls[0]),
        "YOLO_Defect": yolo_model.names[int(box.cls[0])],
        "YOLO_Confidence": round(
            float(box.conf[0]) * 100, 2
        ),
        "X1": x1,
        "Y1": y1,
        "X2": x2,
        "Y2": y2
    })

# Create feature dataframe in the correct order
features_df = pd.DataFrame(
    feature_data,
    columns=FEATURE_COLUMNS
)

print("Number of detected defects:", len(result.boxes))
print("Number of valid feature sets:", len(features_df))

print("\nExtracted features:")
display(features_df)

# Check that the input features match the model
assert list(features_df.columns) == list(FEATURE_COLUMNS)

if not features_df.empty:
    print("\nFeature statistics:")
    display(features_df.describe().round(3))
import pandas as pd

result = results[0]

# Get YOLO defect labels and confidence scores
yolo_defects = []

for box in result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])

    yolo_defects.append({
        "YOLO_Defect": yolo_model.names[class_id],
        "YOLO_Confidence": round(confidence * 100, 2)
    })

# Create the combined report
yolo_df = pd.DataFrame(yolo_defects)

rf_df = features_df[["RF_predicted_class"]].reset_index(drop=True)

combined_report = pd.concat(
    [yolo_df.reset_index(drop=True), rf_df],
    axis=1
)

print("FINAL PCB DEFECT DETECTION REPORT")
display(combined_report)
from google.colab import files

combined_report.to_csv(
    "/content/final_pcb_defect_report.csv",
    index=False
)

files.download("/content/final_pcb_defect_report.csv")

print("Final PCB defect report downloaded successfully!")
!pip install gradio ultralytics joblib opencv-python-headless
import gradio as gr
import joblib
import torch
from ultralytics import YOLO

print("All libraries imported successfully!")
print("Gradio version:", gr.__version__)
print("PyTorch version:", torch.__version__)
print("YOLO imported successfully!")
import gradio as gr
from ultralytics import YOLO
from PIL import Image

# Load the trained YOLO model
yolo_model = YOLO("/content/best.pt")

def detect_pcb_defects(image):
    if image is None:
        return None, "Please upload a PCB image."

    # Run defect detection
    results = yolo_model.predict(
        source=image,
        conf=0.25
    )

    # Draw detection boxes
    output_image = results[0].plot()

    # Collect defect details
    detections = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0]) * 100

        defect_name = yolo_model.names[class_id]

        detections.append(
            f"Defect: {defect_name} | "
            f"Confidence: {confidence:.2f}%"
        )

    if detections:
        report = "\n".join(detections)
    else:
        report = "No defects detected."

    return Image.fromarray(output_image[:, :, ::-1]), report


# Create the web interface
app = gr.Interface(
    fn=detect_pcb_defects,
    inputs=gr.Image(type="pil", label="Upload PCB Image"),
    outputs=[
        gr.Image(label="PCB Detection Results"),
        gr.Textbox(label="Defect Report", lines=8)
    ],
    title="PCB Defect Detection System",
    description="Upload a PCB image to detect defects using YOLO."
)

app.launch(share=True)
# Check that both models are available

import os
import joblib

rf_path = "/content/pcb_random_forest.pkl"
features_path = "/content/pcb_feature_columns.pkl"
yolo_path = "/content/best.pt"

for path in [rf_path, features_path, yolo_path]:
    if os.path.exists(path):
        print("FOUND:", os.path.basename(path))
    else:
        print("NOT FOUND:", path)

if os.path.exists(rf_path) and os.path.exists(features_path):
    rf_model = joblib.load(rf_path)
    feature_columns = joblib.load(features_path)

    print("\nRandom Forest model loaded successfully!")
    print("Expected features:", list(feature_columns))
    print("Classes:", rf_model.classes_)
print("Feature columns:")
for i, feature in enumerate(feature_columns, 1):
    print(i, feature)

print("\nRandom Forest classes:")
print(rf_model.classes_)

print("\nYOLO defect classes:")
print(yolo_model.names)
import pandas as pd

# Check that extracted features match the model's expected columns
X_rf = features_df[feature_columns].copy()

print("Feature shape:", X_rf.shape)
print("Expected features:", len(feature_columns))

# Run Random Forest predictions
rf_predictions = rf_model.predict(X_rf)

print("\nRandom Forest predictions:")
print(rf_predictions)

# Add predictions to the existing feature table
features_df["RF_predicted_class"] = rf_predictions

display(features_df)
# Check Random Forest class labels
print("Random Forest classes:", rf_model.classes_)

# Check YOLO defect labels
print("YOLO defect classes:", yolo_model.names)

# Check the probability of each Random Forest class
if hasattr(rf_model, "predict_proba"):
    probabilities = rf_model.predict_proba(X_rf)

    print("\nRandom Forest prediction probabilities:")

    for i, row in enumerate(probabilities):
        print(f"\nDefect {i + 1}:")
        for class_id, probability in zip(rf_model.classes_, row):
            print(f"Class {class_id}: {probability * 100:.2f}%")
# Check how the Random Forest was trained
print("Random Forest classes:", rf_model.classes_)

print("\nRandom Forest training target mapping:")
if hasattr(rf_model, "classes_"):
    for class_id in rf_model.classes_:
        print("Class ID:", class_id)

# Show the most likely class for each defect
print("\nRandom Forest predictions:")
print(rf_model.predict(X_rf))

print("\nPrediction probabilities:")
print(rf_model.predict_proba(X_rf))
print("YOLO class names:")
print(yolo_model.names)

print("\nRandom Forest classes:")
print(rf_model.classes_)

print("\nRandom Forest model details:")
print(rf_model)
print("Random Forest model attributes:")

print("\nNumber of trees:")
print(rf_model.n_estimators)

print("\nFeature importance:")
for name, importance in zip(
    feature_columns,
    rf_model.feature_importances_
):
    print(f"{name}: {importance:.4f}")
# Map Random Forest class IDs to defect names
rf_class_names = {
    0: "copper",
    1: "open",
    2: "pinhole",
    3: "short",
    4: "spur"
}

# Convert Random Forest predictions into readable labels
features_df["RF_predicted_defect"] = [
    rf_class_names.get(int(pred), "Unknown")
    for pred in rf_model.predict(X_rf)
]

# Display the updated Random Forest predictions
print("RANDOM FOREST DEFECT PREDICTIONS")
display(features_df[
    ["RF_predicted_defect"]
])

# Display the final combined report
print("FINAL PCB DEFECT DETECTION REPORT")
display(combined_report)
# Create a readable final PCB defect report

rf_class_names = {
    0: "copper",
    1: "open",
    2: "pinhole",
    3: "short",
    4: "spur"
}

# Add readable Random Forest predictions
combined_report["RF_predicted_defect"] = (
    combined_report["RF_predicted_class"]
    .map(rf_class_names)
)

print("FINAL PCB DEFECT DETECTION REPORT")
display(combined_report)

# Count the defects detected by YOLO
print("\nYOLO Defect Summary:")
print(combined_report["YOLO_Defect"].value_counts())
print("YOLO class names:")
print(yolo_model.names)

print("\nRandom Forest class IDs:")
print(rf_model.classes_)

print("\nRandom Forest feature columns:")
print(feature_columns)

print("\nRandom Forest predictions:")
print(rf_model.predict(X_rf))

print("\nPrediction probabilities:")
print(rf_model.predict_proba(X_rf))
# Check Random Forest model training information

print("Number of trees:")
print(rf_model.n_estimators)

print("\nExpected feature names:")
print(feature_columns)

print("\nModel classes:")
print(rf_model.classes_)

print("\nFeature importance:")
for name, importance in zip(
    feature_columns,
    rf_model.feature_importances_
):
    print(f"{name}: {importance:.4f}")

print("\nInput feature data:")
display(X_rf)
# Check the classes learned by the Random Forest model

print("Random Forest classes:")
print(rf_model.classes_)

print("\nNumber of trees:")
print(rf_model.n_estimators)

print("\nPrediction probabilities for each defect:")
probabilities = rf_model.predict_proba(X_rf)

for i, row in enumerate(probabilities):
    print(f"\nDefect {i + 1}:")

    for class_id, probability in zip(
        rf_model.classes_, row
    ):
        print(
            f"Class {class_id}: "
            f"{probability * 100:.2f}%"
        )

print("\nFinal predicted class IDs:")
print(rf_model.predict(X_rf))
# Check the Random Forest model's training configuration

print("Model classes:", rf_model.classes_)

print("\nNumber of trees:", rf_model.n_estimators)

print("\nFeature count:", rf_model.n_features_in_)

print("\nModel parameters:")
print(rf_model.get_params())

print("\nClass mapping used by YOLO:")
print(yolo_model.names)
# Check how the Random Forest was trained

print("Random Forest training information:")

print("\nClasses learned by the model:")
print(rf_model.classes_)

print("\nTraining sample counts, if training data is available:")

if 'y_train' in globals():
    import pandas as pd
    print(pd.Series(y_train).value_counts().sort_index())

elif 'y' in globals():
    import pandas as pd
    print(pd.Series(y).value_counts().sort_index())

else:
    print("Training labels are not available in the current variables.")

print("\nFeature names:")
print(feature_columns)

print("\nYOLO class names:")
print(yolo_model.names)
# Check Random Forest predictions

print("1. Model classes:")
print(rf_model.classes_)

print("\n2. Number of features:")
print(rf_model.n_features_in_)

print("\n3. Input feature shape:")
print(X_rf.shape)

print("\n4. Predictions:")
predictions = rf_model.predict(X_rf)
print(predictions)

print("\n5. Prediction probabilities:")
probabilities = rf_model.predict_proba(X_rf)

for i, row in enumerate(probabilities):
    print(f"\nDefect {i + 1}:")

    for class_id, probability in zip(
        rf_model.classes_, row
    ):
        print(f"Class {class_id}: {probability * 100:.2f}%")
import os

print("=== CHECKING AVAILABLE DATASETS ===")

for root, dirs, files in os.walk("/content"):
    dirs[:] = [
        d for d in dirs
        if d not in ["node_modules", ".cache"]
    ]

    for file in files:
        if file.lower().endswith(
            (".csv", ".xlsx", ".json", ".yaml", ".yml")
        ):
            print(os.path.join(root, file))
import pandas as pd
import os

files = [
    "/content/pcb_defect_report.csv",
    "/content/final_pcb_defect_report.csv"
]

for file in files:
    if os.path.exists(file):
        print("\n" + "=" * 60)
        print("FILE:", file)

        df = pd.read_csv(file)

        print("\nShape:", df.shape)
        print("\nColumn names:")
        print(df.columns.tolist())

        print("\nFirst 5 rows:")
        display(df.head())
import joblib
import os

rf_path = "/content/pcb_random_forest.pkl"
features_path = "/content/pcb_feature_columns.pkl"

print("=== RANDOM FOREST MODEL CHECK ===")

print("Model file exists:", os.path.exists(rf_path))
print("Feature file exists:", os.path.exists(features_path))

if os.path.exists(rf_path):
    rf_model = joblib.load(rf_path)

    print("\nModel type:", type(rf_model).__name__)
    print("Number of trees:", rf_model.n_estimators)
    print("Classes:", rf_model.classes_)
    print("Number of features:", rf_model.n_features_in_)

    if hasattr(rf_model, "feature_names_in_"):
        print("Training feature names:",
              list(rf_model.feature_names_in_))

    print("\nClass distribution in model predictions:")

    print("Class IDs:", rf_model.classes_)

    print("Feature importance:")
    if hasattr(rf_model, "feature_importances_"):
        print(rf_model.feature_importances_)

if os.path.exists(features_path):
    feature_columns = joblib.load(features_path)
    print("\nSaved feature columns:")
    print(feature_columns)
print("=== RANDOM FOREST INPUT CHECK ===")

print("\nInput features:")
display(features_df)

print("\nFeature data types:")
print(features_df.dtypes)

print("\nMissing values:")
print(features_df.isnull().sum())

print("\nFeature ranges:")
print(features_df.describe())

print("\nRandom Forest predictions:")
print(rf_model.predict(features_df[feature_columns]))

print("\nPrediction probabilities:")
print(rf_model.predict_proba(features_df[feature_columns]))
print("=== FEATURE VALUES CHECK ===")

print("Feature columns:")
print(feature_columns)

print("\nInput features:")
display(features_df[feature_columns])

print("\nFeature ranges:")
display(features_df[feature_columns].describe().round(4))

print("\nMissing values:")
print(features_df[feature_columns].isnull().sum())

print("\nYOLO detected defects:")
print(combined_report["YOLO_Defect"].tolist())

print("\nRandom Forest predictions:")
print(rf_model.predict(features_df[feature_columns]))
import os

print("=== CHECKING RANDOM FOREST TRAINING FILES ===")

for root, dirs, files in os.walk("/content"):
    dirs[:] = [
        d for d in dirs
        if d not in ["node_modules", ".cache"]
    ]

    for file in files:
        if file.lower().endswith(
            (".csv", ".pkl", ".joblib", ".xlsx", ".parquet")
        ):
            print(os.path.join(root, file))
import joblib
import numpy as np

rf_model = joblib.load("/content/pcb_random_forest.pkl")

print("=== RANDOM FOREST TRAINING CHECK ===")

print("\nClasses learned by the model:")
print(rf_model.classes_)

print("\nNumber of trees:")
print(rf_model.n_estimators)

print("\nExpected number of features:")
print(rf_model.n_features_in_)

print("\nTraining feature names:")
print(list(rf_model.feature_names_in_))

print("\nClass distribution in training trees:")

for class_id in rf_model.classes_:
    counts = []

    for tree in rf_model.estimators_:
        counts.append(
            tree.tree_.value[0, 0, list(rf_model.classes_).index(class_id)]
        )

    print(f"Class {class_id}: checked")
import joblib

rf_model = joblib.load("/content/pcb_random_forest.pkl")

print("=== RANDOM FOREST CLASS INFORMATION ===")

print("\nModel classes:")
print(rf_model.classes_)

print("\nClass weights:")
print(rf_model.class_weight)

print("\nFeature importance ranking:")

feature_names = list(rf_model.feature_names_in_)
importances = rf_model.feature_importances_

for name, importance in sorted(
    zip(feature_names, importances),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{name}: {importance:.4f}")

print("\nModel parameters:")
print(rf_model.get_params())
import os
import pandas as pd
import joblib

print("=== CHECKING RANDOM FOREST TRAINING DATA ===")

# Load the saved model
rf_model = joblib.load("/content/pcb_random_forest.pkl")

# Check available CSV files
csv_files = [
    "/content/pcb_defect_report.csv",
    "/content/final_pcb_defect_report.csv"
]

for file_path in csv_files:
    if os.path.exists(file_path):
        print(f"\nFile found: {file_path}")

        df = pd.read_csv(file_path)

        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.head())

        print("\nData types:")
        print(df.dtypes)

        print("\nMissing values:")
        print(df.isnull().sum())

print("\n=== RANDOM FOREST MODEL CLASSES ===")
print(rf_model.classes_)

print("\n=== MODEL FEATURE NAMES ===")
print(rf_model.feature_names_in_)

print("\n=== MODEL CLASS DISTRIBUTION ===")
for estimator in rf_model.estimators_[:5]:
    print(estimator.classes_)
import joblib
import os

model_path = "/content/pcb_random_forest.pkl"

rf_model = joblib.load(model_path)

print("=== RANDOM FOREST TRAINING DIAGNOSTICS ===")

print("\n1. Model classes:")
print(rf_model.classes_)

print("\n2. Number of trees:")
print(len(rf_model.estimators_))

print("\n3. Number of input features:")
print(rf_model.n_features_in_)

print("\n4. Feature names:")
print(rf_model.feature_names_in_)

print("\n5. Model training metadata:")

for attribute in [
    "n_outputs_",
    "n_classes_",
    "n_features_in_",
    "feature_names_in_",
    "n_samples_seen_"
]:
    if hasattr(rf_model, attribute):
        print(f"{attribute}: {getattr(rf_model, attribute)}")

print("\n6. Available PCB CSV files:")

for filename in os.listdir("/content"):
    if "pcb" in filename.lower() and filename.lower().endswith(".csv"):
        print(filename)
from google.colab import drive
drive.mount('/content/drive')

import os
import shutil

# Create a permanent project folder in Google Drive
backup_folder = "/content/drive/MyDrive/PCB_Defect_Project"

os.makedirs(backup_folder, exist_ok=True)

# Files to back up
files_to_save = [
    "pcb_random_forest.pkl",
    "pcb_feature_columns.pkl",
    "pcb_defect_report.csv",
    "final_pcb_defect_report.csv"
]

for filename in files_to_save:
    source = os.path.join("/content", filename)
    destination = os.path.join(backup_folder, filename)

    if os.path.exists(source):
        shutil.copy2(source, destination)
        print("Backed up:", filename)
    else:
        print("File not found:", filename)

print("\nBackup folder:", backup_folder)
import os

backup_folder = "/content/drive/MyDrive/PCB_Defect_Project"

print("=== PCB PROJECT BACKUP CHECK ===")

for filename in os.listdir(backup_folder):
    filepath = os.path.join(backup_folder, filename)

    if os.path.isfile(filepath):
        print(f"✅ {filename} — {os.path.getsize(filepath):,} bytes")

print("\nBackup verification complete!")
import os

path = "/content/DeepPCB"

print("Dataset folder exists:", os.path.exists(path))

if os.path.exists(path):
    print("\nContents:")
    for item in os.listdir(path):
        print(item)
import os

path = "/content/DeepPCB"

print("Dataset folder exists:", os.path.exists(path))

if os.path.exists(path):
    print("\nContents:")
    for item in os.listdir(path):
        print(item)
import os

path = "/content/DeepPCB"

print("Dataset folder exists:", os.path.exists(path))

if os.path.exists(path):
    print("\nContents:")
    for item in os.listdir(path):
        print(item)
import zipfile
import os

zip_path = "/content/DeepPCB_without_Mousebite.zip"
extract_path = "/content/DeepPCB_without_Mousebite"

print("ZIP file exists:", os.path.exists(zip_path))

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    print("Dataset extracted successfully!")
    print("Extracted contents:")

    for item in os.listdir(extract_path)[:20]:
        print(item)
else:
    print("ZIP file not found. Check its location in the Files panel.")
import os

base = "/content/DeepPCB_without_Mousebite"

for root, dirs, files in os.walk(base):
    for file in files:
        if file.endswith((".yaml", ".yml")):
            print(os.path.join(root, file))
import os

base = "/content/DeepPCB_without_Mousebite"

for root, dirs, files in os.walk(base):
    depth = root.replace(base, "").count(os.sep)

    if depth <= 3:
        print("  " * depth + os.path.basename(root) + "/")

        for file in files[:5]:
            print("  " * (depth + 1) + file)
import os

for item in os.listdir("/content"):
    print(item)
import os

dataset_path = "/content/DeepPCB_without_Mousebite"

print("Dataset exists:", os.path.exists(dataset_path))

if os.path.exists(dataset_path):
    print("\nDataset contents:")
    for item in os.listdir(dataset_path):
        print(item)
import os

zip_path = "/content/DeepPCB_without_Mousebite.zip"

print("ZIP exists:", os.path.exists(zip_path))
import os

zip_path = "/content/DeepPCB_without_Mousebite.zip"

print("ZIP file exists:", os.path.exists(zip_path))
import zipfile
import os

zip_path = "/content/DeepPCB_without_Mousebite.zip"
extract_path = "/content/DeepPCB_without_Mousebite"

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_path)

print("Dataset extracted successfully!")
print("Folders:", os.listdir(extract_path)[:15])
import os

base = "/content/DeepPCB_without_Mousebite"

for root, dirs, files in os.walk(base):
    depth = root.replace(base, "").count(os.sep)

    if depth <= 2:
        print(root)
import yaml
import os

yaml_path = "/content/DeepPCB_without_Mousebite/data.yaml"

with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

print("Number of classes:", data.get("nc"))
print("Class names:", data.get("names"))
from collections import Counter
import os

base = "/content/DeepPCB_without_Mousebite"
splits = ["train", "valid", "test"]

for split in splits:
    label_dir = os.path.join(base, split, "labels")
    counts = Counter()
    invalid = []

    if not os.path.isdir(label_dir):
        print(f"{split}: labels folder missing")
        continue

    for filename in os.listdir(label_dir):
        if not filename.endswith(".txt"):
            continue

        with open(os.path.join(label_dir, filename), "r") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue

                class_id = int(parts[0])
                counts[class_id] += 1

                if class_id < 0 or class_id >= int(data["nc"]):
                    invalid.append((filename, class_id))

    print(f"\n{split} class counts:", dict(counts))
    print(f"Invalid labels: {len(invalid)}")
    if invalid:
        print("Examples:", invalid[:5])
import yaml

yaml_path = "/content/DeepPCB_without_Mousebite/data.yaml"

with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

print("Number of classes:", data["nc"])
print("Class names:", data["names"])
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

# Dataset location
BASE = "/content/DeepPCB_without_Mousebite"

# Read YOLO labels and extract geometric features
def extract_features(split):
    rows = []
    label_dir = os.path.join(BASE, split, "labels")

    for filename in os.listdir(label_dir):
        if not filename.endswith(".txt"):
            continue

        with open(os.path.join(label_dir, filename)) as f:
            for line in f:
                values = line.strip().split()

                if len(values) != 5:
                    continue

                cls, xc, yc, w, h = map(float, values)

                area = w * h
                perimeter = 2 * (w + h)
                aspect_ratio = w / (h + 1e-8)

                rows.append([
                    area, perimeter, aspect_ratio, int(cls)
                ])

    return pd.DataFrame(
        rows,
        columns=["area", "perimeter", "aspect_ratio", "class_id"]
    )

train_df = extract_features("train")
valid_df = extract_features("valid")
test_df = extract_features("test")

print("Training samples:", len(train_df))
print("Validation samples:", len(valid_df))
print("Testing samples:", len(test_df))

print("\nTraining class counts:")
print(train_df["class_id"].value_counts().sort_index())
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Select input features
features = ["area", "perimeter", "aspect_ratio"]

X_train = train_df[features]
y_train = train_df["class_id"]

X_test = test_df[features]
y_test = test_df["class_id"]

# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# Train
print("Training Random Forest...")
rf_model.fit(X_train, y_train)
print("Training completed!")

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluate
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Random Forest Confusion Matrix")
plt.show()

# Save model
save_path = "/content/drive/MyDrive/PCB_Project/random_forest.pkl"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

joblib.dump(rf_model, save_path)

print("\nModel saved at:", save_path)
import os
from ultralytics import YOLO

# Load your trained YOLO model
yolo_path = "/content/drive/MyDrive/PCB_Project/best.pt"

print("YOLO model exists:", os.path.exists(yolo_path))

# Check the dataset
dataset_path = "/content/DeepPCB_without_Mousebite"

for split in ["train", "valid", "test"]:
    image_path = os.path.join(dataset_path, split, "images")
    label_path = os.path.join(dataset_path, split, "labels")

    print(f"\n{split.upper()}")
    print("Images folder exists:", os.path.isdir(image_path))
    print("Labels folder exists:", os.path.isdir(label_path))

    if os.path.isdir(image_path):
        print("Number of images:", len(os.listdir(image_path)))
%pip install ultralytics
from ultralytics import YOLO

print("Ultralytics imported successfully!")
import os
from ultralytics import YOLO

yolo_path = "/content/drive/MyDrive/PCB_Project/best.pt"

print("Model file exists:", os.path.exists(yolo_path))

if os.path.exists(yolo_path):
    yolo_model = YOLO(yolo_path)
    print("YOLO model loaded successfully!")
    print("YOLO classes:", yolo_model.names)
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
from google.colab import drive
drive.mount('/content/drive')
import os

print("Drive folder exists:", os.path.exists("/content/drive"))
print("Drive folder contents:", os.listdir("/content/drive"))
import os

os.makedirs("/content/gdrive", exist_ok=True)

print("New mount folder ready:", os.path.isdir("/content/gdrive"))
print("Folder contents:", os.listdir("/content/gdrive"))
from google.colab import drive

drive.mount("/content/gdrive")
import os

print(os.listdir("/content/gdrive"))
import os

search_path = "/content/gdrive/MyDrive"

print("Searching for YOLO model...")

found = False

for root, dirs, files in os.walk(search_path):
    if "best.pt" in files:
        print("Model found:", os.path.join(root, "best.pt"))
        found = True

if not found:
    print("No best.pt file found in Google Drive.")
import os

print("Searching Colab runtime for best.pt...")

for root, dirs, files in os.walk("/content"):
    if "best.pt" in files:
        print("FOUND:", os.path.join(root, "best.pt"))
import os

for folder in [
    "/content/DeepPCB_without_Mousebite",
    "/content/drive/MyDrive/PCB_Project",
    "/content/gdrive/MyDrive/PCB_Project"
]:
    print("\nFolder:", folder)
    print("Exists:", os.path.exists(folder))

    if os.path.exists(folder):
        print("Contents:", os.listdir(folder)[:15])
import os

print("Searching for YOLO model files...")

for root, dirs, files in os.walk("/content/drive/MyDrive"):
    for file in files:
        if file.endswith((".pt", ".pth")):
            print(os.path.join(root, file))
import os

search_path = "/content"

print("Searching for best.pt...")

for root, dirs, files in os.walk(search_path):
    if "best.pt" in files:
        print("FOUND:", os.path.join(root, "best.pt"))
import os

folder = "/content/drive/MyDrive/PCB_Project"

print("Folder exists:", os.path.exists(folder))

if os.path.exists(folder):
    print("Files:", os.listdir(folder))
import os

search_locations = [
    "/content/runs",
    "/content/drive/MyDrive"
]

for location in search_locations:
    print("\nChecking:", location)

    if os.path.exists(location):
        for root, dirs, files in os.walk(location):
            if "best.pt" in files:
                print("YOLO MODEL FOUND:", os.path.join(root, "best.pt"))
import os

folder = "/content/drive/MyDrive"

for root, dirs, files in os.walk(folder):
    for file in files:
        if file.endswith(".pt"):
            print("Model found:", os.path.join(root, file))
import os

dataset_path = "/content/DeepPCB_without_Mousebite"

print("Dataset exists:", os.path.exists(dataset_path))

if os.path.exists(dataset_path):
    print("Dataset folders:", os.listdir(dataset_path))

print("\nChecking YOLO training folders...")

runs_path = "/content/runs"

if os.path.exists(runs_path):
    print(os.listdir(runs_path))
else:
    print("No YOLO training folder found.")
import yaml

yaml_path = "/content/DeepPCB_without_Mousebite/data.yaml"

with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

print("Number of classes:", data["nc"])
print("Class names:", data["names"])
print("Dataset configuration:")
print(data)
%pip install ultralytics
%pip install ultralytics
from ultralytics import YOLO
import os

dataset_yaml = "/content/DeepPCB_without_Mousebite/data.yaml"

# Start with pretrained YOLO weights
model = YOLO("yolov8n.pt")

# Train on your PCB dataset
results = model.train(
    data=dataset_yaml,
    epochs=50,
    imgsz=640,
    batch=16,
    project="/content/runs/detect",
    name="pcb_training",
    exist_ok=True
)

# Check whether the best model was saved
best_model = "/content/runs/detect/pcb_training/weights/best.pt"

print("Best model exists:", os.path.exists(best_model))
from ultralytics import YOLO
import os

dataset_yaml = "/content/DeepPCB_without_Mousebite/data.yaml"

# Start with pretrained YOLO weights
model = YOLO("yolov8n.pt")

# Train on your PCB dataset
results = model.train(
    data=dataset_yaml,
    epochs=50,
    imgsz=640,
    batch=16,
    project="/content/runs/detect",
    name="pcb_training",
    exist_ok=True
)

# Check whether the best model was saved
best_model = "/content/runs/detect/pcb_training/weights/best.pt"

print("Best model exists:", os.path.exists(best_model))
import os

base = "/content/DeepPCB_without_Mousebite"

print("Dataset folder exists:", os.path.exists(base))

if os.path.exists(base):
    for root, dirs, files in os.walk(base):
        for file in files:
            if file.lower().endswith((".yaml", ".yml")):
                print("YAML FILE FOUND:", os.path.join(root, file))
import os

print("Folders inside /content:")
print(os.listdir("/content"))

print("\nChecking Google Drive:")
if os.path.exists("/content/drive/MyDrive"):
    print(os.listdir("/content/drive/MyDrive"))
else:
    print("Google Drive is not mounted.")
import zipfile
import os

zip_path = "/content/DeepPCB_without_Mousebite.zip"
extract_path = "/content"

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    print("Dataset extracted successfully!")
else:
    print("ZIP file not found!")

print("\nFolders inside /content:")
print(os.listdir("/content"))
import os
import zipfile

zip_path = "/content/DeepPCB_without_Mousebite.zip"

print("File exists:", os.path.exists(zip_path))

if os.path.exists(zip_path):
    print("File size:", os.path.getsize(zip_path), "bytes")
    print("Is valid ZIP:", zipfile.is_zipfile(zip_path))

    with open(zip_path, "rb") as f:
        print("First 20 bytes:", f.read(20))
import os
import zipfile
import subprocess

zip_path = "/content/DeepPCB_without_Mousebite.zip"

print("Checking archive...")

result = subprocess.run(
    ["file", zip_path],
    capture_output=True,
    text=True
)

print(result.stdout)

print("\nChecking ZIP structure...")

with open(zip_path, "rb") as f:
    f.seek(-22, 2)
    print("Last 22 bytes:", f.read(22))
import os
import zipfile

zip_path = "/content/DeepPCB_without_Mousebite.zip"
extract_path = "/content/DeepPCB_recovered"

os.makedirs(extract_path, exist_ok=True)

try:
    with zipfile.ZipFile(zip_path, "r") as z:
        print("Archive opened successfully!")
        print("Number of files:", len(z.namelist()))
        print("First 10 files:", z.namelist()[:10])

        z.extractall(extract_path)
        print("\nExtraction successful!")

except Exception as e:
    print("Normal extraction failed:", repr(e))

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            bad_file = z.testzip()
            print("First corrupted file:", bad_file)
    except Exception as e2:
        print("Archive validation failed:", repr(e2))
import os

print("Searching for dataset folders...")

for base in ["/content", "/content/drive", "/content/gdrive"]:
    if os.path.exists(base):
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in [
                ".config", "sample_data", ".cache"
            ]]

            if "data.yaml" in files:
                print("FOUND data.yaml:", os.path.join(root, "data.yaml"))

            if "train" in dirs and "valid" in dirs:
                print("POSSIBLE DATASET:", root)
import os

print("Contents of /content:")
print(os.listdir("/content"))

print("\nChecking dataset folder:")
path = "/content/DeepPCB_without_Mousebite"

print("Dataset exists:", os.path.exists(path))

if os.path.exists(path):
    print("Files:", os.listdir(path))
import os

path = "/content/DeepPCB_recovered"

print("Dataset exists:", os.path.exists(path))

if os.path.exists(path):
    print("Dataset contents:", os.listdir(path))

    for root, dirs, files in os.walk(path):
        if "data.yaml" in files:
            print("FOUND data.yaml:", os.path.join(root, "data.yaml"))
import os

print("Checking dataset and model files...\n")

for folder in ["/content", "/content/runs", "/content/DeepPCB_recovered"]:
    print("\nFolder:", folder)

    if os.path.exists(folder):
        print("Contents:", os.listdir(folder)[:20])
    else:
        print("Folder does not exist")
import os

print("Searching for trained YOLO models...")

for base in ["/content/runs", "/content/drive/MyDrive"]:
    if os.path.exists(base):
        for root, dirs, files in os.walk(base):
            dirs[:] = [
                d for d in dirs