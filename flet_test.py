import flet as ft
from ultralytics import YOLO
import os
from PIL import Image
import io
import base64

# Define image paths
image_path = "phone.png"
results_folder = "results"
result_image_path = os.path.join(results_folder, "phone.png")

# Ensure input image exists
if not os.path.exists(image_path):
    print(f"Error: Image '{image_path}' not found")
    exit(1)

# Ensure results folder exists
os.makedirs(results_folder, exist_ok=True)

# Load YOLO model and run detection
model = YOLO("model.tflite", task="detect")
results = model(image_path, imgsz=1280)

# Manually save the result to the exact location we want
for i, r in enumerate(results):
    r.save(filename=result_image_path)

# Extract detection details
detections_text = "\n".join([
    f"Detections: {len(r.boxes)} objects found\n"
    f"Classes: {r.boxes.cls.tolist()}\n"
    f"Confidence scores: {r.boxes.conf.tolist()}\n"
    for r in results
])

# Function to convert image to base64 for Flet display
def image_to_base64(img_path):
    if not os.path.exists(img_path):
        print(f"Warning: Image not found at '{img_path}'")
        return None
    
    with open(img_path, "rb") as img_file:
        img_data = img_file.read()
        base64_data = base64.b64encode(img_data).decode('utf-8')
        return base64_data

# Flet GUI
def main(page: ft.Page):
    page.title = "YOLO Object Detection"
    page.padding = 20
    page.theme_mode = ft.ThemeMode.LIGHT
    page.scroll = "auto"
    
    # Convert images to base64
    original_img_base64 = image_to_base64(image_path)
    result_img_base64 = image_to_base64(result_image_path)
    
    # Get label names if available
    label_names = {}
    labels_path = "smartphone_labels.txt"
    if os.path.exists(labels_path):
        with open(labels_path, "r") as f:
            for i, line in enumerate(f):
                label_names[i] = line.strip()
    
    # Format detection text with class names if available
    formatted_detections = []
    for r in results:
        formatted_detections.append(f"Detections: {len(r.boxes)} objects found")
        
        classes = r.boxes.cls.tolist()
        class_names = []
        for cls in classes:
            cls_idx = int(cls)
            if cls_idx in label_names:
                class_names.append(f"{cls_idx} ({label_names[cls_idx]})")
            else:
                class_names.append(str(cls_idx))
        
        formatted_detections.append(f"Classes: {class_names}")
        formatted_detections.append(f"Confidence scores: {[round(c, 2) for c in r.boxes.conf.tolist()]}")
    
    formatted_text = "\n".join(formatted_detections)

    # Create UI elements
    page.add(
        ft.Text("YOLO Object Detection", size=24, weight=ft.FontWeight.BOLD),
        ft.Divider(),
        ft.Row(
            controls=[
                ft.Container(
                    content=ft.Image(
                        src_base64=original_img_base64,
                        width=400,  # Fixed width for the image
                        height=300,  # Fixed height for the image
                        fit=ft.ImageFit.CONTAIN,
                        border_radius=10,
                    ) if original_img_base64 else ft.Text("Original image not found."),
                    padding=10,
                    border=ft.border.all(1, ft.colors.GREY_400),
                    border_radius=10,
                    margin=ft.margin.symmetric(vertical=10)
                ),
                ft.Container(
                    content=ft.Image(
                        src_base64=result_img_base64,
                        width=400,  # Fixed width for the image
                        height=300,  # Fixed height for the image
                        fit=ft.ImageFit.CONTAIN,
                        border_radius=10,
                    ) if result_img_base64 else ft.Text("Result image not found."),
                    padding=10,
                    border=ft.border.all(1, ft.colors.GREY_400),
                    border_radius=10,
                    margin=ft.margin.symmetric(vertical=10)
                ),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        ft.Text("Detection Results", size=16, weight=ft.FontWeight.BOLD),
        ft.Container(
            content=ft.Text(formatted_text),
            width=page.width,
            padding=10,
            border=ft.border.all(1, ft.colors.GREY_400),
            border_radius=10,
            margin=ft.margin.symmetric(vertical=10)
        ),
    )

# Use FLET_APP view for better mobile compatibility
ft.app(target=main, view=ft.AppView.FLET_APP)