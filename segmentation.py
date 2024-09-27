# Import necessary libraries
import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
from PIL import Image

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")  # Ensure this model is downloaded

# Load Segformer model and processor for segmentation
model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

# Directory to save segmented images
output_dir = "segmented_objects"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to perform instance segmentation on each object
def segment_object(object_image, object_id, master_id):
    # Convert to PIL and prepare for segmentation
    pil_image = Image.fromarray(cv2.cvtColor(object_image, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil_image, return_tensors="pt")

    # Perform segmentation
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the segmentation mask
    mask = outputs.logits.argmax(dim=1).squeeze().cpu().numpy()

    if mask.size == 0:
        print(f"Segmentation mask is empty for object ID: {object_id}")
        return None

    # Resize the mask to match the object's size
    binary_mask = cv2.resize(mask, (object_image.shape[1], object_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Convert the mask to a proper 8-bit format
    binary_mask = (binary_mask > 0).astype(np.uint8) * 255

    # Ensure the mask is 2D and fits the image
    if len(binary_mask.shape) == 2:
        binary_mask = binary_mask[:, :, np.newaxis]

    # Apply the mask to the object image
    segmented_image = cv2.bitwise_and(object_image, object_image, mask=binary_mask)

    # Save the segmented object with mask overlay as a new image
    output_path = os.path.join(output_dir, f"segmented_object_{master_id}_{object_id}.jpg")
    cv2.imwrite(output_path, segmented_image)
    print(f"Saved segmented object {object_id} for master image {master_id} at {output_path}")

    return output_path

# Process the input image for both YOLO object detection and Segmentation
def process_image(image_path):
    # Read the input image
    image = cv2.imread(image_path)
    master_id = os.path.splitext(os.path.basename(image_path))[0]  # Use image name as master ID

    # Perform YOLOv8 object detection
    results = yolo_model(image)
    detections = results[0].boxes

    # Loop through each detection
    for idx, detection in enumerate(detections):
        x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Bounding box coordinates

        # Crop detected object
        cropped_object = image[y1:y2, x1:x2]

        # Perform instance segmentation on the cropped object
        segment_object(cropped_object, object_id=idx + 1, master_id=master_id)

# Run the process to segment the objects and save them
process_image("2.jpg")
