import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import os
import torch
from ultralytics import YOLO
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor, pipeline
from groq import Groq





# Initialize Groq client
client = Groq(api_key="Please use your groq api key here")  #Please use your Groq api key 

# Load YOLOv8 model for object detection
yolo_model = YOLO("yolov8n.pt")

# Load Segformer model for semantic segmentation
seg_model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
seg_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

# Function to generate descriptions using Groq API
def generate_description(object_name):
    prompt = f"Describe a {object_name} in detail in about 50 words."
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-groq-70b-8192-tool-use-preview",
    )
    
    generated_description = chat_completion.choices[0].message.content
    
    # Trim the description to 50 words
    description_words = generated_description.split()
    if len(description_words) > 50:
        generated_description = ' '.join(description_words[:50]) + '...'
    
    return generated_description

# Function to perform segmentation on each detected object
def segment_object(object_image):
    pil_image = Image.fromarray(cv2.cvtColor(object_image, cv2.COLOR_BGR2RGB))
    inputs = seg_processor(images=pil_image, return_tensors="pt")

    with torch.no_grad():
        outputs = seg_model(**inputs)

    mask = outputs.logits.argmax(dim=1).squeeze().cpu().numpy()
    binary_mask = cv2.resize(mask, (object_image.shape[1], object_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    binary_mask = (binary_mask > 0).astype(np.uint8) * 255

    if len(binary_mask.shape) == 2:
        binary_mask = binary_mask[:, :, np.newaxis]

    segmented_image = cv2.bitwise_and(object_image, object_image, mask=binary_mask)
    return segmented_image

import cv2

# Function to draw bounding boxes and labels on the original image
def annotate_image(original_image, detections):
    annotated_image = original_image.copy()  # Create a copy to annotate
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Get bounding box coordinates
        class_id = int(detection.cls[0])  # Get class index
        class_name = yolo_model.names[class_id]  # Get class name

        # Draw the bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # Put the label above the bounding box
        cv2.putText(annotated_image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return annotated_image



# Function to segment the entire image
def segment_entire_image(image):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = seg_processor(images=pil_image, return_tensors="pt")

    with torch.no_grad():
        outputs = seg_model(**inputs)

    mask = outputs.logits.argmax(dim=1).squeeze().cpu().numpy()
    binary_mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    binary_mask = (binary_mask > 0).astype(np.uint8) * 255

    if len(binary_mask.shape) == 2:
        binary_mask = binary_mask[:, :, np.newaxis]

    segmented_image = cv2.bitwise_and(image, image, mask=binary_mask)
    return segmented_image
def process_image(image_path):
    image = cv2.imread(image_path)
    results = yolo_model(image)
    detections = results[0].boxes

    # Annotate the original image
    annotated_image = annotate_image(image, detections)

    processed_objects = []
    for idx, detection in enumerate(detections):
        x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Get bounding box coordinates
        detected_class_index = int(detection.cls[0])  # Get the class index
        detected_name = yolo_model.names[detected_class_index]  # Get the name of detected object

        cropped_object = image[y1:y2, x1:x2]  # Crop detected object

        # Segment the object
        segmented_image = segment_object(cropped_object)

        # Generate description
        description = generate_description(detected_name)

        # Store the result in a dictionary
        processed_objects.append({
            'ID': idx + 1,
            'Segmented Image': segmented_image,
            'Annotated Image': annotated_image,  # Store the annotated image
            'Detected Name': detected_name,
            'Description': description
        })

    # Return processed objects, full segmented image, and annotated image
    return processed_objects, segment_entire_image(image), annotated_image  # Return three values


def display_results(objects, full_segmented_image, annotated_image):
    # Display the full segmented image
    st.write("Full Segmented Image")
    st.image(full_segmented_image)

    # Display the annotated image
    st.write("Annotated Image")
    st.image(annotated_image)

    # Initialize an empty DataFrame
    df = pd.DataFrame(columns=["ID", "Segmented Image Path", "Detected Name", "Description"])

    # Iterate over each object and add it to the DataFrame
    for obj in objects:
        # Save segmented image as a temporary file to display later
        segmented_image_path = f"segmented_image_{obj['ID']}.png"
        Image.fromarray(obj['Segmented Image']).save(segmented_image_path)

        # Add the object details to the DataFrame
        new_row = pd.DataFrame([{
            "ID": obj['ID'],
            "Segmented Image Path": segmented_image_path,  # Store the image path
            "Detected Name": obj['Detected Name'],
            "Description": obj['Description']
        }])
        df = pd.concat([df, new_row], ignore_index=True)

    # Display the DataFrame in Streamlit
    st.write("Detected Objects")

    # Iterate through the DataFrame to display each row along with the image
    for idx, row in df.iterrows():
        st.write(f"Object ID: {row['ID']}")
        st.write(f"Detected Name: {row['Detected Name']}")
        st.write(f"Description: {row['Description']}")
        st.image(row['Segmented Image Path'])  # Display the segmented image separately

  

# Streamlit Interface
st.title("Object Detection and Annotation System")

# Image uploader in Streamlit
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Convert the uploaded image into an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    cv2_image = cv2.imdecode(file_bytes, 1)

    # Save the uploaded image for further processing
    temp_image_path = "temp_image.jpg"
    cv2.imwrite(temp_image_path, cv2_image)

    # Process the image (detect, segment, and generate descriptions)
    objects, full_segmented_image, annotated_image = process_image(temp_image_path)  # Unpack three values

    # Display the results in a table
    display_results(objects, full_segmented_image, annotated_image)  # Pass annotated_image to display_results

