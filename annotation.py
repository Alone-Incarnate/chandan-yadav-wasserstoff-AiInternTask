import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# Function to draw bounding boxes and labels on the original image
def annotate_image(original_image, detections):
    annotated_image = original_image.copy()  # Create a copy to annotate
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = map(int, detection[:6])  # Get bounding box coordinates and class index
        class_name = yolo_model.names[class_id]  # Get class name

        # Draw the bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # Put the label above the bounding box
        cv2.putText(annotated_image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return annotated_image

# Load an image
image_path = "2.jpg"
image = cv2.imread(image_path)

# Perform object detection
results = yolo_model(image)

# Iterate over results and annotate the image
for result in results:
    # Check if the result has a 'boxes' attribute and proceed accordingly
    if hasattr(result, 'boxes'):
        detections = result.boxes.data.cpu().numpy()  # Adjust based on actual structure
        # Annotate the image
        annotated_image = annotate_image(image, detections)

# Create a named window before resizing
window_name = "Annotated Image"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Create a resizable window

# Resize the window
window_width, window_height = 800, 600  # Set desired window size
cv2.resizeWindow(window_name, window_width, window_height)

# Display the annotated image
cv2.imshow(window_name, annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
