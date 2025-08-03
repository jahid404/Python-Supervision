import cv2
import supervision as sv
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Load image
image = cv2.imread("resources/images/traffic-1.webp")

# Run inference
results = model(image)[0]

# Convert to Supervision Detections
detections = sv.Detections.from_ultralytics(results)

# Annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Generate labels
labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence in zip(
        detections.class_name, detections.confidence
    )
]

# Annotate
annotated_image = box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels)

# Show the result
cv2.imshow("Traffic Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
