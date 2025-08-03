import cv2
import supervision as sv
from ultralytics import YOLO

# Load model and process image
model = YOLO("yolov8n.pt")
image = cv2.imread("resources/images/traffic-1.webp")
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

# Save image to verify annotations
cv2.imwrite("output.jpg", annotated_image)
print("Saved annotated image to output.jpg")