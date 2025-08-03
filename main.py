import cv2
import supervision as sv
from ultralytics import YOLO

# Load model and process image
model = YOLO("yolov8n.pt")
image = cv2.imread("resources/images/traffic-1.webp")

# Verify image loading
if image is None:
    print("Error: Could not load image")
    exit()

# Process image
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)

# Print detections to verify
print(f"Detected {len(detections)} objects")

# Annotate image
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

# Save image to verify annotations
cv2.imwrite("output.jpg", annotated_image)
print("Saved annotated image to output.jpg")

# Try displaying with a larger window
cv2.namedWindow("Traffic Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Traffic Detection", 800, 600)
cv2.imshow("Traffic Detection", annotated_image)

# Wait longer and verify window opens
key = cv2.waitKey(10000)  # Wait for 10 seconds
if key == ord('q'):
    cv2.destroyAllWindows()