import cv2
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
image = cv2.imread("resources/images/traffic-1.webp")

results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)

class_names = model.model.names  # ← mapping class_id to name

labels = [
    f"{class_names[class_id]} {confidence:.2f}"
    for class_id, confidence in zip(detections.class_id, detections.confidence)
]

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

cv2.imshow("Traffic Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
