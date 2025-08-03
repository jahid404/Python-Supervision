import cv2
import supervision as sv
from ultralytics import YOLO

model = get_model(model_id="yolov8n-640")
image = cv2.imread("resources/images/traffic-1.webp")
results = model.infer(image)[0]
detections = sv.Detections.from_inference(results)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(detections['class_name'], detections.confidence)
]

annotated_image = box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels)

cv2.imshow("Traffic Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
