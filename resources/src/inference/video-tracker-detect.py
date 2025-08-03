import numpy as np
import supervision as sv
from inference.models.utils import get_roboflow_model

model = get_roboflow_model(model_id="yolov8n-640", api_key="5mDZf8wehgAFIsivtQkh")
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model.infer(frame)[0]
    detections = sv.Detections.from_inference(results)
    detections = tracker.update_with_detections(detections)

    labels = [
        f"#{tracker_id} {class_name}"
        for class_name, tracker_id
        in zip(detections.data["class_name"], detections.tracker_id)
    ]

    annotated_frame = box_annotator.annotate(
        frame.copy(), detections=detections)
    return label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels)

sv.process_video(
    source_path="../../videos/traffic-2.mp4",
    target_path="../../videos/results/traffic-2-tracker-detect.mp4",
    callback=callback
)