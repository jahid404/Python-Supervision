import cv2
import numpy as np
import supervision as sv
from inference.models.utils import get_roboflow_model

# Load model and trackers
model = get_roboflow_model(model_id="yolov8n-640", api_key="5mDZf8wehgAFIsivtQkh")
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Set to keep track of seen tracker IDs
seen_tracker_ids = set()

# Total new people counter
new_person_count = 0

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    global new_person_count

    results = model.infer(frame)[0]
    detections = sv.Detections.from_inference(results)
    detections = tracker.update_with_detections(detections)

    # Check for new tracker IDs
    current_ids = set(detections.tracker_id)
    new_ids = current_ids - seen_tracker_ids
    new_person_count += len(new_ids)
    seen_tracker_ids.update(current_ids)

    # Labels for display
    labels = [
        f"#{tracker_id} {class_name}"
        for class_name, tracker_id
        in zip(detections.data["class_name"], detections.tracker_id)
    ]

    # Annotate boxes and labels
    annotated_frame = box_annotator.annotate(
        frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels)

    # 🧠 Draw new person count
    cv2.putText(
        annotated_frame,
        f"New Persons: {new_person_count}",
        org=(10, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA
    )

    return annotated_frame

# Process and save the video
sv.process_video(
    source_path="../../videos/crowd-1.mp4",
    target_path="../../videos/results/crowd-1-tracker-total-detect.mp4",
    callback=callback
)
