import numpy as np
import supervision as sv
from inference.models.utils import get_roboflow_model

model = get_roboflow_model(model_id="yolov8n-640", api_key="5mDZf8wehgAFIsivtQkh")
box_annotator = sv.BoxAnnotator()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model.infer(frame)[0]
    detections = sv.Detections.from_inference(results)
    return box_annotator.annotate(frame.copy(), detections=detections)

sv.process_video(
    source_path="../../videos/crowd-1.mp4",
    target_path="../../videos/results/crowd-1-box-detect.mp4",
    callback=callback
)