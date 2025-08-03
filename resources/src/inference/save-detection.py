import supervision as sv
from inference import get_model

model = get_model(model_id="yolov8n-640")
frames_generator = sv.get_video_frames_generator("../../videos/traffic-1.mp4")

with sv.CSVSink("../../videos/results/traffic.csv") as sink:
    for frame in frames_generator:
        results = model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)
        sink.append(detections)
