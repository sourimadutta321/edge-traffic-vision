import cv2
from ultralytics import YOLO

class TrafficDetector:
    def __init__(self, model_weights="yolov8n.pt", target_classes=None):
        self.model = YOLO(model_weights)
        self.target_classes = target_classes if target_classes else [2, 3, 5, 7]

    def process_frame(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            classes=self.target_classes,
            verbose=False
        )
        detections = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            clss = results[0].boxes.cls.int().cpu().numpy()
            names = results[0].names

            for box, track_id, cls_id in zip(boxes, track_ids, clss):
                detections.append({
                    "box": box,
                    "id": track_id,
                    "class": names[cls_id]
                })
        return detections
