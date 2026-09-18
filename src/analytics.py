import numpy as np

class TrafficAnalytics:
    def __init__(self, line_y=260):
        self.line_y = line_y
        self.tracked_positions = {}
        self.inflow_count = 0
        self.outflow_count = 0
        self.class_counts = {}

    def update_flow(self, track_id, class_name, centroid_y):
        if track_id in self.tracked_positions:
            prev_y = self.tracked_positions[track_id]
            if prev_y < self.line_y <= centroid_y:
                self.inflow_count += 1
                self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
            elif prev_y > self.line_y >= centroid_y:
                self.outflow_count += 1
                self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
        self.tracked_positions[track_id] = centroid_y

    def compute_density(self, boxes, frame_shape):
        frame_area = frame_shape[0] * frame_shape[1]
        if frame_area == 0 or len(boxes) == 0:
            return 0.0
        total_box_area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes)
        return min(1.0, total_box_area / frame_area)
