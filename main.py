import argparse
import os
import cv2
from tqdm import tqdm
from src.detector import TrafficDetector
from src.analytics import TrafficAnalytics
from src.utils import draw_hud

def parse_args():
    parser = argparse.ArgumentParser(description="Traffic Vision Analytics Pipeline")
    parser.add_argument("--source", type=str, default="data/sample_traffic.mp4", help="Input video path")
    parser.add_argument("--output", type=str, default="output/processed_traffic.mp4", help="Output video path")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--line_y", type=int, default=420, help="Y coordinate for tripwire counting")
    return parser.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(args.source):
        raise FileNotFoundError(f"Source video not found: {args.source}")
        
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    cap = cv2.VideoCapture(args.source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    detector = TrafficDetector(model_weights=args.model)
    analytics = TrafficAnalytics(line_y=args.line_y)

    print(f"[INFO] Processing {args.source} -> {args.output} ({total_frames} frames)...")
    pbar = tqdm(total=total_frames)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.process_frame(frame)
        active_boxes = []

        for d in detections:
            box = d["box"]
            track_id = d["id"]
            cls_name = d["class"]
            active_boxes.append(box)

            cy = int((box[1] + box[3]) / 2)
            analytics.update_flow(track_id, cls_name, cy)

            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 100, 0), 2)
            cv2.putText(frame, f"#{track_id} {cls_name}", (int(box[0]), int(box[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        density = analytics.compute_density(active_boxes, frame.shape)
        annotated_frame = draw_hud(frame, analytics.inflow_count, analytics.outflow_count, density, args.line_y)
        
        out.write(annotated_frame)
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()

    print("\n--- Processing Summary ---")
    print(f"Total Inflow Count  : {analytics.inflow_count}")
    print(f"Total Outflow Count : {analytics.outflow_count}")
    print(f"Vehicle Breakdown   : {analytics.class_counts}")
    print(f"Output saved to     : {args.output}")

if __name__ == "__main__":
    main()
