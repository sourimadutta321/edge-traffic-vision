import cv2

def draw_hud(frame, inflow, outflow, density, line_y):
    h, w = frame.shape[:2]
    cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 2)
    cv2.putText(frame, "COUNTING LINE", (20, line_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"Southbound: {inflow}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Northbound: {outflow}", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Occupancy: {density * 100:.1f}%", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame
