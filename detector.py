import cv2
from ultralytics import YOLO


class HumanDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.model.conf = 0.3  # lower threshold

    def detect_person(self, frame, roi):
        x1, y1, x2, y2 = roi
        roi_frame = frame[y1:y2, x1:x2]

        results = self.model(roi_frame, verbose=False)[0]

        best_box = None
        max_conf = 0

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == 0 and conf > max_conf:  # person class
                max_conf = conf
                best_box = box.xyxy[0]

        if best_box is not None:
            bx1, by1, bx2, by2 = map(int, best_box)

            # Adjust coordinates to full frame
            bx1 += x1
            bx2 += x1
            by1 += y1
            by2 += y1

            return True, (bx1, by1, bx2, by2)

        return False, None