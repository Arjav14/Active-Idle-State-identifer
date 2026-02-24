import numpy as np
from sort import Sort

class OperatorTracker:
    def __init__(self):
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

    def update(self, detections):
        if len(detections) == 0:
            return []

        det_array = np.array(detections)
        tracks = self.tracker.update(det_array)
        return tracks