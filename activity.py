import time
import cv2
import numpy as np


class ActivityMonitor:
    def __init__(self, idle_minutes, break_minutes, absence_seconds):
        self.idle_threshold = idle_minutes * 60
        self.break_threshold = break_minutes * 60
        self.absence_threshold = absence_seconds

        self.last_motion_time = time.time()
        self.last_seen_time = time.time()

        self.operator_present = False
        self.absence_alert_triggered = False

    def update_presence(self, detected):
        current_time = time.time()

        if detected:
            self.operator_present = True
            self.last_seen_time = current_time
            self.absence_alert_triggered = False
        else:
            self.operator_present = False

    def check_absence(self):
        if not self.operator_present:
            elapsed = time.time() - self.last_seen_time

            if elapsed > self.absence_threshold:
                if not self.absence_alert_triggered:
                    self.absence_alert_triggered = True
                    return True

        return False

    def check_break(self):
        if not self.operator_present:
            if time.time() - self.last_seen_time > self.break_threshold:
                return True
        return False

    def detect_motion(self, prev_gray, gray):
        diff = cv2.absdiff(prev_gray, gray)
        motion = np.sum(diff)

        if motion > 5000:
            self.last_motion_time = time.time()

        if time.time() - self.last_motion_time > self.idle_threshold:
            return True

        return False