import cv2
import time
import yaml
import logging
import numpy as np

from detector import HumanDetector
from db_logger import SystemLogger
from alert import AlertSystem


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    logging.basicConfig(level=logging.INFO)

    cap = cv2.VideoCapture(config["camera"]["source"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])

    if not cap.isOpened():
        logging.error("Camera failed.")
        return

    detector = HumanDetector()
    logger = SystemLogger(
        config["logging"]["csv_path"],
        config["logging"]["db_path"]
    )
    alert = AlertSystem(config["alerts"]["email_enabled"])

    roi = (
        config["roi"]["x1"],
        config["roi"]["y1"],
        config["roi"]["x2"],
        config["roi"]["y2"]
    )

    absence_threshold = 10
    idle_threshold = 15
    sleep_threshold = 30

    last_seen_time = time.time()
    last_motion_time = time.time()
    normal_height = None

    prev_gray = None

    # Detection buffer to avoid flicker
    missed_frames = 0
    max_missed_frames = 8

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        detected, box = detector.detect_person(frame, roi)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -------------------------
        # Detection Stabilization
        # -------------------------
        if detected:
            missed_frames = 0
            last_seen_time = current_time
        else:
            missed_frames += 1

        # If detection missed only few frames, still consider present
        if missed_frames < max_missed_frames:
            is_present = True
        else:
            is_present = False

        state = None

        # =====================================================
        # PERSON PRESENT LOGIC
        # =====================================================
        if is_present and box is not None:

            x1, y1, x2, y2 = box
            height = y2 - y1

            # Smooth baseline height
            if normal_height is None:
                normal_height = height
            else:
                normal_height = 0.95 * normal_height + 0.05 * height

            # Motion detection
            motion = 0
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
                motion = np.sum(thresh) / 255

            # ACTIVE
            if motion > 3000:
                last_motion_time = current_time
                state = "ACTIVE"

            else:
                idle_time = current_time - last_motion_time

                if idle_time > idle_threshold:
                    state = "IDLE"
                else:
                    state = "ACTIVE"

                # Sleeping detection
                height_ratio = height / normal_height if normal_height else 1

                if height_ratio < 0.85 and idle_time > sleep_threshold:
                    state = "SLEEPING"

            # Draw person box
            cv2.rectangle(frame,
                          (x1, y1),
                          (x2, y2),
                          (0, 255, 0),
                          2)

        # =====================================================
        # ABSENCE LOGIC
        # =====================================================
        if state is None:
            if current_time - last_seen_time > absence_threshold:
                state = "ABSENT"
                cv2.putText(frame,
                            "ALERT: OPERATOR NOT PRESENT",
                            (40, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3)
            else:
                state = "ACTIVE"

        prev_gray = gray

        # =====================================================
        # Display State
        # =====================================================
        color = (0, 255, 0)

        if state == "IDLE":
            color = (0, 255, 255)
        elif state in ["SLEEPING", "ABSENT"]:
            color = (0, 0, 255)

        cv2.putText(frame,
                    f"STATE: {state}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    3)

        # Draw ROI
        cv2.rectangle(frame,
                      (roi[0], roi[1]),
                      (roi[2], roi[3]),
                      (255, 0, 0),
                      2)

        cv2.imshow("Operator Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()