import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    return wh / (
        (bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) +
        (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh
    )


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)
        self.kf.H = np.zeros((4,7))
        self.kf.H[:4,:4] = np.eye(4)
        self.kf.R *= 10.
        self.kf.P *= 1000.
        self.kf.Q *= 0.01

        self.kf.x[:4] = np.reshape(bbox[:4], (4,1))
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.kf.update(np.reshape(bbox[:4], (4,1)))

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.kf.x[:4].reshape((4,))

    def get_state(self):
        return self.kf.x[:4].reshape((4,))


class Sort:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, detections):
        if len(self.trackers) == 0:
            for det in detections:
                self.trackers.append(KalmanBoxTracker(det))
            return np.array([])

        trks = np.zeros((len(self.trackers), 4))
        for t, trk in enumerate(self.trackers):
            trks[t] = trk.predict()

        iou_matrix = np.zeros((len(detections), len(trks)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(trks):
                iou_matrix[d, t] = iou(det, trk)

        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.asarray(matched_indices).T

        unmatched_dets = []
        for d in range(len(detections)):
            if d not in matched_indices[:,0]:
                unmatched_dets.append(d)

        unmatched_trks = []
        for t in range(len(trks)):
            if t not in matched_indices[:,1]:
                unmatched_trks.append(t)

        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_dets.append(m[0])
                unmatched_trks.append(m[1])
            else:
                self.trackers[m[1]].update(detections[m[0]])

        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[i]))

        ret = []
        for trk in self.trackers:
            if trk.time_since_update < 1:
                ret.append(np.concatenate((trk.get_state(), [trk.id])))

        self.trackers = [
            t for t in self.trackers if t.time_since_update <= self.max_age
        ]

        return np.array(ret)