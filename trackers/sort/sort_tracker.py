# trackers/sort/sort_tracker.py

import numpy as np
from .sort import KalmanBoxTracker, iou
from scipy.optimize import linear_sum_assignment


class SortManager:
    def __init__(self, iou_threshold=0.3):
        self.trackers = []
        self.iou_threshold = iou_threshold

    def update(self, detections):
        """
        detections: list of [x1, y1, x2, y2]
        returns: list of [x1, y1, x2, y2, track_id]
        """

        predicted_boxes = []
        for tracker in self.trackers:
            predicted_boxes.append(tracker.predict()[0])

        matches, unmatched_dets, unmatched_trks = self._associate(
            detections, predicted_boxes
        )

        for d, t in matches:
            self.trackers[t].update(detections[d])

        for d in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[d]))

        results = []
        for tracker in self.trackers:
            bbox = tracker.get_state()[0]
            results.append([
                int(bbox[0]), int(bbox[1]),
                int(bbox[2]), int(bbox[3]),
                tracker.id
            ])

        return results

    def _associate(self, detections, trackers):
        if len(trackers) == 0:
            return [], list(range(len(detections))), []

        iou_matrix = np.zeros((len(detections), len(trackers)))

        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = iou(det, trk)

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matches = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(trackers)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                matches.append((r, c))
                unmatched_dets.remove(r)
                unmatched_trks.remove(c)

        return matches, unmatched_dets, unmatched_trks
