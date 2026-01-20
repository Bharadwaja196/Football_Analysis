from ultralytics import YOLO  # type: ignore

# import supervision as sv
from trackers.sort.sort_tracker import SortManager
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import torch  # type: ignore

from utils import (
    get_center_of_bbox,
    get_bbox_width,
    get_foot_position,
)


class Tracker:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {self.device}")

        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.model.fuse()
        self.sort_tracker = SortManager(iou_threshold=0.3)

    # ----------------------------
    # ADD POSITIONS
    # ----------------------------
    def add_position_to_tracks(self, tracks):
        for obj, obj_tracks in tracks.items():
            for f, frame_tracks in enumerate(obj_tracks):
                for tid, track in frame_tracks.items():
                    bbox = track.get("bbox", None)
                    if bbox is None:
                        continue

                    # Skip invalid / NaN bboxes
                    bbox_arr = np.array(bbox, dtype=float)
                    if np.isnan(bbox_arr).any():
                        continue
                    if obj == "ball":
                        pos = get_center_of_bbox(bbox)
                    else:
                        pos = get_foot_position(bbox)

                    # ALWAYS set position
                    tracks[obj][f][tid]["position"] = pos

    # ----------------------------
    # BALL INTERPOLATION
    # ----------------------------
    def interpolate_ball_positions(self, ball_tracks):
        ball_positions = [
            frame.get(1, {}).get("bbox", [np.nan] * 4) for frame in ball_tracks
        ]

        df = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])
        df = df.interpolate().bfill()

        return [{1: {"bbox": row.tolist()}} for row in df.to_numpy()]

    # ----------------------------
    # YOLO DETECTION (NO RESIZE)
    # ----------------------------
    def detect_frames(self, frames):
        batch_size = 16
        detections = []

        for i in range(0, len(frames), batch_size):
            preds = self.model.predict(
                frames[i : i + batch_size],
                conf=0.05,
                iou=0.6,
                device=self.device,
                verbose=False,
            )
            detections.extend(preds)

        return detections

    # ----------------------------
    # TRACK CREATION
    # ----------------------------
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        # Reset SORT state per run
        self.sort_tracker.trackers = []
        detections = self.detect_frames(frames)
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)

        tracks = {"players": [], "referees": [], "ball": []}

        # Get class names from model (more reliable than per-detection)
        # Try model.names first, then fall back to first detection's names
        cls_names = {}
        if hasattr(self.model, "names") and self.model.names:
            cls_names = self.model.names
        elif len(detections) > 0 and hasattr(detections[0], "names"):
            cls_names = detections[0].names

        for f, det in enumerate(detections):
            # ✅ CREATE FRAME SLOTS FIRST
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # Prepare SORT detections
            player_detections = []
            for d in det.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls_id = d
                cls_name = cls_names.get(int(cls_id), str(int(cls_id)))
                if cls_name in ["player", "goalkeeper"]:
                    player_detections.append([x1, y1, x2, y2])

            # Update SORT
            sort_tracks = self.sort_tracker.update(player_detections)

            # Store players
            for x1, y1, x2, y2, track_id in sort_tracks:
                tracks["players"][f][int(track_id)] = {"bbox": [x1, y1, x2, y2]}

            # Store referees (no tracking; per-frame ids)
            ref_id = 1
            for d in det.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls_id = d
                cls_name = cls_names.get(int(cls_id), str(int(cls_id)))
                if cls_name == "referee":
                    tracks["referees"][f][ref_id] = {"bbox": [x1, y1, x2, y2]}
                    ref_id += 1

            # Store ball (pick best-confidence ball if multiple)
            # Always create entry for interpolation (even if no ball detected)
            best_ball = None  # (conf, bbox)
            for d in det.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls_id = d
                cls_name = cls_names.get(int(cls_id), str(int(cls_id)))
                if cls_name == "ball":
                    bbox = [x1, y1, x2, y2]
                    if best_ball is None or float(conf) > best_ball[0]:
                        best_ball = (float(conf), bbox)

            # Always create ball entry (empty dict if no ball detected - interpolation will handle it)
            if best_ball is not None:
                tracks["ball"][f][1] = {"bbox": best_ball[1]}
            else:
                # Create empty entry so interpolation can fill it
                tracks["ball"][f][1] = {"bbox": [np.nan, np.nan, np.nan, np.nan]}

        if stub_path:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    # ----------------------------
    # DRAW HELPERS
    # ----------------------------
    # def draw_ellipse(self, frame, bbox, color, tid=None):
    def draw_ellipse(self, frame, bbox, color, tid=None):
        if any(np.isnan(bbox)):
            return frame
        y2 = int(bbox[3])
        x, _ = get_center_of_bbox(bbox)
        w = get_bbox_width(bbox)
        cv2.ellipse(frame, (x, y2), (w, int(0.35 * w)), 0, -45, 235, color, 2)
        if tid is not None:
            cv2.putText(
                frame,
                str(tid),
                (x - 10, y2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )
            return frame

        y2 = int(bbox[3])
        x, _ = get_center_of_bbox(bbox)
        w = get_bbox_width(bbox)

        cv2.ellipse(frame, (x, y2), (w, int(0.35 * w)), 0, -45, 235, color, 2)

        if tid is not None:
            cv2.putText(
                frame,
                str(tid),
                (x - 10, y2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )
        return frame

    def draw_triangle(self, frame, bbox, color):
        if any(np.isnan(bbox)):
            return frame
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        pts = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        cv2.drawContours(frame, [pts], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [pts], 0, (0, 0, 0), 2)
        return frame

    # ----------------------------
    # FINAL DRAW
    # ----------------------------
    def draw_annotations(self, frames, tracks, team_ball_control):
        output = []

        for f, frame in enumerate(frames):
            frame = frame.copy()

            for tid, p in tracks["players"][f].items():
                frame = self.draw_ellipse(
                    frame, p["bbox"], p.get("team_color", (0, 0, 255)), tid
                )
                if p.get("has_ball", False):
                    frame = self.draw_triangle(frame, p["bbox"], (0, 0, 255))

            for _, r in tracks["referees"][f].items():
                frame = self.draw_ellipse(frame, r["bbox"], (0, 255, 255))

            for _, b in tracks["ball"][f].items():
                frame = self.draw_triangle(frame, b["bbox"], (0, 255, 0))

            output.append(frame)

        return output
