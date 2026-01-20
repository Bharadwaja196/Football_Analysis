import argparse
import numpy as np
import torch  # type: ignore

from utils.video_utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


def main():
    print("[INFO] CUDA:", torch.cuda.is_available())

    # ----------------------------
    # ARGUMENT PARSING
    # ----------------------------
    parser = argparse.ArgumentParser(description="Football Video Analysis")
    parser.add_argument("--video", required=True, help="Path to input football video")
    args = parser.parse_args()

    # ----------------------------
    # READ VIDEO
    # ----------------------------
    video_frames = read_video(args.video)
    if len(video_frames) == 0:
        raise RuntimeError("No video frames loaded")

    print(f"[INFO] Loaded {len(video_frames)} frames")

    # ----------------------------
    # INITIALIZE TRACKER
    # ----------------------------
    tracker = Tracker("models/best.pt")

    # ----------------------------
    # OBJECT TRACKING
    # ----------------------------
    tracks = tracker.get_object_tracks(video_frames)

    # Ball interpolation FIRST
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Add positions
    tracker.add_position_to_tracks(tracks)

    # ----------------------------
    # CAMERA MOTION
    # ----------------------------
    camera_est = CameraMovementEstimator(video_frames[0])
    cam_movement = camera_est.get_camera_movement(video_frames)
    camera_est.add_adjust_positions_to_tracks(tracks, cam_movement)

    # ----------------------------
    # VIEW TRANSFORMATION
    # ----------------------------
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # ----------------------------
    # SPEED & DISTANCE
    # ----------------------------
    speed_est = SpeedAndDistance_Estimator()
    speed_est.add_speed_and_distance_to_tracks(tracks)

    # ----------------------------
    # TEAM ASSIGNMENT
    # ----------------------------
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    for f, players in enumerate(tracks["players"]):
        for pid, p in players.items():
            team = team_assigner.get_player_team(video_frames[f], p["bbox"], pid)
            p["team"] = team
            p["team_color"] = team_assigner.team_colors[team]

    # ----------------------------
    # BALL POSSESSION
    # ----------------------------
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for f, players in enumerate(tracks["players"]):
        if 1 not in tracks["ball"][f]:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
            continue

        ball_bbox = tracks["ball"][f][1]["bbox"]
        pid = player_assigner.assign_ball_to_player(players, ball_bbox)

        if pid != -1:
            players[pid]["has_ball"] = True
            team_ball_control.append(players[pid]["team"])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

    team_ball_control = np.array(team_ball_control)

    # ----------------------------
    # DRAW OUTPUT (STABLE METHOD)
    # ----------------------------
    output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    speed_est.draw_speed_and_distance(output_frames, tracks)

    save_video("output_videos/output_video.avi", output_frames)

    print("[SUCCESS] Video saved successfully")


if __name__ == "__main__":
    main()
