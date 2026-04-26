"""
Football Analytics Visualizations Generator
============================================
Generates professional analytics visualizations from the ACTUAL tracking data
produced by our YOLO + ByteTrack pipeline on the input video.

All outputs are saved as high-quality PNG images in the output_videos/ folder.
"""

import os
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Arc
from collections import defaultdict

warnings.filterwarnings('ignore')

# ---- Configuration ----
OUTPUT_DIR = 'output_videos'

BG_COLOR = '#0e1117'
PITCH_COLOR = '#1a1f2e'
LINE_COLOR = '#3d4f6f'
TEXT_COLOR = '#e0e6ed'
TEAM1_COLOR = '#00d4aa'
TEAM2_COLOR = '#ff6b6b'
ACCENT_COLOR = '#ffeb3b'
TITLE_FONT_SIZE = 16
SUBTITLE_FONT_SIZE = 10

# Pitch dimensions in pixels (approximate from the video frame 1920x1080)
FRAME_W = 1920
FRAME_H = 1080


def load_pipeline_data():
    """Fallback: re-run the full pipeline from stubs (only used in standalone mode).
    When called from main.py, data is passed directly -- no hardcoded paths needed.
    """
    print('[INFO] No pre-computed data passed. Re-loading from pipeline stubs...')

    from utils.video_utils import read_video
    from trackers import Tracker
    from camera_movement_estimator import CameraMovementEstimator
    from view_transformer import ViewTransformer
    from speed_and_distance_estimator import SpeedAndDistance_Estimator
    from team_assigner import TeamAssigner
    from player_ball_assigner import PlayerBallAssigner

    # Auto-detect input video (use first .mp4 found in input_videos/)
    import glob
    video_files = glob.glob('input_videos/*.mp4')
    if not video_files:
        raise FileNotFoundError('No .mp4 files found in input_videos/ folder!')
    video_path = video_files[0]
    print(f'[INFO] Auto-detected input video: {video_path}')

    video_frames = read_video(video_path)
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True,
                                        stub_path='stubs/track_stubs.pkl')
    tracker.add_position_to_tracks(tracks)

    cam = CameraMovementEstimator(video_frames[0])
    cam_move = cam.get_camera_movement(video_frames, read_from_stub=True,
                                        stub_path='stubs/camera_movement_stub.pkl')
    cam.add_adjust_positions_to_tracks(tracks, cam_move)

    vt = ViewTransformer()
    vt.add_transformed_position_to_tracks(tracks)

    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    sde = SpeedAndDistance_Estimator()
    sde.add_speed_and_distance_to_tracks(tracks)

    # Assign teams
    ta = TeamAssigner()
    ta.assign_team_color(video_frames[0], tracks['players'][0])
    for fn, pt in enumerate(tracks['players']):
        for pid, t in pt.items():
            team = ta.get_player_team(video_frames[fn], t['bbox'], pid)
            tracks['players'][fn][pid]['team'] = team
            tracks['players'][fn][pid]['team_color'] = ta.team_colors[team]

    # Assign ball possession
    pa = PlayerBallAssigner()
    team_ball_control = []
    for fn, pt in enumerate(tracks['players']):
        bb = tracks['ball'][fn][1]['bbox']
        ap = pa.assign_ball_to_player(pt, bb)
        if ap != -1:
            tracks['players'][fn][ap]['has_ball'] = True
            team_ball_control.append(tracks['players'][fn][ap]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)

    num_frames = len(tracks['players'])
    fps = 24
    duration_sec = num_frames / fps

    print(f'[INFO] Loaded {num_frames} frames ({duration_sec:.1f}s at {fps}fps)')
    print(f'[INFO] Unique player IDs tracked: {len(set(pid for f in tracks["players"] for pid in f))}')

    return tracks, team_ball_control, num_frames, fps


def add_title(fig, title, subtitle='', y_title=0.96, y_sub=0.925):
    """Add styled title and subtitle to figure."""
    fig.text(0.5, y_title, title, ha='center', va='center',
             fontsize=TITLE_FONT_SIZE, fontweight='bold', color=TEXT_COLOR,
             fontfamily='sans-serif',
             path_effects=[path_effects.withStroke(linewidth=2, foreground=BG_COLOR)])
    if subtitle:
        fig.text(0.5, y_sub, subtitle, ha='center', va='center',
                 fontsize=SUBTITLE_FONT_SIZE, color='#8893a5',
                 fontfamily='sans-serif')


def add_watermark(fig):
    fig.text(0.97, 0.02, 'Football Analysis Project', ha='right', va='bottom',
             fontsize=7, color='#3d4f6f', fontfamily='sans-serif', fontstyle='italic')


def draw_pitch(ax, color=PITCH_COLOR, line_color=LINE_COLOR):
    """Draw a simplified football pitch on the axes (normalized 0-1 coords)."""
    ax.set_xlim(0, FRAME_W)
    ax.set_ylim(FRAME_H, 0)  # Inverted Y (image coords)
    ax.set_facecolor(color)
    ax.set_aspect('equal')
    ax.axis('off')

    # Outer boundary
    rect_kw = dict(fill=False, edgecolor=line_color, linewidth=1.5)
    ax.add_patch(plt.Rectangle((60, 40), FRAME_W - 120, FRAME_H - 80, **rect_kw))

    # Center line
    ax.plot([FRAME_W / 2, FRAME_W / 2], [40, FRAME_H - 40],
            color=line_color, linewidth=1)

    # Center circle
    circle = plt.Circle((FRAME_W / 2, FRAME_H / 2), 80,
                          fill=False, edgecolor=line_color, linewidth=1)
    ax.add_patch(circle)

    # Center dot
    ax.plot(FRAME_W / 2, FRAME_H / 2, 'o', color=line_color, markersize=3)

    # Penalty areas (left and right)
    ax.add_patch(plt.Rectangle((60, FRAME_H / 2 - 200), 200, 400, **rect_kw))
    ax.add_patch(plt.Rectangle((FRAME_W - 260, FRAME_H / 2 - 200), 200, 400, **rect_kw))

    # Goal areas
    ax.add_patch(plt.Rectangle((60, FRAME_H / 2 - 80), 70, 160, **rect_kw))
    ax.add_patch(plt.Rectangle((FRAME_W - 130, FRAME_H / 2 - 80), 70, 160, **rect_kw))


# ---- 1. PLAYER HEAT MAP (per team) ----
def generate_heat_maps(tracks, team_colors, num_frames, output_dir):
    """Generate premium heat maps with smooth gaussian KDE on a vertical pitch."""
    print('[1/4] Generating Team Heat Maps...')
    from scipy.ndimage import gaussian_filter

    # Full pitch dimensions (we'll map transformed coords onto a standard pitch)
    PITCH_L = 105  # length in meters
    PITCH_W = 68   # width in meters

    # View transformer maps to a 23.32 x 68 section -- scale to full pitch for display
    # We'll use pixel positions (more data) mapped onto a normalized pitch

    for team_id in [1, 2]:
        team_label = f'Team {team_id}'
        color = TEAM1_COLOR if team_id == 1 else TEAM2_COLOR

        # Collect positions (use pixel positions, normalize to pitch coords)
        xs, ys = [], []
        for frame_num in range(num_frames):
            for pid, info in tracks['players'][frame_num].items():
                if info.get('team') == team_id:
                    pos = info.get('position')
                    if pos:
                        # Normalize pixel coords to pitch coords (0-105, 0-68)
                        nx = pos[0] / FRAME_W * PITCH_L
                        ny = pos[1] / FRAME_H * PITCH_W
                        xs.append(nx)
                        ys.append(ny)

        # Create figure with dark background
        fig = plt.figure(figsize=(9, 14))
        fig.set_facecolor('#0a0e14')

        # Main pitch axes
        ax = fig.add_axes([0.05, 0.06, 0.82, 0.82])
        ax.set_facecolor('#2d5a27')  # Grass green

        # Add grass stripe effect
        for i in range(0, PITCH_W, 6):
            if (i // 6) % 2 == 0:
                ax.axhspan(i, min(i + 6, PITCH_W), color='#326b2b', alpha=0.3, zorder=0)

        ax.set_xlim(0, PITCH_L)
        ax.set_ylim(0, PITCH_W)
        ax.set_aspect('equal')
        ax.axis('off')

        # Draw pitch markings (vertical orientation: x=length, y=width)
        pitch_line_color = '#ffffff'
        lw = 1.5
        alpha = 0.7

        # Outer boundary
        ax.plot([0, PITCH_L, PITCH_L, 0, 0],
                [0, 0, PITCH_W, PITCH_W, 0],
                color=pitch_line_color, linewidth=lw + 0.5, alpha=alpha, zorder=5)

        # Halfway line
        ax.plot([PITCH_L / 2, PITCH_L / 2], [0, PITCH_W],
                color=pitch_line_color, linewidth=lw, alpha=alpha, zorder=5)

        # Center circle
        circle = plt.Circle((PITCH_L / 2, PITCH_W / 2), 9.15,
                              fill=False, edgecolor=pitch_line_color,
                              linewidth=lw, alpha=alpha, zorder=5)
        ax.add_patch(circle)
        ax.plot(PITCH_L / 2, PITCH_W / 2, 'o', color=pitch_line_color,
                markersize=3, alpha=alpha, zorder=5)

        # Penalty areas (16.5m from goal line, 40.32m wide)
        pa_w = 40.32
        pa_d = 16.5
        # Left penalty area
        ax.plot([0, pa_d, pa_d, 0],
                [(PITCH_W - pa_w) / 2, (PITCH_W - pa_w) / 2,
                 (PITCH_W + pa_w) / 2, (PITCH_W + pa_w) / 2],
                color=pitch_line_color, linewidth=lw, alpha=alpha, zorder=5)
        # Right penalty area
        ax.plot([PITCH_L, PITCH_L - pa_d, PITCH_L - pa_d, PITCH_L],
                [(PITCH_W - pa_w) / 2, (PITCH_W - pa_w) / 2,
                 (PITCH_W + pa_w) / 2, (PITCH_W + pa_w) / 2],
                color=pitch_line_color, linewidth=lw, alpha=alpha, zorder=5)

        # Goal areas (5.5m from goal line, 18.32m wide)
        ga_w = 18.32
        ga_d = 5.5
        ax.plot([0, ga_d, ga_d, 0],
                [(PITCH_W - ga_w) / 2, (PITCH_W - ga_w) / 2,
                 (PITCH_W + ga_w) / 2, (PITCH_W + ga_w) / 2],
                color=pitch_line_color, linewidth=lw, alpha=alpha, zorder=5)
        ax.plot([PITCH_L, PITCH_L - ga_d, PITCH_L - ga_d, PITCH_L],
                [(PITCH_W - ga_w) / 2, (PITCH_W - ga_w) / 2,
                 (PITCH_W + ga_w) / 2, (PITCH_W + ga_w) / 2],
                color=pitch_line_color, linewidth=lw, alpha=alpha, zorder=5)

        # Penalty spots
        ax.plot(11, PITCH_W / 2, 'o', color=pitch_line_color, markersize=3,
                alpha=alpha, zorder=5)
        ax.plot(PITCH_L - 11, PITCH_W / 2, 'o', color=pitch_line_color,
                markersize=3, alpha=alpha, zorder=5)

        # Penalty arcs
        from matplotlib.patches import Arc
        arc1 = Arc((11, PITCH_W / 2), 18.3, 18.3, angle=0,
                    theta1=-53, theta2=53,
                    color=pitch_line_color, linewidth=lw, alpha=alpha, zorder=5)
        arc2 = Arc((PITCH_L - 11, PITCH_W / 2), 18.3, 18.3, angle=0,
                    theta1=127, theta2=233,
                    color=pitch_line_color, linewidth=lw, alpha=alpha, zorder=5)
        ax.add_patch(arc1)
        ax.add_patch(arc2)

        # Goals
        ax.plot([0, 0], [(PITCH_W - 7.32) / 2, (PITCH_W + 7.32) / 2],
                color='#ffffff', linewidth=3, alpha=0.9, zorder=6)
        ax.plot([PITCH_L, PITCH_L], [(PITCH_W - 7.32) / 2, (PITCH_W + 7.32) / 2],
                color='#ffffff', linewidth=3, alpha=0.9, zorder=6)

        # ---- Generate KDE heatmap ----
        if xs:
            # Create density grid
            grid_res = 200
            x_grid = np.linspace(0, PITCH_L, grid_res)
            y_grid = np.linspace(0, PITCH_W, grid_res)

            # 2D histogram then smooth
            heatmap, xedges, yedges = np.histogram2d(
                xs, ys, bins=[grid_res, grid_res],
                range=[[0, PITCH_L], [0, PITCH_W]]
            )

            # Apply gaussian smoothing for silky smooth look
            heatmap = gaussian_filter(heatmap.T, sigma=12)

            # Normalize
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()

            # Custom warm colormap (transparent -> yellow -> orange -> red -> dark red)
            from matplotlib.colors import ListedColormap
            cmap_colors = []
            steps = 256
            for i in range(steps):
                t = i / (steps - 1)
                if t < 0.15:
                    # Transparent
                    cmap_colors.append((0, 0, 0, 0))
                elif t < 0.35:
                    # Fade in cool (light blue tint)
                    a = (t - 0.15) / 0.20
                    cmap_colors.append((0.6, 0.85, 0.95, a * 0.4))
                elif t < 0.50:
                    # Cool to warm transition (yellow-green)
                    a = (t - 0.35) / 0.15
                    r = 0.6 + a * 0.4
                    g = 0.85 - a * 0.1
                    cmap_colors.append((r, g, 0.2, 0.5 + a * 0.15))
                elif t < 0.70:
                    # Warm yellow to orange
                    a = (t - 0.50) / 0.20
                    cmap_colors.append((1.0, 0.75 - a * 0.25, 0.0, 0.65 + a * 0.15))
                elif t < 0.85:
                    # Orange to red
                    a = (t - 0.70) / 0.15
                    cmap_colors.append((1.0 - a * 0.15, 0.5 - a * 0.3, 0.0, 0.8 + a * 0.1))
                else:
                    # Deep red / dark core
                    a = (t - 0.85) / 0.15
                    cmap_colors.append((0.55 - a * 0.15, 0.1, 0.05, 0.9 + a * 0.1))

            heat_cmap = ListedColormap(cmap_colors)

            # Plot heatmap
            ax.imshow(heatmap, extent=[0, PITCH_L, 0, PITCH_W],
                       origin='lower', cmap=heat_cmap, aspect='auto',
                       interpolation='bilinear', zorder=2)

        # ---- Title bar ----
        fig.text(0.05, 0.94, f'TEAM {team_id}', fontsize=10, color='#8893a5',
                 fontfamily='monospace')
        fig.text(0.35, 0.94, 'MATCH ANALYSIS', fontsize=10, color='#8893a5',
                 fontfamily='monospace')
        fig.text(0.65, 0.94, f'HEATMAP ', fontsize=10, color=color,
                 fontweight='bold', fontfamily='monospace')
        fig.text(0.78, 0.94, 'PLAYER POSSESSION DENSITY', fontsize=8,
                 color=TEXT_COLOR, fontfamily='monospace')

        # Separator line
        line_ax = fig.add_axes([0.05, 0.925, 0.87, 0.002])
        line_ax.set_facecolor(LINE_COLOR)
        line_ax.axis('off')

        # ---- Color legend (right side) ----
        legend_ax = fig.add_axes([0.89, 0.30, 0.03, 0.35])
        legend_ax.set_facecolor('#0a0e14')

        # Create gradient bar
        gradient = np.linspace(0, 1, 256).reshape(-1, 1)
        # Simple warm gradient for legend
        legend_cmap = LinearSegmentedColormap.from_list('legend',
            [(0.6, 0.85, 0.95), (1.0, 1.0, 0.3), (1.0, 0.6, 0.0),
             (0.85, 0.2, 0.0), (0.4, 0.05, 0.05)], N=256)

        legend_ax.imshow(gradient, aspect='auto', cmap=legend_cmap,
                          origin='lower', extent=[0, 1, 0, 1])
        legend_ax.set_xticks([])
        legend_ax.set_yticks([])

        # Labels
        legend_ax.text(1.6, 0.95, 'Warm', color='#ff6b3b', fontsize=8,
                        fontweight='bold', va='top', fontfamily='sans-serif')
        legend_ax.text(1.6, 0.80, 'High', color='#cccccc', fontsize=7,
                        va='top', fontfamily='sans-serif')
        legend_ax.text(1.6, 0.15, 'Cool', color='#7bc8e8', fontsize=8,
                        fontweight='bold', va='bottom', fontfamily='sans-serif')
        legend_ax.text(1.6, 0.05, 'Low', color='#cccccc', fontsize=7,
                        va='bottom', fontfamily='sans-serif')

        for spine in legend_ax.spines.values():
            spine.set_edgecolor(LINE_COLOR)
            spine.set_linewidth(0.5)

        # Watermark
        fig.text(0.90, 0.04, 'Football\nAnalysis\nProject', ha='center',
                 va='bottom', fontsize=7, color='#3d4f6f',
                 fontfamily='sans-serif', fontstyle='italic',
                 fontweight='bold', linespacing=1.3)

        filename = f'heat_map_team{team_id}.png'
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='#0a0e14')
        plt.close(fig)
        print(f'    [OK] Saved: {filepath}')


# ---- 2. DISTANCE COVERED ----
def generate_distance_chart(tracks, num_frames, output_dir):
    """Generate a bar chart of total distance covered per player."""
    print('[2/4] Generating Distance Covered Chart...')

    # Get final distance per player
    player_distance = {}
    player_teams = {}

    # Use last frame where player appears
    for frame_num in range(num_frames - 1, -1, -1):
        for pid, info in tracks['players'][frame_num].items():
            if pid not in player_distance:
                dist = info.get('distance')
                team = info.get('team')
                if dist is not None and dist > 0:
                    player_distance[pid] = dist
                if team is not None:
                    player_teams[pid] = team

    # Sort by distance
    sorted_players = sorted(player_distance.items(), key=lambda x: x[1], reverse=True)
    sorted_players = sorted_players[:15]  # Top 15

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.set_facecolor(BG_COLOR)
    ax.set_facecolor(PITCH_COLOR)

    labels = [f'Player {pid}' for pid, _ in sorted_players]
    distances = [d for _, d in sorted_players]
    colors = [TEAM1_COLOR if player_teams.get(pid, 0) == 1 else TEAM2_COLOR
              for pid, _ in sorted_players]

    bars = ax.bar(range(len(labels)), distances, color=colors, alpha=0.85,
                   edgecolor='none', width=0.7)

    # Add value labels on top
    for i, (bar, dist) in enumerate(zip(bars, distances)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{dist:.1f}m', ha='center', va='bottom',
                color=TEXT_COLOR, fontsize=8, fontweight='bold')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, color=TEXT_COLOR, fontsize=9, rotation=45, ha='right')
    ax.set_ylabel('Distance (meters)', color=TEXT_COLOR, fontsize=11)
    ax.tick_params(axis='y', colors=TEXT_COLOR)
    ax.spines['bottom'].set_color(LINE_COLOR)
    ax.spines['left'].set_color(LINE_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Team legend
    from matplotlib.lines import Line2D
    team_legend = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=TEAM1_COLOR,
               markersize=10, label='Team 1', linestyle='None'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=TEAM2_COLOR,
               markersize=10, label='Team 2', linestyle='None'),
    ]
    ax.legend(handles=team_legend, loc='upper right', facecolor=BG_COLOR,
              edgecolor=LINE_COLOR, labelcolor=TEXT_COLOR, fontsize=9)

    add_title(fig, 'Distance Covered by Players',
              f'Top 15 players by total distance  |  From {num_frames} tracked frames')
    add_watermark(fig)

    filepath = os.path.join(output_dir, 'distance_covered.png')
    fig.savefig(filepath, dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    print(f'    [OK] Saved: {filepath}')


# ---- 4. BALL POSSESSION PIE CHART ----
def generate_possession_chart(team_ball_control, output_dir):
    """Generate a ball possession pie/donut chart."""
    print('[3/4] Generating Ball Possession Chart...')

    team1_pct = sum(1 for x in team_ball_control if x == 1) / len(team_ball_control) * 100
    team2_pct = sum(1 for x in team_ball_control if x == 2) / len(team_ball_control) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.set_facecolor(BG_COLOR)
    fig.subplots_adjust(wspace=0.3)

    # -- Left: Donut chart --
    ax1 = axes[0]
    ax1.set_facecolor(BG_COLOR)
    sizes = [team1_pct, team2_pct]
    colors_pie = [TEAM1_COLOR, TEAM2_COLOR]
    explode = (0.02, 0.02)

    wedges, texts, autotexts = ax1.pie(
        sizes, explode=explode, colors=colors_pie, autopct='%1.1f%%',
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.4, edgecolor=BG_COLOR, linewidth=2),
        textprops=dict(color=TEXT_COLOR, fontsize=13, fontweight='bold')
    )

    # Center text
    ax1.text(0, 0, 'Ball\nPossession', ha='center', va='center',
             fontsize=12, fontweight='bold', color=TEXT_COLOR)

    ax1.legend(['Team 1', 'Team 2'], loc='lower center',
               facecolor=BG_COLOR, edgecolor=LINE_COLOR,
               labelcolor=TEXT_COLOR, fontsize=10, ncol=2,
               bbox_to_anchor=(0.5, -0.05))

    # -- Right: Possession timeline --
    ax2 = axes[1]
    ax2.set_facecolor(PITCH_COLOR)

    # Rolling possession (window of 50 frames)
    window = 50
    t1_rolling = []
    for i in range(len(team_ball_control)):
        start = max(0, i - window)
        segment = team_ball_control[start:i + 1]
        t1_pct = sum(1 for x in segment if x == 1) / len(segment) * 100
        t1_rolling.append(t1_pct)

    frames = np.arange(len(t1_rolling))
    t2_rolling = [100 - x for x in t1_rolling]

    ax2.fill_between(frames, t1_rolling, 50, where=[x >= 50 for x in t1_rolling],
                      color=TEAM1_COLOR, alpha=0.4, interpolate=True)
    ax2.fill_between(frames, t1_rolling, 50, where=[x < 50 for x in t1_rolling],
                      color=TEAM2_COLOR, alpha=0.4, interpolate=True)
    ax2.plot(frames, t1_rolling, color=TEXT_COLOR, linewidth=1.5, alpha=0.8)
    ax2.axhline(y=50, color=LINE_COLOR, linestyle='--', linewidth=1, alpha=0.5)

    ax2.set_ylim(0, 100)
    ax2.set_xlim(0, len(t1_rolling))
    ax2.set_ylabel('Team 1 Possession %', color=TEXT_COLOR, fontsize=10)
    ax2.set_xlabel('Frame', color=TEXT_COLOR, fontsize=10)
    ax2.tick_params(colors=TEXT_COLOR)
    ax2.spines['bottom'].set_color(LINE_COLOR)
    ax2.spines['left'].set_color(LINE_COLOR)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Labels
    ax2.text(len(t1_rolling) * 0.95, 75, 'Team 1', color=TEAM1_COLOR,
             fontsize=11, fontweight='bold', ha='right')
    ax2.text(len(t1_rolling) * 0.95, 25, 'Team 2', color=TEAM2_COLOR,
             fontsize=11, fontweight='bold', ha='right')
    ax2.set_title('Possession Timeline (Rolling)', color=TEXT_COLOR,
                   fontsize=12, fontweight='bold', pad=10)

    add_title(fig, f'Ball Possession: Team 1 ({team1_pct:.1f}%) vs Team 2 ({team2_pct:.1f}%)',
              'Computed from ball-to-player proximity across all frames')
    add_watermark(fig)

    filepath = os.path.join(output_dir, 'ball_possession.png')
    fig.savefig(filepath, dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    print(f'    [OK] Saved: {filepath}')


# ---- 5. PLAYER MOVEMENT TRAILS ----
def generate_movement_trails(tracks, num_frames, output_dir):
    """Generate movement trail map showing player trajectories."""
    print('[4/4] Generating Player Movement Trails...')

    # Collect trajectories per player
    trajectories = defaultdict(lambda: {'x': [], 'y': [], 'team': None})

    for frame_num in range(num_frames):
        for pid, info in tracks['players'][frame_num].items():
            pos = info.get('position')
            team = info.get('team')
            if pos:
                trajectories[pid]['x'].append(pos[0])
                trajectories[pid]['y'].append(pos[1])
            if team is not None:
                trajectories[pid]['team'] = team

    # Also collect ball trajectory
    ball_x, ball_y = [], []
    for frame_num in range(num_frames):
        ball_info = tracks['ball'][frame_num].get(1, {})
        bbox = ball_info.get('bbox')
        if bbox:
            bx = (bbox[0] + bbox[2]) / 2
            by = (bbox[1] + bbox[3]) / 2
            ball_x.append(bx)
            ball_y.append(by)

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.set_facecolor(BG_COLOR)
    draw_pitch(ax)

    # Plot player trails (only players with significant data)
    for pid, traj in trajectories.items():
        if len(traj['x']) < 50:
            continue  # Skip short-lived tracks
        color = TEAM1_COLOR if traj['team'] == 1 else TEAM2_COLOR
        ax.plot(traj['x'], traj['y'], color=color, alpha=0.35,
                linewidth=1.2, zorder=3)
        # Mark start position
        ax.scatter(traj['x'][0], traj['y'][0], s=30, color=color,
                   edgecolors=TEXT_COLOR, linewidth=0.5, zorder=5, alpha=0.7)

    # Plot ball trail
    if ball_x:
        ax.plot(ball_x, ball_y, color=ACCENT_COLOR, alpha=0.5,
                linewidth=1, zorder=4, linestyle='-')
        ax.scatter(ball_x[0], ball_y[0], s=50, color=ACCENT_COLOR,
                   edgecolors='white', linewidth=1, zorder=6, marker='o')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=TEAM1_COLOR, linewidth=2, label='Team 1 Trails'),
        Line2D([0], [0], color=TEAM2_COLOR, linewidth=2, label='Team 2 Trails'),
        Line2D([0], [0], color=ACCENT_COLOR, linewidth=2, label='Ball Trail'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', facecolor=BG_COLOR,
              edgecolor=LINE_COLOR, labelcolor=TEXT_COLOR, fontsize=9)

    add_title(fig, 'Player & Ball Movement Trails',
              f'Full trajectories of all tracked players + ball across {num_frames} frames')
    add_watermark(fig)

    filepath = os.path.join(output_dir, 'movement_trails.png')
    fig.savefig(filepath, dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    print(f'    [OK] Saved: {filepath}')


# ---- MAIN ----
def generate_all_analytics(tracks=None, team_ball_control=None):
    """Generate all football analytics visualizations from actual video data.
    
    Args:
        tracks: Pre-computed tracking data from main.py pipeline. If None, 
                will re-load from stubs (standalone mode).
        team_ball_control: Pre-computed ball control list from main.py. If None,
                           will re-compute from stubs.
    """
    print('=' * 60)
    print('[*] FOOTBALL ANALYTICS VISUALIZATION GENERATOR [*]')
    print('    (Using actual tracking data from output video)')
    print('=' * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Use passed data or fall back to re-loading from stubs
    if tracks is not None and team_ball_control is not None:
        num_frames = len(tracks['players'])
        fps = 24
        print(f'[INFO] Using pre-computed pipeline data ({num_frames} frames)')
    else:
        tracks, team_ball_control, num_frames, fps = load_pipeline_data()

    # Extract team colors from the data itself (dynamic, not hardcoded)
    team_colors = {}
    for pid, info in tracks['players'][0].items():
        tc = info.get('team_color')
        team = info.get('team')
        if tc is not None and team is not None:
            team_colors[team] = tc

    print('-' * 60)

    # Generate all visualizations
    generate_heat_maps(tracks, team_colors, num_frames, OUTPUT_DIR)
    generate_distance_chart(tracks, num_frames, OUTPUT_DIR)
    generate_possession_chart(team_ball_control, OUTPUT_DIR)
    generate_movement_trails(tracks, num_frames, OUTPUT_DIR)

    print('-' * 60)
    print(f'[OK] All analytics saved to: {os.path.abspath(OUTPUT_DIR)}/')
    print('   - heat_map_team1.png     (Team 1 heat map)')
    print('   - heat_map_team2.png     (Team 2 heat map)')
    print('   - distance_covered.png   (Distance covered per player)')
    print('   - ball_possession.png    (Possession donut + timeline)')
    print('   - movement_trails.png    (Player & ball trajectories)')
    print('=' * 60)


if __name__ == '__main__':
    generate_all_analytics()
