# video_utils.py
import cv2


def read_video(path, resize_width=960):
    cap = cv2.VideoCapture(path)
    frames = []

    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 🔽 Resize to reduce memory (CRITICAL)
        h, w = frame.shape[:2]
        scale = resize_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)

        frame = cv2.resize(frame, (new_w, new_h))

        frames.append(frame)

    cap.release()
    return frames


def save_video(output_path, frames, fps=25):
    if not isinstance(frames, list):
        raise TypeError(f"Expected frames to be a list of images, got {type(frames)}")

    if len(frames) == 0:
        raise ValueError("No frames to save")

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
