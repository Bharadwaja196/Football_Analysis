import cv2
import sys 
sys.path.append('../')
from utils import measure_distance ,get_foot_position

class SpeedAndDistance_Estimator():
    def __init__(self):
        self.frame_window=5
        self.frame_rate=24
        # Approximate conversion: pixels to meters for players outside the
        # perspective-transform polygon. This is a rough estimate based on
        # typical broadcast camera zoom levels. It won't be as accurate as
        # the perspective transform, but gives a reasonable fallback.
        self.pixel_to_meter_fallback = 0.05  # ~20 pixels per meter

    def add_speed_and_distance_to_tracks(self,tracks):
        total_distance= {}
        # Store last known speed/distance per player so we can carry forward
        # when tracking gaps occur
        last_known = {}

        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue 
            number_of_frames = len(object_tracks)
            for frame_num in range(0,number_of_frames, self.frame_window):
                last_frame = min(frame_num+self.frame_window,number_of_frames-1 )

                for track_id,_ in object_tracks[frame_num].items():
                    # --- Fix 2: Handle tracking gaps ---
                    # If the player is not in the last_frame, try to find the
                    # closest frame within the window where they still exist
                    actual_last_frame = last_frame
                    if track_id not in object_tracks[last_frame]:
                        # Search backwards from last_frame to find the player
                        found = False
                        for search_frame in range(last_frame - 1, frame_num, -1):
                            if track_id in object_tracks[search_frame]:
                                actual_last_frame = search_frame
                                found = True
                                break
                        if not found:
                            # Player only exists in the start frame of this window.
                            # Carry forward last known values if available.
                            if track_id in last_known:
                                for frame_num_batch in range(frame_num, last_frame):
                                    if track_id not in tracks[object][frame_num_batch]:
                                        continue
                                    tracks[object][frame_num_batch][track_id]['speed'] = last_known[track_id]['speed']
                                    tracks[object][frame_num_batch][track_id]['distance'] = last_known[track_id]['distance']
                            continue

                    # --- Fix 1: Fallback pixel-based estimation ---
                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[actual_last_frame][track_id]['position_transformed']

                    use_fallback = False
                    if start_position is None or end_position is None:
                        # Use raw adjusted positions (pixel-space) as fallback
                        start_position = object_tracks[frame_num][track_id].get('position_adjusted')
                        end_position = object_tracks[actual_last_frame][track_id].get('position_adjusted')
                        if start_position is None or end_position is None:
                            # Last resort: use raw pixel positions
                            start_position = object_tracks[frame_num][track_id].get('position')
                            end_position = object_tracks[actual_last_frame][track_id].get('position')
                        if start_position is None or end_position is None:
                            continue
                        use_fallback = True
                    
                    distance_covered = measure_distance(start_position,end_position)

                    # Convert pixel distance to meters if using fallback
                    if use_fallback:
                        distance_covered = distance_covered * self.pixel_to_meter_fallback

                    time_elapsed = (actual_last_frame-frame_num)/self.frame_rate
                    if time_elapsed == 0:
                        continue
                    speed_meteres_per_second = distance_covered/time_elapsed
                    speed_km_per_hour = speed_meteres_per_second*3.6

                    if object not in total_distance:
                        total_distance[object]= {}
                    
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    
                    total_distance[object][track_id] += distance_covered

                    # Save as last known values for carry-forward
                    if track_id not in last_known:
                        last_known[track_id] = {}
                    last_known[track_id]['speed'] = speed_km_per_hour
                    last_known[track_id]['distance'] = total_distance[object][track_id]

                    for frame_num_batch in range(frame_num,last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]
    
    def draw_speed_and_distance(self,frames,tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue 
                for _, track_info in object_tracks[frame_num].items():
                   if "speed" in track_info:
                       speed = track_info.get('speed',None)
                       distance = track_info.get('distance',None)
                       if speed is None or distance is None:
                           continue
                       
                       bbox = track_info['bbox']
                       position = get_foot_position(bbox)
                       position = list(position)
                       position[1]+=40

                       position = tuple(map(int,position))
                       cv2.putText(frame, f"{speed:.2f} km/h",position,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
                       cv2.putText(frame, f"{distance:.2f} m",(position[0],position[1]+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
            output_frames.append(frame)
        
        return output_frames