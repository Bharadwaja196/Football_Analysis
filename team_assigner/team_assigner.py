import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_jersey_pixels(self, image):
        """
        Filter out grass (green pitch) and skin-tone pixels from a BGR image
        and return only the remaining (jersey-like) pixels as a 2-D array.

        Using HSV gives us colour-space separation of hue from brightness,
        which is far more robust than raw BGR for both masking and clustering.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # --- Grass / pitch mask (green hues, any brightness/saturation) ---
        # OpenCV Hue range is 0-180. Green sits roughly at 35-85.
        lower_green = np.array([25, 40,  40])
        upper_green = np.array([90, 255, 255])
        grass_mask = cv2.inRange(hsv, lower_green, upper_green)

        # --- Skin-tone mask (helps isolate jersey from hands/neck/face) ---
        # Skin occupies two hue bands around red/orange (~0-20 and ~160-180).
        lower_skin1 = np.array([0,   20, 60])
        upper_skin1 = np.array([20, 160, 255])
        lower_skin2 = np.array([160, 20, 60])
        upper_skin2 = np.array([180, 160, 255])
        skin_mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_skin1, upper_skin1),
            cv2.inRange(hsv, lower_skin2, upper_skin2)
        )

        # Union of what we want to REMOVE
        exclude_mask = cv2.bitwise_or(grass_mask, skin_mask)
        keep_mask    = cv2.bitwise_not(exclude_mask)

        # Return only the valid pixels (shape: [N, 3]) in BGR
        jersey_pixels = image[keep_mask > 0]
        return jersey_pixels

    def _dominant_color_via_kmeans(self, pixels, n_clusters=2):
        """
        Run K-Means on a flat pixel array and return the colour of the
        largest cluster (= dominant colour among those pixels).
        """
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++",
                        n_init=10, random_state=42)
        kmeans.fit(pixels)
        counts = np.bincount(kmeans.labels_)
        dominant = np.argmax(counts)
        return kmeans.cluster_centers_[dominant], kmeans

    def _fallback_color(self, image):
        """
        Last-resort colour extraction using the original corner-assumption
        trick.  Only used when masking leaves fewer than MIN_PIXELS pixels.
        """
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++",
                        n_init=10, random_state=42)
        kmeans.fit(image_2d)

        labels          = kmeans.labels_
        clustered       = labels.reshape(image.shape[0], image.shape[1])
        corners         = [clustered[0, 0], clustered[0, -1],
                           clustered[-1, 0], clustered[-1, -1]]
        bg_cluster      = max(set(corners), key=corners.count)
        player_cluster  = 1 - bg_cluster
        return kmeans.cluster_centers_[player_cluster]

    # ------------------------------------------------------------------
    # Public API  (same signatures as before – drop-in replacement)
    # ------------------------------------------------------------------

    MIN_JERSEY_PIXELS = 50   # minimum pixels required after masking

    def get_player_color(self, frame, bbox):
        """
        Return the BGR colour that best represents the player's jersey.

        Steps
        -----
        1. Crop the *top half* of the bounding box (jersey, not legs/boots).
        2. Strip grass and skin pixels via HSV masking.
        3. If enough pixels remain, the dominant K-Means cluster is the jersey.
        4. Fall back to the corner-based method when masking yields too little.
        """
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        player_crop = frame[y1:y2, x1:x2]

        if player_crop.size == 0:
            return np.array([128.0, 128.0, 128.0])   # grey sentinel

        top_half = player_crop[0: int(player_crop.shape[0] / 2), :]

        jersey_pixels = self._get_jersey_pixels(top_half)

        if len(jersey_pixels) >= self.MIN_JERSEY_PIXELS:
            color, _ = self._dominant_color_via_kmeans(jersey_pixels,
                                                        n_clusters=2)
        else:
            # Masking stripped too many pixels — use fallback
            color = self._fallback_color(top_half)

        return color

    def assign_team_color(self, frame, player_detections):
        """
        Collect one representative jersey colour per visible player and fit
        a 2-cluster K-Means to separate the two teams.  Called once on the
        first frame.
        """
        player_colors = []
        for _, detection in player_detections.items():
            color = self.get_player_color(frame, detection["bbox"])
            player_colors.append(color)

        if len(player_colors) < 2:
            # Can't cluster with fewer than 2 samples – bail out gracefully
            return

        kmeans = KMeans(n_clusters=2, init="k-means++",
                        n_init=10, random_state=42)
        kmeans.fit(player_colors)

        self.kmeans          = kmeans
        self.team_colors[1]  = kmeans.cluster_centers_[0]
        self.team_colors[2]  = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Predict which team a player belongs to based on their jersey colour.
        Results are cached so each player_id is clustered only once.
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)
        team_id = int(self.kmeans.predict(player_color.reshape(1, -1))[0]) + 1

        self.player_team_dict[player_id] = team_id
        return team_id
