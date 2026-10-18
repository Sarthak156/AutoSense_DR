from collections import defaultdict, deque


class DistanceEstimator:
    """
    Monocular distance estimator.

    Distance is estimated using:

        D = (H * F) / h

    H = approximate real-world object height
    F = camera focal length in pixels
    h = bounding-box height in pixels

    The raw estimate is smoothed per tracking ID.
    """

    def __init__(
        self,
        focal_length=700,
        smoothing_window=5
    ):

        self.focal_length = focal_length
        self.smoothing_window = smoothing_window

        # Approximate heights in meters.
        self.object_heights = {

            "person": 1.70,
            "pedestrian": 1.70,

            "rider": 1.70,

            "car": 1.50,

            "truck": 3.20,

            "bus": 3.20,

            "motorcycle": 1.40,

            "bicycle": 1.50,

            "train": 3.50,

            "traffic light": 0.60,

            "traffic sign": 0.75
        }

        # Previous distance estimates for each track.
        self.history = defaultdict(
            lambda: deque(
                maxlen=self.smoothing_window
            )
        )

    def estimate(
        self,
        bbox,
        class_name,
        track_id=None
    ):

        x1, y1, x2, y2 = bbox

        pixel_height = y2 - y1

        if pixel_height <= 0:
            return None

        real_height = self.object_heights.get(
            class_name.lower()
        )

        if real_height is None:
            real_height = 1.50

        raw_distance = (
            real_height *
            self.focal_length
        ) / pixel_height

        # ----------------------------------------------------
        # No tracking ID
        # ----------------------------------------------------

        if track_id is None:

            return round(
                raw_distance,
                2
            )

        # ----------------------------------------------------
        # Smooth distance
        # ----------------------------------------------------

        history = self.history[
            track_id
        ]

        history.append(
            raw_distance
        )

        smoothed_distance = (
            sum(history) /
            len(history)
        )

        return round(
            smoothed_distance,
            2
        )

    def clear_track(self, track_id):

        if track_id in self.history:

            del self.history[
                track_id
            ]

    def clear_all(self):

        self.history.clear()