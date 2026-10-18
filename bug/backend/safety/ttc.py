from collections import defaultdict, deque


class TTCEstimator:

    def __init__(
        self,
        history_size=8,
        min_closing_speed=0.1
    ):

        self.history_size = history_size
        self.min_closing_speed = min_closing_speed

        # Distance history for every tracked object
        self.distance_history = defaultdict(
            lambda: deque(
                maxlen=history_size
            )
        )

        # Time history for every tracked object
        self.time_history = defaultdict(
            lambda: deque(
                maxlen=history_size
            )
        )

    def update(
        self,
        track_id,
        distance,
        timestamp
    ):

        if track_id is None or distance is None:
            return None

        history = self.distance_history[
            track_id
        ]

        times = self.time_history[
            track_id
        ]

        history.append(distance)
        times.append(timestamp)

        # Need at least 2 measurements
        if len(history) < 2:
            return None

        # ----------------------------------------------------
        # Estimate closing speed using oldest/newest samples
        # ----------------------------------------------------

        old_distance = history[0]
        new_distance = history[-1]

        old_time = times[0]
        new_time = times[-1]

        delta_time = (
            new_time - old_time
        )

        if delta_time <= 0:
            return None

        # Positive means object is getting closer
        closing_speed = (
            old_distance - new_distance
        ) / delta_time

        # ----------------------------------------------------
        # Object is not approaching
        # ----------------------------------------------------

        if closing_speed <= self.min_closing_speed:

            return {
                "ttc": None,
                "closing_speed": closing_speed,
                "approaching": False
            }

        # ----------------------------------------------------
        # TTC
        # ----------------------------------------------------

        ttc = (
            new_distance /
            closing_speed
        )

        # Ignore absurdly large TTC values
        if ttc > 60:
            ttc = None

        return {
            "ttc": (
                round(ttc, 2)
                if ttc is not None
                else None
            ),

            "closing_speed": round(
                closing_speed,
                2
            ),

            "approaching": True
        }

    def remove_track(
        self,
        track_id
    ):

        self.distance_history.pop(
            track_id,
            None
        )

        self.time_history.pop(
            track_id,
            None
        )

    def clear(self):

        self.distance_history.clear()
        self.time_history.clear()