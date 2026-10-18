class CollisionRisk:
    """
    Rule-based collision risk assessment.

    This is a research/prototype safety layer.
    It does NOT directly control a real vehicle.
    """

    RISK_PRIORITY = {
        "SAFE": 0,
        "WARNING": 1,
        "DANGER": 2,
        "CRITICAL": 3
    }

    def __init__(self):

        # TTC thresholds in seconds
        self.ttc_critical = 1.0
        self.ttc_danger = 2.0
        self.ttc_warning = 4.0

        # Distance thresholds in meters
        self.distance_critical = 3.0
        self.distance_danger = 6.0
        self.distance_warning = 12.0

    def evaluate(
        self,
        distance,
        ttc,
        closing_speed,
        in_path,
        class_name
    ):

        # ----------------------------------------------------
        # Objects outside driving path
        # ----------------------------------------------------

        if not in_path:

            return "SAFE"

        # ----------------------------------------------------
        # No useful distance information
        # ----------------------------------------------------

        if distance is None:

            return "SAFE"

        # ----------------------------------------------------
        # CRITICAL
        # ----------------------------------------------------

        if distance <= self.distance_critical:

            return "CRITICAL"

        if (
            ttc is not None
            and ttc <= self.ttc_critical
        ):

            return "CRITICAL"

        # ----------------------------------------------------
        # DANGER
        # ----------------------------------------------------

        if distance <= self.distance_danger:

            return "DANGER"

        if (
            ttc is not None
            and ttc <= self.ttc_danger
        ):

            return "DANGER"

        # ----------------------------------------------------
        # WARNING
        # ----------------------------------------------------

        if distance <= self.distance_warning:

            return "WARNING"

        if (
            ttc is not None
            and ttc <= self.ttc_warning
        ):

            return "WARNING"

        # ----------------------------------------------------
        # SAFE
        # ----------------------------------------------------

        return "SAFE"

    def get_priority(
        self,
        risk
    ):

        return self.RISK_PRIORITY.get(
            risk,
            0
        )