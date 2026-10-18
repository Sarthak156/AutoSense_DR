import cv2


class SafetyOverlay:

    def __init__(self):
        pass

    # ============================================================
    # DRAW OBJECT
    # ============================================================

    def draw_object(
        self,
        frame,
        bbox,
        class_name,
        track_id=None,
        distance=None,
        ttc=None,
        risk="SAFE",
        in_path=False
    ):

        x1, y1, x2, y2 = bbox

        # --------------------------------------------------------
        # STATUS
        # --------------------------------------------------------

        if not in_path:

            status = "OUTSIDE LANE"

            box_color = (150, 150, 150)

        elif risk == "CRITICAL":

            status = "CRITICAL"

            box_color = (0, 0, 255)

        elif risk == "DANGER":

            status = "DANGER"

            box_color = (0, 80, 255)

        elif risk == "WARNING":

            status = "WARNING"

            box_color = (0, 220, 255)

        else:

            status = "SAFE"

            box_color = (0, 220, 0)

        # --------------------------------------------------------
        # BOUNDING BOX
        # --------------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )

        # --------------------------------------------------------
        # TRACK ID
        # --------------------------------------------------------

        if track_id is not None:

            object_name = (
                f"{class_name.upper()} #{track_id}"
            )

        else:

            object_name = (
                class_name.upper()
            )

        # --------------------------------------------------------
        # DISTANCE
        # --------------------------------------------------------

        if distance is not None:

            distance_text = (
                f"D {distance:.1f}m"
            )

        else:

            distance_text = "D --"

        # --------------------------------------------------------
        # TTC
        # --------------------------------------------------------

        if ttc is not None:

            ttc_text = (
                f"TTC {ttc:.1f}s"
            )

        else:

            ttc_text = "TTC --"

        # --------------------------------------------------------
        # LABEL
        # --------------------------------------------------------

        if in_path:

            label = (
                f"{object_name} | "
                f"{distance_text} | "
                f"{ttc_text}"
            )

        else:

            label = (
                f"{object_name} | "
                f"{distance_text} | "
                f"OUTSIDE LANE"
            )

        # --------------------------------------------------------
        # LABEL POSITION
        # --------------------------------------------------------

        font = cv2.FONT_HERSHEY_SIMPLEX

        font_scale = 0.55

        thickness = 1

        (text_width, text_height), baseline = (
            cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )
        )

        label_x = x1

        label_y = y1 - 8

        # If label would go above frame,
        # put it below the bounding box.

        if label_y - text_height < 0:

            label_y = y2 + text_height + 8

        # Keep label inside frame horizontally.

        label_x = max(
            0,
            min(
                label_x,
                frame.shape[1] - text_width - 8
            )
        )

        # --------------------------------------------------------
        # LABEL BACKGROUND
        # --------------------------------------------------------

        cv2.rectangle(
            frame,
            (
                label_x - 4,
                label_y - text_height - 5
            ),
            (
                label_x + text_width + 4,
                label_y + baseline
            ),
            (15, 20, 25),
            -1
        )

        # --------------------------------------------------------
        # LABEL TEXT
        # --------------------------------------------------------

        cv2.putText(
            frame,
            label,
            (
                label_x,
                label_y
            ),
            font,
            font_scale,
            box_color,
            thickness,
            cv2.LINE_AA
        )

    # ============================================================
    # DRAW ROI
    # ============================================================

    def draw_roi(
        self,
        frame,
        polygon
    ):

        points = (
            __import__("numpy")
            .array(
                polygon,
                dtype=__import__("numpy").int32
            )
        )

        overlay = frame.copy()

        # Very subtle ROI shading

        cv2.fillPoly(
            overlay,
            [points],
            (255, 255, 0)
        )

        frame = cv2.addWeighted(
            overlay,
            0.025,
            frame,
            0.975,
            0
        )

        # Thin boundary

        cv2.polylines(
            frame,
            [points],
            True,
            (255, 255, 0),
            2
        )

        return frame

    # ============================================================
    # DASHBOARD
    # ============================================================

    def draw_dashboard(
        self,
        frame,
        risk,
        decision,
        fps,
        object_count,
        closest_distance
    ):

        # --------------------------------------------------------
        # DASHBOARD POSITION
        # --------------------------------------------------------

        x = 20
        y = 20

        width = 380
        height = 135

        # --------------------------------------------------------
        # BACKGROUND
        # --------------------------------------------------------

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (15, 20, 25),
            -1
        )

        # --------------------------------------------------------
        # RISK COLOR
        # --------------------------------------------------------

        if risk == "CRITICAL":

            risk_color = (0, 0, 255)

        elif risk == "DANGER":

            risk_color = (0, 80, 255)

        elif risk == "WARNING":

            risk_color = (0, 220, 255)

        else:

            risk_color = (0, 220, 0)

        # --------------------------------------------------------
        # RISK
        # --------------------------------------------------------

        cv2.putText(
            frame,
            f"RISK  {risk}",
            (x + 18, y + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            risk_color,
            2,
            cv2.LINE_AA
        )

        # --------------------------------------------------------
        # DECISION
        # --------------------------------------------------------

        cv2.putText(
            frame,
            f"ACTION  {decision}",
            (x + 18, y + 67),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (240, 240, 240),
            2,
            cv2.LINE_AA
        )

        # --------------------------------------------------------
        # CLOSEST DISTANCE
        # --------------------------------------------------------

        if closest_distance is not None:

            distance_text = (
                f"{closest_distance:.1f}m"
            )

        else:

            distance_text = "--"

        # --------------------------------------------------------
        # STATS
        # --------------------------------------------------------

        stats = (
            f"FPS {fps:.1f}   "
            f"Objects {object_count}   "
            f"Closest {distance_text}"
        )

        cv2.putText(
            frame,
            stats,
            (x + 18, y + 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 180, 180),
            1,
            cv2.LINE_AA
        )

        return frame