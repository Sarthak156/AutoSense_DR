import cv2
import numpy as np


class DrivingROI:

    def __init__(
        self,
        width_ratio=0.48,
        height_start=0.615,
        top_width_ratio=0.206,
        bottom_width_ratio=0.48,
        ego_center_ratio=0.67,
        bottom_margin=0.0,

        # Independent corner controls
        top_left_ratio=None,
        top_right_ratio=None,
        bottom_left_ratio=None,
        bottom_right_ratio=None
    ):

        self.width_ratio = width_ratio
        self.height_start = height_start
        self.top_width_ratio = top_width_ratio
        self.bottom_width_ratio = bottom_width_ratio
        self.ego_center_ratio = ego_center_ratio
        self.bottom_margin = bottom_margin

        # ========================================================
        # TOP CORNERS
        # ========================================================

        if top_left_ratio is None:
            self.top_left_ratio = (
                ego_center_ratio -
                top_width_ratio / 2
            )
        else:
            self.top_left_ratio = top_left_ratio

        if top_right_ratio is None:
            self.top_right_ratio = (
                ego_center_ratio +
                top_width_ratio / 2
            )
        else:
            self.top_right_ratio = top_right_ratio

        # ========================================================
        # BOTTOM CORNERS
        # ========================================================

        if bottom_left_ratio is None:
            self.bottom_left_ratio = (
                ego_center_ratio -
                bottom_width_ratio / 2
            )
        else:
            self.bottom_left_ratio = bottom_left_ratio

        if bottom_right_ratio is None:
            self.bottom_right_ratio = (
                ego_center_ratio +
                bottom_width_ratio / 2
            )
        else:
            self.bottom_right_ratio = bottom_right_ratio


    # ============================================================
    # GET POLYGON
    # ============================================================

    def get_polygon(
        self,
        frame_width,
        frame_height
    ):

        top_y = int(
            frame_height *
            self.height_start
        )

        # Keep the bottom slightly above the very bottom
        # when bottom_margin is specified.
        bottom_y = int(
            frame_height *
            (1.0 - self.bottom_margin)
        )

        bottom_y = max(
            top_y + 1,
            min(bottom_y, frame_height)
        )

        top_left = (
            int(
                frame_width *
                self.top_left_ratio
            ),
            top_y
        )

        top_right = (
            int(
                frame_width *
                self.top_right_ratio
            ),
            top_y
        )

        bottom_right = (
            int(
                frame_width *
                self.bottom_right_ratio
            ),
            bottom_y
        )

        bottom_left = (
            int(
                frame_width *
                self.bottom_left_ratio
            ),
            bottom_y
        )

        return [
            top_left,
            top_right,
            bottom_right,
            bottom_left
        ]


    # ============================================================
    # OBJECT IN DRIVING PATH
    # ============================================================

    def is_in_path(
        self,
        bbox,
        frame_width,
        frame_height
    ):

        x1, y1, x2, y2 = bbox

        # Bottom-center of detected object
        object_x = int(
            (x1 + x2) / 2
        )

        object_y = int(y2)

        polygon = self.get_polygon(
            frame_width,
            frame_height
        )

        points = np.array(
            polygon,
            dtype=np.int32
        )

        result = cv2.pointPolygonTest(
            points,
            (object_x, object_y),
            False
        )

        return result >= 0


    # ============================================================
    # DRAW ROI
    # ============================================================

    def draw(self, frame):

        height, width = frame.shape[:2]

        polygon = self.get_polygon(
            width,
            height
        )

        points = np.array(
            polygon,
            dtype=np.int32
        )

        # --------------------------------------------------------
        # Subtle shading
        # --------------------------------------------------------

        overlay = frame.copy()

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

        # --------------------------------------------------------
        # ROI boundary
        # --------------------------------------------------------

        cv2.polylines(
            frame,
            [points],
            True,
            (255, 255, 0),
            3
        )

        return frame