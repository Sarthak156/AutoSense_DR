from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path, confidence=0.35):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame):
        """
        Run object detection on a single frame.

        Returns:
            Ultralytics Results object
        """

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False
        )

        return results[0]