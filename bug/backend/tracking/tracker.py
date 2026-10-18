from ultralytics import YOLO


class ObjectTracker:

    def __init__(self, model_path, confidence=0.35):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def track(self, frame):

        results = self.model.track(
            source=frame,
            conf=self.confidence,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        return results[0]

    def get_objects(self, result):

        objects = []

        if result.boxes is None:
            return objects

        boxes = result.boxes

        for i in range(len(boxes)):

            # Bounding box
            x1, y1, x2, y2 = (
                boxes.xyxy[i]
                .cpu()
                .numpy()
                .tolist()
            )

            # Class
            cls = int(
                boxes.cls[i]
                .cpu()
                .item()
            )

            # Confidence
            confidence = float(
                boxes.conf[i]
                .cpu()
                .item()
            )

            # Track ID
            track_id = None

            if boxes.id is not None:
                track_id = int(
                    boxes.id[i]
                    .cpu()
                    .item()
                )

            objects.append({
                "track_id": track_id,
                "class_id": cls,
                "confidence": confidence,
                "bbox": [
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ]
            })

        return objects