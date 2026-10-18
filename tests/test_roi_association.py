"""Tests for explainable, non-destructive object/ROI association."""

from backend.perception.roi_association import ROIObjectAssociator


POLYGON = [(120, 100), (360, 100), (500, 460), (20, 460)]
IMAGE_SHAPE = (480, 640, 3)


def test_inside_and_outside_detections_are_both_returned_and_labeled():
    associator = ROIObjectAssociator()
    boxes = [
        [180, 190, 280, 390],   # in corridor
        [530, 180, 620, 370],   # outside corridor
    ]
    results = [
        associator.associate(box, POLYGON, IMAGE_SHAPE, "car", 0.90, idx,
                             roi_state="detected", roi_confidence=0.82,
                             roi_uncertainty=False)
        for idx, box in enumerate(boxes)
    ]

    assert len(results) == len(boxes)  # no ROI-based deletion
    assert results[0]["in_path"] is True
    assert results[0]["roi_relevance"] == "in_corridor"
    assert results[1]["in_path"] is False
    assert results[1]["roi_relevance"] == "outside_corridor"
    assert results[1]["roi_label"] == "OUTSIDE CORRIDOR"


def test_bbox_overlap_can_associate_when_bottom_center_is_outside():
    associator = ROIObjectAssociator()
    # This box straddles the right polygon edge; its bottom-center lies just
    # outside but meaningful box area overlaps the forward corridor.
    box = [410, 200, 535, 360]
    result = associator.associate(
        box, POLYGON, IMAGE_SHAPE, "truck", 0.95,
        roi_state="detected", roi_confidence=0.8, roi_uncertainty=False,
    )
    assert result["roi_overlap"] > 0.12
    assert result["in_path"] is True
    assert result["bottom_center_in_roi"] is False


def test_confidence_class_and_track_history_affect_score_without_hiding_objects():
    high = ROIObjectAssociator()
    low = ROIObjectAssociator()
    box = [250, 220, 390, 400]
    high_result = high.associate(
        box, POLYGON, IMAGE_SHAPE, "car", 0.95, 10,
        roi_state="detected", roi_confidence=0.8, roi_uncertainty=False,
    )
    low_result = low.associate(
        box, POLYGON, IMAGE_SHAPE, "traffic sign", 0.25, 10,
        roi_state="detected", roi_confidence=0.8, roi_uncertainty=False,
    )
    assert high_result["roi_score"] > low_result["roi_score"]
    assert high_result["roi_track_history"] is True
    assert "roi_label" in low_result

    # Track history smooths a brief geometry-score change; it does not remove it.
    previous = high.associate(
        [250, 220, 390, 400], POLYGON, IMAGE_SHAPE, "car", 0.95, 10,
        roi_state="detected", roi_confidence=0.8, roi_uncertainty=False,
    )
    moved = high.associate(
        [560, 200, 635, 360], POLYGON, IMAGE_SHAPE, "car", 0.95, 10,
        roi_state="detected", roi_confidence=0.8, roi_uncertainty=False,
    )
    assert moved["roi_track_history"] is True
    assert moved["roi_score"] > moved["roi_raw_score"]
    assert previous["roi_score"] > moved["roi_raw_score"]


def test_track_history_memory_is_bounded_and_resettable():
    associator = ROIObjectAssociator(max_tracked_objects=2)
    for track_id in (10, 11, 12):
        associator.associate(
            [200, 200, 300, 390], POLYGON, IMAGE_SHAPE, "car", 0.8, track_id,
            roi_state="detected", roi_confidence=0.8, roi_uncertainty=False,
        )
    assert len(associator._track_scores) == 2
    assert 10 not in associator._track_scores  # least recently seen evicted
    associator.reset()
    assert len(associator._track_scores) == 0


def test_fallback_or_reused_geometry_is_always_labeled_uncertain():
    associator = ROIObjectAssociator()
    result = associator.associate(
        [200, 200, 300, 390], POLYGON, IMAGE_SHAPE, "person", 0.88, 5,
        roi_state="fallback", roi_confidence=0.0, roi_uncertainty=True,
    )
    assert result["in_path"] is True  # configured fallback remains usable
    assert result["roi_relevance"] == "uncertain"
    assert "UNCERTAIN" in result["roi_label"]
    assert result["roi_uncertainty"] is True
