import pytest

from avocado_ai.domain import RipenessStage
from avocado_ai.ml.fusion import fuse_predictions
from tests.conftest import THRESHOLDS, make_prediction


def test_single_prediction_is_returned_unchanged():
    p = make_prediction([0.1, 0.1, 0.6, 0.1, 0.1])
    assert fuse_predictions([p], THRESHOLDS) is p


def test_empty_input_raises():
    with pytest.raises(ValueError):
        fuse_predictions([], THRESHOLDS)


def test_agreeing_photos_increase_confidence():
    p = make_prediction([0.05, 0.1, 0.6, 0.2, 0.05])
    fused = fuse_predictions([p, p], THRESHOLDS)
    assert fused.stage is RipenessStage.RIPE_FIRST_STAGE
    assert fused.confidence > p.confidence
    assert fused.images_used == 2
    assert abs(sum(fused.probabilities) - 1.0) < 1e-9


def test_stages_are_not_averaged_as_numbers():
    # Stage 3 vs stage 5 must never "average" to stage 4.
    ripe = make_prediction([0.01, 0.04, 0.85, 0.05, 0.05])
    overripe = make_prediction([0.01, 0.04, 0.05, 0.05, 0.85])
    fused = fuse_predictions([ripe, overripe], THRESHOLDS)
    assert fused.stage is not RipenessStage.RIPE_SECOND_STAGE


def test_heterogeneous_photos_raise_a_warning():
    unripe = make_prediction([0.9, 0.025, 0.025, 0.025, 0.025])
    overripe = make_prediction([0.025, 0.025, 0.025, 0.025, 0.9])
    assert fuse_predictions([unripe, overripe], THRESHOLDS).warning is not None


def test_overripe_has_no_days_left():
    p = make_prediction([0.01, 0.01, 0.03, 0.05, 0.90], days=3.0)
    fused = fuse_predictions([p, p], THRESHOLDS)
    assert fused.stage is RipenessStage.OVERRIPE
    assert fused.days_until_overripe == 0.0


def test_days_weighted_by_support_for_final_stage():
    strong = make_prediction([0.0, 0.0, 0.9, 0.1, 0.0], days=4.0)
    weak = make_prediction([0.0, 0.0, 0.6, 0.4, 0.0], days=10.0)
    fused = fuse_predictions([strong, weak], THRESHOLDS)
    expected = (0.9 * 4.0 + 0.6 * 10.0) / (0.9 + 0.6)
    assert fused.days_until_overripe == pytest.approx(expected)
