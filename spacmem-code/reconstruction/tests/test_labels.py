import pytest

from video_world_state.labels import label_with_candidates


def test_label_aliases_keep_credible_runner_ups():
    label, aliases, _ = label_with_candidates(
        {"printer": 56.0, "copier": 10.0, "paper": 1.0, "shelf": 6.0}
    )

    assert label == "printer"
    assert aliases == ["copier", "shelf"]


def test_label_aliases_drop_one_frame_noise():
    label, aliases, _ = label_with_candidates({"chair": 126.0, "tv": 1.0, "plant": 1.0})

    assert label == "chair"
    assert aliases == []


def test_winning_label_preserves_insertion_order_for_a_tie():
    label, aliases, _ = label_with_candidates({"bag": 21.0, "backpack": 21.0})

    assert label == "bag"
    assert aliases == ["backpack"]


def test_scored_candidates_and_aliases_are_two_views_of_one_ranking():
    label, aliases, candidates = label_with_candidates(
        {"chair": 4.0, "stool": 2.0, "office chair": 0.5}
    )

    assert label == "chair"
    assert aliases == ["stool"]
    assert candidates == [
        {"label": "stool", "support": pytest.approx(2 / 6.5)},
        {"label": "office chair", "support": pytest.approx(0.5 / 6.5)},
    ]


@pytest.mark.parametrize(
    "scores",
    [{}, {"": 1.0}, {"chair": 0.0}, {"chair": True}, {"chair": -1.0}, {"chair": float("nan")}],
)
def test_invalid_label_scores_are_rejected(scores):
    with pytest.raises(ValueError):
        label_with_candidates(scores)
