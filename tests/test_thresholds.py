import pytest

from churn.thresholds import pick_threshold, threshold_table

Y = [1, 1, 0, 0]
PROBA = [0.9, 0.4, 0.6, 0.1]


def test_table_counts_by_hand():
    t = threshold_table(Y, PROBA, thresholds=[0.3, 0.5, 0.7])
    # cutoff 0.3 flags 3 customers: both churners caught, one false alarm
    assert (t.loc[0.3, "caught"], t.loc[0.3, "false_alarms"], t.loc[0.3, "missed"]) == (2, 1, 0)
    assert t.loc[0.3, "recall"] == 1.0
    assert t.loc[0.3, "precision"] == pytest.approx(2 / 3)
    # cutoff 0.7 flags 1 customer: no false alarm, one churner missed
    assert (t.loc[0.7, "caught"], t.loc[0.7, "false_alarms"], t.loc[0.7, "missed"]) == (1, 0, 1)
    assert t.loc[0.7, "precision"] == 1.0
    assert t.loc[0.7, "recall"] == 0.5


def test_higher_cutoff_never_flags_more_customers():
    t = threshold_table(Y, PROBA, thresholds=[0.2, 0.4, 0.6, 0.8])
    assert t["flagged"].is_monotonic_decreasing


def test_pick_threshold_by_f1_and_cost():
    t = threshold_table(Y, PROBA, thresholds=[0.3, 0.5, 0.7])
    # F1: 0.3 -> 0.8, 0.5 -> 0.5, 0.7 -> 0.667
    assert pick_threshold(t, by="f1") == 0.3
    # missing a churner costs 5, a false alarm 1: cutoff 0.3 costs 1 (the cheapest)
    assert pick_threshold(t, by="cost") == 0.3


def test_pick_threshold_rejects_unknown_metric():
    t = threshold_table(Y, PROBA, thresholds=[0.5])
    with pytest.raises(ValueError):
        pick_threshold(t, by="vibes")
