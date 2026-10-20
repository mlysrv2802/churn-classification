import pytest

from churn.registry import MODEL_FILE, ModelRegistry


def _meta(auc: float) -> dict:
    return {"test_metrics": {"roc_auc": auc}, "threshold": 0.35, "trained_at": "t"}


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(tmp_path)


def test_first_model_is_promoted_automatically(registry):
    v1 = registry.save({"fake": "model"}, _meta(0.84))
    promoted, reason = registry.maybe_promote(v1)
    assert promoted and "first model" in reason
    assert registry.production_version() == v1


def test_small_drop_is_still_promoted(registry):
    v1 = registry.save("m1", _meta(0.84)); registry.maybe_promote(v1)
    v2 = registry.save("m2", _meta(0.835))  # dropped 0.005, allowed up to 0.01
    assert registry.maybe_promote(v2)[0] is True
    assert registry.production_version() == v2


def test_big_drop_is_rejected_and_production_is_unchanged(registry):
    v1 = registry.save("m1", _meta(0.84)); registry.maybe_promote(v1)
    v2 = registry.save("m2", _meta(0.80))  # dropped 0.04
    promoted, reason = registry.maybe_promote(v2)
    assert promoted is False and "kept" in reason
    assert registry.production_version() == v1


def test_rollback_switches_production(registry):
    v1 = registry.save("m1", _meta(0.84)); registry.maybe_promote(v1)
    v2 = registry.save("m2", _meta(0.85)); registry.maybe_promote(v2)
    registry.rollback(v1)
    assert registry.production_version() == v1


@pytest.mark.parametrize("bad", ["does-not-exist", "../outside", "a/b", ""])
def test_rollback_rejects_unknown_or_unsafe_versions(registry, bad):
    registry.save("m1", _meta(0.84))
    with pytest.raises(ValueError):
        registry.rollback(bad)


def test_two_quick_saves_get_different_ids(registry):
    ids = {registry.save("m", _meta(0.8)) for _ in range(3)}
    assert len(ids) == 3


def test_list_versions_marks_production(registry):
    v1 = registry.save("m1", _meta(0.84)); registry.maybe_promote(v1)
    v2 = registry.save("m2", _meta(0.70)); registry.maybe_promote(v2)  # rejected
    listed = {v["version"]: v["is_production"] for v in registry.list_versions()}
    assert listed == {v1: True, v2: False}


def test_load_round_trips_model_and_metadata(registry):
    v1 = registry.save({"hello": "world"}, _meta(0.84)); registry.maybe_promote(v1)
    model, metadata = registry.load()
    assert model == {"hello": "world"}
    assert metadata["version"] == v1
    assert (registry.versions_dir / v1 / MODEL_FILE).exists()


def test_loading_with_no_production_model_fails_clearly(registry):
    with pytest.raises(FileNotFoundError):
        registry.load()
    assert registry.list_versions() == []
