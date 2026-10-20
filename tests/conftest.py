import pytest

from churn.train import train_and_save


@pytest.fixture(scope="session")
def saved_dir(tmp_path_factory):
    """Train and save a model once for the whole test run (shared by serving and API tests)."""
    model_dir = tmp_path_factory.mktemp("models")
    train_and_save(model_dir=model_dir)
    return model_dir
