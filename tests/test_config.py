import os
import subprocess
import sys


def test_paths_can_be_overridden_by_environment(tmp_path):
    env = {
        **os.environ,
        "CHURN_MODELS_DIR": str(tmp_path / "models"),
        "CHURN_DATA_PATH": str(tmp_path / "data.csv"),
    }
    code = "from churn.config import MODELS_DIR, DATA_PATH; print(MODELS_DIR); print(DATA_PATH)"
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    assert out == [str(tmp_path / "models"), str(tmp_path / "data.csv")]
