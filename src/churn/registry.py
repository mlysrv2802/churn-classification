import json
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib

from churn.config import MAX_ACCEPTABLE_AUC_DROP, MODELS_DIR

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class ModelRegistry:
    """Versioned model storage with a 'production' pointer.

    models/
      versions/<id>/model.joblib + metadata.json   one folder per training run
      production.json                              which version the API should serve
    """

    def __init__(self, root: Path = MODELS_DIR):
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.pointer = self.root / "production.json"

    # --- saving -----------------------------------------------------------
    def save(self, model, metadata: dict) -> str:
        """Store a model as a new version and return its id (not yet production)."""
        version = self._new_version_id()
        folder = self.versions_dir / version
        folder.mkdir(parents=True)
        joblib.dump(model, folder / MODEL_FILE)
        saved = {**metadata, "version": version}
        (folder / METADATA_FILE).write_text(json.dumps(saved, indent=2), encoding="utf-8")
        return version

    def _new_version_id(self) -> str:
        base = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        version, n = base, 1
        while (self.versions_dir / version).exists():
            n += 1
            version = f"{base}-{n}"
        return version

    # --- reading ----------------------------------------------------------
    def _folder(self, version: str) -> Path:
        if not _VERSION_PATTERN.match(version):
            raise ValueError(f"Invalid version id: {version!r}")
        folder = self.versions_dir / version
        if not folder.is_dir():
            raise ValueError(f"Unknown model version: {version}")
        return folder

    def metadata(self, version: str) -> dict:
        return json.loads((self._folder(version) / METADATA_FILE).read_text(encoding="utf-8"))

    def production_version(self) -> str | None:
        if not self.pointer.exists():
            return None
        return json.loads(self.pointer.read_text(encoding="utf-8"))["version"]

    def load(self, version: str | None = None):
        """Return (model, metadata) for a version, or for production when version is None."""
        version = version or self.production_version()
        if version is None:
            raise FileNotFoundError(f"No production model in {self.root}")
        folder = self._folder(version)
        return joblib.load(folder / MODEL_FILE), self.metadata(version)

    def list_versions(self) -> list[dict]:
        if not self.versions_dir.exists():
            return []
        production = self.production_version()
        return [
            {**self.metadata(folder.name), "is_production": folder.name == production}
            for folder in sorted(self.versions_dir.iterdir())
            if folder.is_dir()
        ]

    # --- promotion --------------------------------------------------------
    def promote(self, version: str, reason: str) -> None:
        self._folder(version)  # fails if the version does not exist
        record = {
            "version": version,
            "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reason": reason,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        self.pointer.write_text(json.dumps(record, indent=2), encoding="utf-8")

    def maybe_promote(
        self, version: str, max_drop: float = MAX_ACCEPTABLE_AUC_DROP
    ) -> tuple[bool, str]:
        """Promote unless the new model's test ROC-AUC dropped by more than max_drop."""
        current = self.production_version()
        if current is None:
            reason = "first model, nothing to compare with"
            self.promote(version, reason)
            return True, reason

        new_auc = self.metadata(version)["test_metrics"]["roc_auc"]
        old_auc = self.metadata(current)["test_metrics"]["roc_auc"]
        summary = f"ROC-AUC {new_auc:.4f} vs production {old_auc:.4f} (allowed drop {max_drop})"
        if new_auc >= old_auc - max_drop:
            self.promote(version, summary)
            return True, summary
        return False, f"kept {current} in production: {summary}"

    def rollback(self, version: str) -> None:
        """Manually point production at an older (or any) saved version."""
        self.promote(version, "manual rollback")
