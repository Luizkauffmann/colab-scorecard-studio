"""
Versioned outputs on Google Drive (or any folder).

Every notebook run writes to ``<output_dir>/<run_id>/<stage>/``. A runtime
reset loses the VM, not these files; :meth:`RunStore.latest` reopens the
most recent run so the notebook can resume instead of starting over.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Optional

import pandas as pd

STAGES = {
    "config": "00_config",
    "intake": "01_intake",
    "split": "02_split",
    "screening": "03_screening",
    "binning": "04_binning",
    "model": "05_model",
    "scorecard": "06_scorecard",
    "rescore": "07_rescore",
}


class RunStore:
    def __init__(self, output_dir: str, run_id: Optional[str] = None):
        self.output_dir = output_dir
        self.run_id = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.root = os.path.join(output_dir, self.run_id)
        os.makedirs(self.root, exist_ok=True)

    @classmethod
    def latest(cls, output_dir: str) -> Optional["RunStore"]:
        if not os.path.isdir(output_dir):
            return None
        runs = sorted(d for d in os.listdir(output_dir)
                      if d.startswith("run_") and os.path.isdir(os.path.join(output_dir, d)))
        return cls(output_dir, runs[-1]) if runs else None

    def folder(self, stage: str) -> str:
        path = os.path.join(self.root, STAGES.get(stage, stage))
        os.makedirs(path, exist_ok=True)
        return path

    def path(self, stage: str, filename: str) -> str:
        return os.path.join(self.folder(stage), filename)

    def write_json(self, stage: str, filename: str, obj: Any) -> str:
        p = self.path(stage, filename)
        with open(p, "w") as fh:
            json.dump(obj, fh, indent=2, default=str)
        return p

    def read_json(self, stage: str, filename: str) -> Any:
        with open(self.path(stage, filename)) as fh:
            return json.load(fh)

    def write_frame(self, stage: str, filename: str, df: pd.DataFrame) -> str:
        p = self.path(stage, filename)
        if filename.endswith(".parquet"):
            df.to_parquet(p, index=False)
        else:
            df.to_csv(p, index=False)
        return p

    def read_frame(self, stage: str, filename: str) -> pd.DataFrame:
        p = self.path(stage, filename)
        return pd.read_parquet(p) if filename.endswith(".parquet") else pd.read_csv(p)

    def __repr__(self) -> str:
        return f"RunStore({self.root!r})"


def in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def mount_drive(mount_point: str = "/content/drive") -> bool:
    """Mount Google Drive in Colab. Returns False outside Colab."""
    if not in_colab():
        return False
    from google.colab import drive  # type: ignore
    if not os.path.isdir(os.path.join(mount_point, "MyDrive")):
        drive.mount(mount_point)
    return True
