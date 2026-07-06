"""Uniform loader for the Telescope Detectors object used across ablations."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_detector(
    observer_model: str,
    performer_model: Optional[str] = None,
    token: Optional[str] = None,
):
    """Load a Detectors instance. Falls back to HF_TOKEN via utils if token not supplied.

    performer_model defaults to observer_model so a single --model flag is enough
    for the common case where they're the same.
    """
    from llm_text_detectors import Detectors
    from llm_text_detectors.utils import get_hugging_face_auth_token

    if token is None:
        token = get_hugging_face_auth_token()
    if performer_model is None:
        performer_model = observer_model
    return Detectors(observer_model, performer_model, token)
