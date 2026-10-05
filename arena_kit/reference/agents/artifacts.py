"""
Bot artifact assembly utilities.

ACTUATOR format only.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Optional

from mjarena.agents.types import BotArtifact


def assemble_bot_artifact(
    *,
    morphology_ref: str,
    controller_code: str,
    metadata: Optional[Dict[str, object]] = None,
) -> BotArtifact:
    """
    Construct a BotArtifact from pre-generated morphology and controller outputs.

    Args:
        morphology_ref: Path or reference to the robot morphology XML.
        controller_code: Python source code for the controller.
        metadata: Optional additional metadata.

    Returns:
        BotArtifact ready for evaluation.
    """
    code_hash = hashlib.sha256(controller_code.encode("utf-8")).hexdigest()
    return BotArtifact(
        morphology_ref=morphology_ref,
        controller_format="ACTUATOR",
        controller_code=controller_code,
        hash=code_hash,
        metadata=metadata or {},
    )
