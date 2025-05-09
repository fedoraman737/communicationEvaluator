import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import re

from dotenv import load_dotenv

from models.response import Response
from models.evaluation import Evaluation


# ──────────────────── configurable flags ────────────────────
# ─────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
load_dotenv()


class LLMEvaluator:
    """Evaluate advisor responses with the locally stored Microsoft Phi-3 model."""

    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode # This will effectively always be true now

    # ─────────────── model loading ───────────────
    # ─────────────── public API ───────────────
    def evaluate_response(self, response: Response) -> Evaluation:
        return self._generate_test_evaluation(response)

    # ─────────────── prompt construction ───────────────
    # ─────────────── model call ───────────────
    # ─────────────── test-mode stub ───────────────
    def _generate_test_evaluation(self, response: Response) -> Evaluation:
        return Evaluation(
            response_id=response.id,
            empathy_score=7.0,
            positioning_score=7.0,
            persuasion_score=7.0,
            overall_score=7.0,
            strengths=["Test strength"],
            areas_for_improvement=["Test area"],
            feedback="This is a stub evaluation (test_mode=True).",
            created_at=datetime.now(),
        )