import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import torch
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from models.response import Response
from models.evaluation import Evaluation


# ──────────────────── configurable flags ────────────────────
USE_4BIT = True          # toggle 4-bit quantisation
NUM_BEAMS = 2            # beam count for generation
# ─────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
load_dotenv()


class LLMEvaluator:
    """Evaluate advisor responses with the locally stored DeepSeek-Chat model."""

    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode
        self.model = None
        self.tokenizer = None
        if not self.test_mode:
            self._load_model()

    # ─────────────── model loading ───────────────
    def _load_model(self) -> None:
        model_path = Path("models/deepseek")
        if not model_path.exists():
            raise FileNotFoundError(
                "DeepSeek model not found. Run download_model.py first."
            )

        logger.info("Loading DeepSeek-Chat model from %s", model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True
        )

        if USE_4BIT:
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                device_map="auto",
                quantization_config=qcfg,
                use_cache=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto",
                use_cache=True,
            )

        logger.info("DeepSeek-Chat model loaded (device=%s)", self.model.device)

    # ─────────────── public API ───────────────
    def evaluate_response(self, response: Response) -> Evaluation:
        if self.test_mode:
            return self._generate_test_evaluation(response)

        prompt = self._create_evaluation_prompt(response)
        evaluation_dict = self._evaluate_with_model(prompt)

        return Evaluation(
            response_id=response.id,
            empathy_score=evaluation_dict["empathy_score"],
            positioning_score=evaluation_dict["positioning_score"],
            persuasion_score=evaluation_dict["persuasion_score"],
            overall_score=evaluation_dict["overall_score"],
            strengths=evaluation_dict["strengths"],
            areas_for_improvement=evaluation_dict["areas_for_improvement"],
            feedback=evaluation_dict["feedback"],
            created_at=datetime.now(),
        )

    # ─────────────── prompt construction ───────────────
    def _create_evaluation_prompt(self, response: Response) -> str:
        scenario_description = ""
        parts = str(response.id).split("||", 1) if "||" in str(response.id) else [response.id]
        if len(parts) > 1:
            response.id, scenario_description = parts

        response_text = (
            response.text.replace("\u2018", "'")
            .replace("\u2019", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
        )

        rubric = (
            "Use this scoring scale rigorously:\n"
            "0–1  Unacceptable: harms rapport or violates policy\n"
            "2–3  Very poor: little empathy or assistance\n"
            "4–5  Adequate: meets bare minimum, no extra effort\n"
            "6–7  Good: meets needs with some empathy/persuasion\n"
            "8–9  Excellent: proactive, highly empathetic & persuasive\n"
            "10   Outstanding: textbook perfect\n\n"
            "A curt policy-only refusal like the one below should score 1–2 overall.\n"
        )

        header = (
            "You are an expert communication-skills evaluator. Analyse the "
            "following customer-service interaction and return *only JSON*.\n"
        )

        scenario_block = (
            f'Customer Scenario: "{scenario_description}"\n'
            if scenario_description
            else f"Customer Scenario ID: {response.scenario_id}\n"
        )

        body = (
            f'{scenario_block}'
            f'Advisor Response: "{response_text}"\n\n'
            "Return strictly valid JSON using this schema:\n"
            "{\n"
            '  "empathy_score": <0-10>,\n'
            '  "positioning_score": <0-10>,\n'
            '  "persuasion_score": <0-10>,\n'
            '  "overall_score": <0-10>,\n'
            '  "strengths": ["bullet #1", "bullet #2", "..."],\n'
            '  "areas_for_improvement": ["bullet #1", "bullet #2", "..."],\n'
            '  "feedback": "A **minimum of 150 words** over at least three '
            'paragraphs. In the first paragraph, justify each numeric score '
            'by quoting or paraphrasing specific phrases the advisor used. '
            'In the second, propose concrete alternative wording the advisor '
            'could use to improve empathy, positioning, and persuasion. '
            'In the third, summarise key takeaways in plain language."\n'
            "}"
        )

        return header + rubric + body

    # ─────────────── model call ───────────────
    def _evaluate_with_model(self, prompt: str) -> Dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=768,
                temperature=0.3,
                top_k=50,
                top_p=0.95,
                num_beams=NUM_BEAMS,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = output_ids[0][input_ids.shape[1]:]
        raw_reply = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()

        logger.info("\nRAW MODEL REPLY\n%s\n%s", "-" * 80, raw_reply)

        try:
            if raw_reply.startswith("```"):
                raw_reply = raw_reply.strip("`\n ")
            result = json.loads(raw_reply)

            defaults = {
                "empathy_score": 5.0,
                "positioning_score": 5.0,
                "persuasion_score": 5.0,
                "overall_score": 5.0,
                "strengths": [],
                "areas_for_improvement": [],
                "feedback": "",
            }
            for k, v in defaults.items():
                result.setdefault(k, v)

            for key in ["empathy_score", "positioning_score", "persuasion_score", "overall_score"]:
                try:
                    result[key] = max(0.0, min(10.0, float(result[key])))
                except (ValueError, TypeError):
                    logger.warning("Invalid %s; defaulting to 5", key)
                    result[key] = 5.0

            return result

        except Exception as exc:
            logger.error("Failed to parse model JSON: %s", exc)
            logger.error("Raw reply was:\n%s", raw_reply)
            return {
                "empathy_score": 5.0,
                "positioning_score": 5.0,
                "persuasion_score": 5.0,
                "overall_score": 5.0,
                "strengths": ["Error parsing model reply"],
                "areas_for_improvement": [
                    "Ensure the model outputs valid JSON (check prompt formatting)"
                ],
                "feedback": "Automatic fallback because the reply was not valid JSON.",
            }

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
