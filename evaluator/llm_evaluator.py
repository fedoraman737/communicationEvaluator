import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import re

import torch
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from threading import Thread

from models.response import Response
from models.evaluation import Evaluation


# ──────────────────── configurable flags ────────────────────
USE_4BIT = True          # toggle 4-bit quantisation
NUM_BEAMS = 2            # beam count for generation (used when streaming is disabled)
ENABLE_STREAMING = True  # toggle streaming output to console
DEBUG_LEVEL = logging.DEBUG  # Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_TO_FILE = True       # Log to file in addition to console
LOG_FILE = "llm_evaluator.log"  # Path to log file
# ─────────────────────────────────────────────────────────────

# Configure logging
logging.basicConfig(
    level=DEBUG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
    ]
)

# Add file handler if enabled
if LOG_TO_FILE:
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

logger = logging.getLogger(__name__)
logger.info("LLMEvaluator initializing with debugging enabled")
load_dotenv()


class LLMEvaluator:
    """Evaluate advisor responses with the locally stored Microsoft Phi-3 model."""

    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode
        self.model = None
        self.tokenizer = None
        if not self.test_mode:
            self._load_model()

    # ─────────────── model loading ───────────────
    def _load_model(self) -> None:
        model_path = Path("models/phi3")
        if not model_path.exists():
            raise FileNotFoundError(
                "Phi-3 model not found. Run download_model.py first."
            )

        logger.info("Loading Microsoft Phi-3-mini-4k-instruct model from %s", model_path)
        load_start_time = time.time()

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
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto",
                use_cache=True,
                trust_remote_code=True,
            )

        load_end_time = time.time()
        logger.info("Phi-3 model loaded (device=%s, loading time: %.2f seconds)", 
                   self.model.device, load_end_time - load_start_time)

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

        # Format prompt for Phi-3 model
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

        # Phi-3 specific prompt formatting
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
            '  "feedback": "A **minimum of 150 words** of feedback. Make sure to justify each numeric score '
            'by quoting or paraphrasing specific phrases, propose concrete alternative wording, '
            'and summarise key takeaways in plain language."\n'
            "}\n\n"
            "IMPORTANT: For the feedback field, create a single coherent paragraph with NO line breaks or newlines.\n"
            "The feedback must be a single-line string with escape sequences (\\n) for any required paragraph breaks.\n"
            "Do NOT create actual multiline text in your JSON output as this will cause parsing errors.\n"
            "Follow standard JSON syntax perfectly, with all strings properly escaped."
        )

        final_prompt = header + rubric + body
        
        # Log token count for the prompt
        input_tokens = len(self.tokenizer.encode(final_prompt))
        logger.info("Prompt created: %d tokens", input_tokens)
        
        return final_prompt

    # ─────────────── model call ───────────────
    def _evaluate_with_model(self, prompt: str) -> Dict[str, Any]:
        # Format message for Phi-3
        messages = [{"role": "user", "content": prompt}]
        
        logger.info("====== EVALUATION REQUEST ======")
        logger.info(f"Prompt length: {len(prompt)} characters")
        
        # Apply chat template - Phi-3 uses a different format than DeepSeek
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        input_token_count = input_ids.shape[1]
        logger.info("Input tokens after applying chat template: %d", input_token_count)
        logger.info("====== BEGINNING INFERENCE ======")

        # Set up streaming if enabled
        inference_start_time = time.time()
        if ENABLE_STREAMING:
            streamer = TextIteratorStreamer(
                self.tokenizer, 
                skip_prompt=True,
                timeout=10.0,
                skip_special_tokens=True
            )
            generation_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": 768,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.95,
                "num_beams": 1,  # Must be 1 for streaming
                "do_sample": True,  # Enable sampling when using temperature/top_p
                "pad_token_id": self.tokenizer.eos_token_id,
                "streamer": streamer,
            }
            
            # Create a thread to run generation
            thread = Thread(target=self._generate_with_streaming, args=(generation_kwargs,))
            thread.start()
            
            # Print tokens as they're generated
            output_text = ""
            token_count = 0
            token_timestamps = []
            print("\n----- STREAMING OUTPUT START -----")
            
            for new_text in streamer:
                current_time = time.time()
                print(new_text, end="", flush=True)
                output_text += new_text
                
                # Count tokens in each chunk
                new_tokens = len(self.tokenizer.encode(new_text)) - 1  # -1 to avoid double counting
                if new_tokens < 0:
                    new_tokens = 0
                token_count += new_tokens
                
                # Record timestamp for token generation rate calculation
                if new_tokens > 0:
                    token_timestamps.append((current_time, new_tokens))
            
            print("\n----- STREAMING OUTPUT END -----\n")
            
            thread.join()
            raw_reply = output_text
            output_tokens = token_count
        else:
            # Regular non-streaming generation with beam search
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
            output_tokens = len(generated_tokens)
            raw_reply = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip()

        inference_end_time = time.time()
        inference_duration = inference_end_time - inference_start_time
        
        # Log statistics with clear section markers for better visibility in logs
        logger.info("====== GENERATION COMPLETE ======")
        logger.info("====== PERFORMANCE METRICS ======")
        logger.info("- Input tokens: %d", input_token_count)
        logger.info("- Output tokens: %d", output_tokens)
        logger.info("- Total tokens: %d", input_token_count + output_tokens)
        logger.info("- Generation time: %.2f seconds", inference_duration)
        logger.info("- Tokens per second: %.2f", output_tokens / inference_duration if inference_duration > 0 else 0)
        
        if ENABLE_STREAMING and token_timestamps:
            # Calculate token generation timeline
            logger.info("====== TOKEN GENERATION TIMELINE ======")
            last_time = inference_start_time
            cumulative_tokens = 0
            
            for i, (timestamp, tokens) in enumerate(token_timestamps):
                time_delta = timestamp - last_time
                cumulative_tokens += tokens
                if i > 0 and time_delta > 0:
                    rate = tokens / time_delta
                    logger.info(f"  {i+1:2d}. +{tokens:3d} tokens at {timestamp - inference_start_time:.2f}s "
                               f"({rate:.2f} t/s, total: {cumulative_tokens})")
                last_time = timestamp
        
        # Log the raw model output with clear markers
        logger.info("====== RAW MODEL REPLY ======")
        # Log in chunks to avoid issues with very large replies
        chunk_size = 2000
        for i in range(0, len(raw_reply), chunk_size):
            chunk = raw_reply[i:i+chunk_size]
            if i == 0:
                logger.info(f"REPLY (part {i//chunk_size + 1}):\n{chunk}")
            else:
                logger.info(f"REPLY (part {i//chunk_size + 1}):\n{chunk}")
        
        logger.info("====== JSON PROCESSING ======")

        # Clean the raw reply string to handle potential invalid JSON characters
        # Log the raw reply for debugging
        logger.info("Beginning JSON cleaning process")
        logger.info(f"Raw JSON length: {len(raw_reply)} characters")
        
        # Handle JSON string issues, particularly with the "feedback" field that may contain newlines
        # First, replace escaped newlines to preserve them
        cleaned_reply = raw_reply.replace("\\n", "\\\\n")
        
        # Look for unescaped newlines within string values (common in the feedback field)
        # This regex pattern will find string patterns and help us identify problematic newlines
        pattern = r'"([^"\\]*(?:\\.[^"\\]*)*)"'
        
        def fix_newlines_in_strings(match):
            # For each string found, replace actual newlines with escaped newlines
            content = match.group(1)
            fixed_content = content.replace("\n", "\\n")
            return f'"{fixed_content}"'
        
        # Apply the regex substitution
        cleaned_reply = re.sub(pattern, fix_newlines_in_strings, cleaned_reply)
        
        # Log the cleaning results
        logger.info(f"Cleaned JSON length: {len(cleaned_reply)} characters")
        if len(cleaned_reply) != len(raw_reply):
            logger.info("JSON was modified during cleaning")

        try:
            # Strip markdown fences if present
            if cleaned_reply.startswith("```"):
                logger.info("Detected markdown code fences, attempting to extract JSON")
                # Assuming the JSON content is within the fences
                json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned_reply, re.IGNORECASE)
                if json_match:
                    cleaned_reply = json_match.group(1).strip()
                    logger.info("Successfully extracted JSON from code fences")
                else:
                    # Fallback if regex fails but fences are detected
                    cleaned_reply = cleaned_reply.strip("`\\n ")
                    logger.info("Fallback: Stripped code fences manually")

            # Attempt to parse JSON with detailed error handling
            try:
                result = json.loads(cleaned_reply)
                logger.info("Successfully parsed JSON")
            except json.JSONDecodeError as json_err:
                # Handle specific JSON parsing errors with more info
                logger.error(f"JSON parsing error: {json_err}")
                logger.error(f"Error at position {json_err.pos}, line {json_err.lineno}, column {json_err.colno}")
                logger.error(f"Error context: '{cleaned_reply[max(0, json_err.pos-40):json_err.pos]}' >>> '{cleaned_reply[json_err.pos:min(len(cleaned_reply), json_err.pos+40)]}'")
                
                # Try a more aggressive JSON cleaning approach as fallback
                logger.info("Attempting more aggressive JSON cleaning...")
                
                # Replace literal newlines with \n in the entire string
                cleaned_reply = cleaned_reply.replace('\n', '\\n')
                
                # Try to fix common JSON structural issues
                # Remove any trailing commas before closing brackets/braces
                cleaned_reply = re.sub(r',\s*}', '}', cleaned_reply)
                cleaned_reply = re.sub(r',\s*]', ']', cleaned_reply)
                
                # Ensure double quotes are used for all keys and string values
                cleaned_reply = re.sub(r'(\w+):', r'"\1":', cleaned_reply)
                cleaned_reply = re.sub(r"'([^']*)'", r'"\1"', cleaned_reply)
                
                logger.info(f"Fallback cleaned JSON: {cleaned_reply[:100]}...")
                result = json.loads(cleaned_reply)

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
            
            # Special handling for the feedback paragraph structure issue
            logger.info("Attempting alternative JSON parsing approach")
            try:
                # Try to extract JSON using a custom approach for handling paragraph breaks in feedback
                json_structure = {}
                
                # Extract scores using regex patterns
                score_pattern = r'"(\w+_score)":\s*(\d+(?:\.\d+)?)'
                scores = re.findall(score_pattern, cleaned_reply)
                for score_name, score_value in scores:
                    json_structure[score_name] = float(score_value)
                
                # Extract arrays (strengths and areas_for_improvement)
                array_pattern = r'"(strengths|areas_for_improvement)":\s*\[(.*?)\]'
                arrays = re.findall(array_pattern, cleaned_reply, re.DOTALL)
                for array_name, array_content in arrays:
                    # Split by commas, but not within quotes
                    items = re.findall(r'"(.*?)"', array_content)
                    json_structure[array_name] = items
                
                # Extract feedback with paragraphs
                feedback_pattern = r'"feedback":\s*"(.*?)(?:"\s*}|",\s*")'
                feedback_match = re.search(feedback_pattern, cleaned_reply, re.DOTALL)
                if feedback_match:
                    # Get the feedback content and clean up any escaped quotes
                    feedback = feedback_match.group(1)
                    feedback = feedback.replace('\\"', '"')
                    # Replace actual newlines with spaces to maintain paragraph structure in plaintext
                    feedback = re.sub(r'\s*\n\s*', ' ', feedback)
                    # Replace multiple spaces with a single space
                    feedback = re.sub(r'\s+', ' ', feedback)
                    json_structure["feedback"] = feedback
                
                logger.info("Successfully extracted JSON using alternative parsing")
                return json_structure
            except Exception as parsing_exc:
                logger.error("Alternative parsing also failed: %s", parsing_exc)
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

    # ─────────────── streaming helper ───────────────
    def _generate_with_streaming(self, generation_kwargs: Dict) -> None:
        """Helper method to run generation in a separate thread when streaming"""
        with torch.no_grad():
            self.model.generate(**generation_kwargs)

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