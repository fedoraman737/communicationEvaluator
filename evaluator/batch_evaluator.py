import os
import uuid
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import codecs
import re
import requests

# Import models
from models.response import Response
from models.evaluation import Evaluation
from evaluator.llm_evaluator import LLMEvaluator

logger = logging.getLogger(__name__)

def detect_encoding(file_path: str) -> str:
    """
    Detect the encoding of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Detected encoding
    """
    encodings_to_try = ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1', 'iso-8859-1']
    
    # First, check for BOM
    with open(file_path, 'rb') as f:
        raw = f.read(4)  # Read the first 4 bytes
    
    # Check for common BOMs
    if raw.startswith(codecs.BOM_UTF8):
        return 'utf-8-sig'
    elif raw.startswith(codecs.BOM_UTF16_LE):
        return 'utf-16-le'
    elif raw.startswith(codecs.BOM_UTF16_BE):
        return 'utf-16-be'
    
    # Try each encoding
    for encoding in encodings_to_try:
        try:
            with codecs.open(file_path, 'r', encoding=encoding) as f:
                f.read(100)  # Try to read some of the file
            return encoding
        except UnicodeDecodeError:
            continue
    
    # Default to latin-1 as it can handle any byte sequence
    return 'latin-1'

def sanitize_text(text: str) -> str:
    """
    Sanitize text by removing or replacing problematic characters.
    
    Args:
        text: Text to sanitize
        
    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        return "" if pd.isna(text) else str(text)
    
    # Replace common "smart" quotes and apostrophes with standard ones
    text = text.replace(''', "'").replace(''', "'").replace('"', '"').replace('"', '"')
    
    # Replace common dash characters
    text = text.replace('–', '-').replace('—', '-')
    
    # Replace other problematic characters
    text = text.replace('\u2022', '*')  # bullet point
    text = text.replace('\u00A0', ' ')  # non-breaking space
    
    # Remove any remaining control characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    
    return text

class BatchEvaluator:
    """
    Handles batch evaluation of multiple responses using Anthropic's Batch API.
    """
    
    def __init__(self, llm_evaluator: LLMEvaluator):
        """Initialize the batch evaluator with an LLM evaluator.
        
        Args:
            llm_evaluator: An instance of LLMEvaluator to use for creating prompts
        """
        self.llm_evaluator = llm_evaluator
        self.provider = os.getenv('LLM_PROVIDER', 'anthropic')
        self.test_mode = self.llm_evaluator.test_mode
        self.client = self.llm_evaluator.client
        self.model_name = self.llm_evaluator.model_name
        self.batch_id = None
        
    def process_csv(self, file_path: str) -> str:
        """Process a CSV file containing response data.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            batch_id: Identifier for the batch job
        """
        try:
            # Try to detect the encoding first
            detected_encoding = detect_encoding(file_path)
            logger.info(f"Detected encoding: {detected_encoding} for file {file_path}")
            
            # Try multiple encodings
            encodings_to_try = [detected_encoding, 'utf-8', 'cp1252', 'latin-1', 'iso-8859-1']
            df = None
            
            for encoding in encodings_to_try:
                try:
                    # For Excel files
                    if file_path.endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(file_path)
                        break
                    # For CSV files
                    else:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                except UnicodeDecodeError:
                    logger.warning(f"Failed to read with {encoding} encoding, trying next...")
                except Exception as e:
                    logger.error(f"Error reading file with {encoding} encoding: {str(e)}")
                    
            if df is None:
                raise ValueError("Could not read the file with any of the attempted encodings")
            
            # Check if this is the special format where scenarios are column headers
            is_special_format = False
            scenario_columns = []
            
            # Try to detect scenario columns (they typically have long text in headers)
            for col in df.columns:
                if isinstance(col, str) and len(col) > 50:  # Assume long column names are scenarios
                    scenario_columns.append(col)
                    is_special_format = True
            
            # Process the data based on the detected format
            if is_special_format and scenario_columns:
                logger.info(f"Detected special CSV format with {len(scenario_columns)} scenario columns")
                
                # Transform the data: for each row and each scenario column, create an entry
                transformed_data = []
                
                for idx, row in df.iterrows():
                    for scenario_idx, scenario_col in enumerate(scenario_columns, 1):
                        response_text = row.get(scenario_col, "")
                        if pd.notna(response_text) and response_text.strip():  # Only process non-empty responses
                            transformed_data.append({
                                "Id": f"{row.get('Id', idx)}_{scenario_idx}",
                                "Email": row.get("Email", "anonymous"),
                                "Your speech": scenario_idx,
                                "Response": response_text,
                                "scenario_text": scenario_col  # Store the scenario text
                            })
                
                # Create a new dataframe with the transformed data
                if transformed_data:
                    df = pd.DataFrame(transformed_data)
                else:
                    raise ValueError("No valid responses found in the CSV file")
            else:
                # Validate required columns for standard format
                required_columns = ["Id", "Response"]
                for col in required_columns:
                    if col not in df.columns:
                        raise ValueError(f"CSV file must contain a '{col}' column")
            
            # Sanitize text fields
            if "Response" in df.columns:
                df["Response"] = df["Response"].apply(sanitize_text)
            
            # Create batch requests
            batch_requests = []
            
            for idx, row in df.iterrows():
                # Extract data from row
                response_id = str(row.get("Id", idx))
                response_text = row.get("Response", "")
                advisor_id = row.get("Email", "anonymous")
                scenario_id = row.get("Your speech", 1)  # Default to scenario 1 if not specified
                scenario_text = row.get("scenario_text", "")  # Get the scenario text if available
                
                # Create a Response object
                response = Response(
                    advisor_id=advisor_id,
                    scenario_id=scenario_id,
                    text=response_text,
                    submitted_at=datetime.now(),
                    id=response_id
                )
                
                # Generate evaluation prompt
                prompt = self.llm_evaluator._create_evaluation_prompt(response)
                
                # If we have the scenario text, add it to the prompt
                if scenario_text:
                    prompt = prompt.replace(
                        f"Customer Scenario ID: {response.scenario_id}",
                        f"Customer Scenario: \"{scenario_text}\""
                    )
                
                # Create batch request
                batch_request = {
                    "custom_id": response_id,
                    "params": {
                        "model": self.model_name,
                        "max_tokens": 1024,
                        "system": "You are a communication skills expert who evaluates customer service responses.",
                        "messages": [
                            {"role": "user", "content": [
                                {"type": "text", "text": prompt}
                            ]}
                        ]
                    }
                }
                
                # For OpenAI, add response_format
                if self.provider == 'openai':
                    batch_request["params"]["response_format"] = {"type": "json_object"}
                
                batch_requests.append(batch_request)
            
            # If in test mode, return a fake batch ID
            if self.test_mode:
                self.batch_id = f"test_batch_{str(uuid.uuid4())}"
                logger.info(f"Created test batch with ID: {self.batch_id}")
                return self.batch_id
            
            # Submit batch request to Anthropic
            if self.provider == 'anthropic':
                try:
                    response = self.client.messages.batches.create(
                        requests=batch_requests
                    )
                    self.batch_id = response.id
                    logger.info(f"Created Anthropic batch with ID: {self.batch_id}")
                    return self.batch_id
                except Exception as e:
                    # Log more detailed error information
                    logger.error(f"Error creating Anthropic batch: {str(e)}")
                    # Check for specific Anthropic error types if available
                    if hasattr(e, 'status_code'):
                        logger.error(f"Status code: {e.status_code}")
                    if hasattr(e, 'response') and hasattr(e.response, 'text'):
                        logger.error(f"Response body: {e.response.text}")
                    raise
            
            # For other providers (like OpenAI), process sequentially
            # This is a fallback if batch API is not available
            else:
                self.batch_id = f"sequential_batch_{str(uuid.uuid4())}"
                # Process will happen in get_batch_results
                logger.info(f"Created sequential batch with ID: {self.batch_id}")
                return self.batch_id
            
        except Exception as e:
            logger.error(f"Error processing CSV file: {str(e)}")
            raise
    
    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get the status of a batch job.
        
        Args:
            batch_id: The batch job ID
            
        Returns:
            A dictionary with batch status information
        """
        # If in test mode, return fake status
        if self.test_mode or batch_id.startswith("test_batch_"):
            return {
                "id": batch_id,
                "type": "message_batch",
                "processing_status": "ended",
                "request_counts": {
                    "processing": 0,
                    "succeeded": 10,  # Fake count
                    "errored": 0,
                    "canceled": 0,
                    "expired": 0
                },
                "created_at": datetime.now().isoformat(),
                "ended_at": datetime.now().isoformat(),
            }
        
        # For Anthropic, get actual batch status
        if self.provider == 'anthropic' and not batch_id.startswith("sequential_batch_"):
            try:
                # Pass the batch_id as a positional argument, not a keyword argument
                response = self.client.messages.batches.retrieve(batch_id)
                return {
                    "id": response.id,
                    "type": response.type,
                    "processing_status": response.processing_status,
                    "request_counts": response.request_counts,
                    "created_at": response.created_at,
                    "ended_at": response.ended_at,
                }
            except Exception as e:
                logger.error(f"Error getting batch status: {str(e)}")
                raise
        
        # For sequential processing, assume it's completed
        return {
            "id": batch_id,
            "type": "sequential_batch",
            "processing_status": "ended",
            "request_counts": {
                "processing": 0,
                "succeeded": 10,  # Placeholder
                "errored": 0,
                "canceled": 0,
                "expired": 0
            },
            "created_at": datetime.now().isoformat(),
            "ended_at": datetime.now().isoformat(),
        }
    
    def get_batch_results(self, batch_id: str, original_csv_path: str) -> List[Evaluation]:
        """Get the results of a completed batch job.
        
        Args:
            batch_id: The batch job ID
            original_csv_path: Path to the original CSV file
            
        Returns:
            A list of Evaluation objects
        """
        evaluations = []
        
        # Try to detect the encoding first
        detected_encoding = detect_encoding(original_csv_path)
        logger.info(f"Detected encoding: {detected_encoding} for file {original_csv_path}")
        
        # Read the original CSV to get response data
        encodings_to_try = [detected_encoding, 'utf-8', 'cp1252', 'latin-1', 'iso-8859-1']
        df = None
        
        for encoding in encodings_to_try:
            try:
                # For Excel files
                if original_csv_path.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(original_csv_path)
                    break
                # For CSV files
                else:
                    df = pd.read_csv(original_csv_path, encoding=encoding)
                    break
            except UnicodeDecodeError:
                logger.warning(f"Failed to read with {encoding} encoding, trying next...")
            except Exception as e:
                logger.error(f"Error reading file with {encoding} encoding: {str(e)}")
                
        if df is None:
            raise ValueError("Could not read the file with any of the attempted encodings")
        
        # Check if this is the special format where scenarios are column headers
        is_special_format = False
        scenario_columns = []
        
        # Try to detect scenario columns (they typically have long text in headers)
        for col in df.columns:
            if isinstance(col, str) and len(col) > 50:  # Assume long column names are scenarios
                scenario_columns.append(col)
                is_special_format = True
        
        # Process the data based on the detected format
        if is_special_format and scenario_columns:
            logger.info(f"Detected special CSV format with {len(scenario_columns)} scenario columns")
            
            # Transform the data: for each row and each scenario column, create an entry
            transformed_data = []
            
            for idx, row in df.iterrows():
                for scenario_idx, scenario_col in enumerate(scenario_columns, 1):
                    response_text = row.get(scenario_col, "")
                    if pd.notna(response_text) and response_text.strip():  # Only process non-empty responses
                        transformed_data.append({
                            "Id": f"{row.get('Id', idx)}_{scenario_idx}",
                            "Email": row.get("Email", "anonymous"),
                            "Your speech": scenario_idx,
                            "Response": response_text,
                            "scenario_text": scenario_col  # Store the scenario text
                        })
            
            # Create a new dataframe with the transformed data
            if transformed_data:
                df = pd.DataFrame(transformed_data)
            else:
                raise ValueError("No valid responses found in the CSV file")
        
        # Sanitize text fields
        if "Response" in df.columns:
            df["Response"] = df["Response"].apply(sanitize_text)
            
        responses_map = {}
        
        # Create a map of response ID to Response object
        for idx, row in df.iterrows():
            response_id = str(row.get("Id", idx))
            response_text = row.get("Response", "")
            advisor_id = row.get("Email", "anonymous")
            scenario_id = row.get("Your speech", 1)
            scenario_text = row.get("scenario_text", "")  # Get the scenario text if available
            
            response = Response(
                advisor_id=advisor_id,
                scenario_id=scenario_id,
                text=response_text,
                submitted_at=datetime.now(),
                id=response_id
            )
            
            # Store the scenario text in the response object's metadata if needed
            if scenario_text:
                # We'll pass this information in the response_id for now
                # In a real implementation, you'd extend the Response class to include scenario_text
                response.id = f"{response_id}||{scenario_text[:50]}"
            
            responses_map[response_id] = response
        
        # If in test mode, generate fake evaluations
        if self.test_mode or batch_id.startswith("test_batch_"):
            for response_id, response in responses_map.items():
                evaluation = Evaluation(
                    empathy_score=7.5,
                    positioning_score=8.0,
                    persuasion_score=6.5,
                    overall_score=7.3,
                    strengths=["Clear communication", "Empathetic tone"],
                    areas_for_improvement=["Could add more context", "Consider using more persuasive techniques"],
                    feedback="Good response overall. Consider adding more specific details to strengthen your answer.",
                    response_id=response_id
                )
                evaluations.append(evaluation)
            return evaluations
        
        # For Anthropic batch API
        batch_had_errors = False
        if self.provider == 'anthropic' and not batch_id.startswith("sequential_batch_"):
            try:
                # Get batch status to check for results URL
                batch_status = self.client.messages.batches.retrieve(batch_id)
                
                if batch_status.processing_status != "ended":
                    raise ValueError(f"Batch processing has not ended yet. Current status: {batch_status.processing_status}")
                
                if not batch_status.results_url:
                    raise ValueError("Results URL not available")
                
                # Check for errors in the batch
                if batch_status.request_counts.succeeded == 0 and batch_status.request_counts.errored > 0:
                    logger.warning(f"All batch requests failed. Errored: {batch_status.request_counts.errored}")
                    batch_had_errors = True
                
                # Log batch status for debugging
                logger.info(f"Batch status: {batch_status.processing_status}, "
                            f"Succeeded: {batch_status.request_counts.succeeded}, "
                            f"Errored: {batch_status.request_counts.errored}")
                
                # Fetch results from the provided URL even if there are errors
                anthropic_version = os.getenv('ANTHROPIC_API_VERSION', '2023-06-01')
                headers = {
                    "x-api-key": os.getenv('ANTHROPIC_API_KEY'),
                    "anthropic-version": anthropic_version
                }
                
                try:
                    logger.info(f"Fetching batch results using Anthropic API version: {anthropic_version}")
                    response = requests.get(batch_status.results_url, headers=headers)
                    response.raise_for_status()
                except requests.exceptions.RequestException as req_error:
                    logger.error(f"Error fetching batch results: {str(req_error)}")
                    if hasattr(req_error.response, 'text'):
                        logger.error(f"Response error details: {req_error.response.text}")
                    raise
                
                # Process each line (JSONL format)
                any_success = False
                error_details = []
                
                for line in response.text.strip().split('\n'):
                    result_data = json.loads(line)
                    custom_id = result_data.get("custom_id")
                    result = result_data.get("result", {})
                    
                    if result.get("type") == "succeeded":
                        any_success = True
                        message_content = result.get("message", {}).get("content", [])
                        content_text = ""
                        
                        # Extract text content
                        for content_item in message_content:
                            if content_item.get("type") == "text":
                                content_text += content_item.get("text", "")
                        
                        # Parse the JSON response
                        try:
                            # Try to find the JSON in the content
                            json_start = content_text.find('{')
                            json_end = content_text.rfind('}') + 1
                            
                            if json_start >= 0 and json_end > json_start:
                                json_str = content_text[json_start:json_end]
                                evaluation_result = json.loads(json_str)
                            else:
                                # If no JSON found, try the entire content
                                evaluation_result = json.loads(content_text)
                            
                            # Create Evaluation object
                            if custom_id in responses_map:
                                evaluation = Evaluation(
                                    empathy_score=evaluation_result.get("empathy_score", 5.0),
                                    positioning_score=evaluation_result.get("positioning_score", 5.0),
                                    persuasion_score=evaluation_result.get("persuasion_score", 5.0),
                                    overall_score=evaluation_result.get("overall_score", 5.0),
                                    strengths=evaluation_result.get("strengths", []),
                                    areas_for_improvement=evaluation_result.get("areas_for_improvement", []),
                                    feedback=evaluation_result.get("feedback", ""),
                                    response_id=custom_id
                                )
                                evaluations.append(evaluation)
                        except json.JSONDecodeError:
                            logger.error(f"Error parsing JSON from response for ID {custom_id}")
                    else:
                        # Log detailed error information
                        error_type = result.get("error", {}).get("type", "unknown")
                        error_message = result.get("error", {}).get("message", "No error message")
                        logger.warning(f"Request {custom_id} failed with error type: {error_type}, message: {error_message}")
                        error_details.append(f"{error_type}: {error_message}")
                
                # If batch had errors but some succeeded, log them
                if batch_had_errors and any_success:
                    logger.warning(f"Batch had partial success. {len(evaluations)} succeeded out of {len(responses_map)}")
                
                # If no successful evaluations, fall back to sequential processing
                if not any_success and batch_had_errors:
                    logger.warning(f"No successful evaluations from batch API. Common errors: {error_details[:3]}")
                    logger.warning("Falling back to sequential processing...")
                    
                    # Fall back to sequential processing
                    for response_id, response in responses_map.items():
                        try:
                            evaluation = self.llm_evaluator.evaluate_response(response)
                            evaluations.append(evaluation)
                        except Exception as e:
                            logger.error(f"Error in sequential evaluation for response {response_id}: {str(e)}")
            
            except Exception as e:
                logger.error(f"Error getting batch results: {str(e)}")
                logger.warning("Falling back to sequential processing...")
                
                # Fall back to sequential processing if batch API fails completely
                for response_id, response in responses_map.items():
                    try:
                        evaluation = self.llm_evaluator.evaluate_response(response)
                        evaluations.append(evaluation)
                    except Exception as e:
                        logger.error(f"Error in sequential evaluation for response {response_id}: {str(e)}")
        
        # For sequential processing (OpenAI or other providers)
        elif batch_id.startswith("sequential_batch_"):
            for response_id, response in responses_map.items():
                try:
                    # Use the existing LLM evaluator to evaluate each response
                    evaluation = self.llm_evaluator.evaluate_response(response)
                    evaluations.append(evaluation)
                except Exception as e:
                    logger.error(f"Error evaluating response {response_id}: {str(e)}")
        
        return evaluations
    
    def export_results_to_csv(self, evaluations: List[Evaluation], output_path: str) -> str:
        """Export evaluation results to a CSV file.
        
        Args:
            evaluations: List of Evaluation objects
            output_path: Path to save the CSV file
            
        Returns:
            Path to the saved CSV file
        """
        # Create dataframe from evaluations
        data = []
        for eval in evaluations:
            data.append({
                "response_id": eval.response_id,
                "empathy_score": eval.empathy_score,
                "positioning_score": eval.positioning_score,
                "persuasion_score": eval.persuasion_score,
                "overall_score": eval.overall_score,
                "strengths": "; ".join(eval.strengths),
                "areas_for_improvement": "; ".join(eval.areas_for_improvement),
                "feedback": eval.feedback
            })
        
        # Create dataframe and save to CSV with UTF-8 encoding
        results_df = pd.DataFrame(data)
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')  # utf-8-sig for Excel compatibility
        
        return output_path 