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
    Sanitize text by removing problematic characters and normalizing whitespace.
    
    Args:
        text: Text to sanitize
        
    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        return ""
    
    # Replace smart quotes and other problematic characters
    text = text.replace(''', "'").replace(''', "'").replace('"', '"').replace('"', '"')
    text = text.replace('–', '-').replace('—', '-')
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text

class BatchEvaluator:
    """
    Handles batch evaluation of multiple responses using Anthropic's Batch API.
    """
    
    def __init__(self, llm_evaluator: LLMEvaluator):
        """Initialize the batch evaluator."""
        self.llm_evaluator = llm_evaluator
        self.batches = {}  # Store batch information
        
    def process_csv(self, file_path: str) -> str:
        """
        Process a CSV file containing responses.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Batch ID
        """
        # Generate a unique batch ID
        batch_id = str(uuid.uuid4())
        
        # Store batch information
        self.batches[batch_id] = {
            'status': 'processing',
            'file_path': file_path,
            'start_time': datetime.now().isoformat(),
            'completed_responses': 0,
            'total_responses': 0,
            'results': []
        }
        
        try:
            # Detect file encoding
            encoding = detect_encoding(file_path)
            
            # Read the CSV file
            df = pd.read_csv(file_path, encoding=encoding)
            
            # Update total responses count
            self.batches[batch_id]['total_responses'] = len(df)
            
            # Process each response
            for _, row in df.iterrows():
                try:
                    # Create response object
                    response = Response(
                        advisor_id=str(row.get('advisor_id', 'unknown')),
                        scenario_id=str(row.get('scenario_id', 'unknown')),
                        text=sanitize_text(str(row.get('response_text', ''))),
                        submitted_at=datetime.now()
                    )
                    
                    # Evaluate the response
                    evaluation = self.llm_evaluator.evaluate_response(response)
                    
                    # Store the result
                    self.batches[batch_id]['results'].append(evaluation)
                    self.batches[batch_id]['completed_responses'] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing response: {e}")
                    continue
            
            # Mark batch as completed
            self.batches[batch_id]['status'] = 'completed'
            self.batches[batch_id]['end_time'] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            self.batches[batch_id]['status'] = 'failed'
            self.batches[batch_id]['error'] = str(e)
        
        return batch_id
    
    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Get the status of a batch job.
        
        Args:
            batch_id: ID of the batch
            
        Returns:
            Dictionary containing batch status information
        """
        if batch_id not in self.batches:
            raise ValueError(f"Batch ID {batch_id} not found")
        
        batch = self.batches[batch_id]
        return {
            'batch_id': batch_id,
            'status': batch['status'],
            'completed_responses': batch['completed_responses'],
            'total_responses': batch['total_responses'],
            'start_time': batch['start_time'],
            'end_time': batch.get('end_time'),
            'error': batch.get('error')
        }
    
    def get_batch_results(self, batch_id: str, original_csv_path: str) -> List[Evaluation]:
        """
        Get the results of a completed batch job.
        
        Args:
            batch_id: ID of the batch
            original_csv_path: Path to the original CSV file
            
        Returns:
            List of Evaluation objects
        """
        if batch_id not in self.batches:
            raise ValueError(f"Batch ID {batch_id} not found")
        
        batch = self.batches[batch_id]
        if batch['status'] != 'completed':
            raise ValueError(f"Batch {batch_id} is not completed")
        
        return batch['results']
    
    def export_results_to_csv(self, evaluations: List[Evaluation], output_path: str) -> str:
        """
        Export evaluation results to a CSV file.
        
        Args:
            evaluations: List of Evaluation objects
            output_path: Path to save the CSV file
            
        Returns:
            Path to the saved CSV file
        """
        # Convert evaluations to a list of dictionaries
        results = []
        for eval in evaluations:
            results.append({
                'response_id': eval.response_id,
                'score': eval.score,
                'feedback': eval.feedback,
                'timestamp': eval.timestamp
            })
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        
        return output_path 