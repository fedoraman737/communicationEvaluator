import os
import requests
import json
import pandas as pd
from datetime import datetime
import argparse
import sys

# Add the parent directory to sys.path so we can import the app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.response import Response
from evaluator.llm_evaluator import LLMEvaluator

def parse_forms_export(excel_file):
    """
    Parse a Microsoft Forms Excel export into a list of responses.
    
    Args:
        excel_file: Path to the Excel file exported from Microsoft Forms
        
    Returns:
        A list of Response objects
    """
    try:
        # Read the Excel file
        df = pd.read_excel(excel_file)
        
        # Extract the relevant columns
        # This assumes the Excel export has specific column names
        # Adjust these column names to match your actual Forms export
        responses = []
        
        for _, row in df.iterrows():
            try:
                # Map the Excel columns to our Response model
                advisor_id = row.get('ID', row.get('Name', 'Anonymous'))
                scenario_id = int(row.get('Scenario ID', 1))
                response_text = row.get('Your Response', '')
                
                # Skip empty responses
                if not response_text:
                    continue
                
                # Create a Response object
                response = Response(
                    advisor_id=str(advisor_id),
                    scenario_id=scenario_id,
                    text=response_text,
                    submitted_at=datetime.now()
                )
                
                responses.append(response)
                
            except Exception as e:
                print(f"Error processing row: {e}")
                continue
        
        return responses
    
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []

def evaluate_responses(responses, output_file):
    """
    Evaluate a list of responses and save the results to a file.
    
    Args:
        responses: List of Response objects
        output_file: Path to save the evaluation results
    """
    # Initialize the evaluator with test_mode if no API keys available
    api_key_available = bool(os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY'))
    evaluator = LLMEvaluator(test_mode=not api_key_available)
    
    # Prepare data for output
    output_data = []
    
    # Process each response
    for response in responses:
        print(f"Evaluating response from {response.advisor_id} for scenario {response.scenario_id}")
        
        try:
            # Evaluate the response
            evaluation = evaluator.evaluate_response(response)
            
            # Add to output data
            output_data.append({
                'advisor_id': response.advisor_id,
                'scenario_id': response.scenario_id,
                'empathy_score': evaluation.empathy_score,
                'positioning_score': evaluation.positioning_score,
                'persuasion_score': evaluation.persuasion_score,
                'overall_score': evaluation.overall_score,
                'strengths': ', '.join(evaluation.strengths),
                'areas_for_improvement': ', '.join(evaluation.areas_for_improvement),
                'feedback': evaluation.feedback,
                'response_text': response.text,
                'evaluated_at': evaluation.created_at.isoformat()
            })
            
        except Exception as e:
            print(f"Error evaluating response: {e}")
            continue
    
    # Create a DataFrame and save to Excel
    if output_data:
        df = pd.DataFrame(output_data)
        df.to_excel(output_file, index=False)
        print(f"Evaluation results saved to {output_file}")
    else:
        print("No evaluation results to save")

def main():
    parser = argparse.ArgumentParser(description='Process Microsoft Forms export for communication evaluation')
    parser.add_argument('input_file', help='Path to the Microsoft Forms Excel export file')
    parser.add_argument('--output', default='evaluation_results.xlsx', help='Path to save the evaluation results')
    
    args = parser.parse_args()
    
    # Parse the Forms export
    responses = parse_forms_export(args.input_file)
    
    if not responses:
        print("No valid responses found in the Excel file")
        return
    
    print(f"Found {len(responses)} responses to evaluate")
    
    # Evaluate the responses
    evaluate_responses(responses, args.output)

if __name__ == '__main__':
    main() 