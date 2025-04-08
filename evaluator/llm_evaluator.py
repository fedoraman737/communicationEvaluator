import os
from typing import Dict, List, Any, Union
import json
import logging
import openai
import anthropic
from dotenv import load_dotenv

from models.response import Response
from models.evaluation import Evaluation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

#TODO: Implement batch processing of responses via Anthropic batch processing API
#TODO: Ensure that there is a way to track the status of the batch processing

class LLMEvaluator:
    """
    Class for evaluating advisor responses using LLMs.
    """
    
    def __init__(self, test_mode=False):
        """Initialize the LLM evaluator with the configured provider.
        
        Args:
            test_mode: If True, will not raise errors for missing API keys
        """
        self.provider = os.getenv('LLM_PROVIDER', 'openai')
        # Default model names based on provider
        if self.provider == 'anthropic':
            self.model_name = os.getenv('MODEL_NAME', 'claude-3-7-sonnet-20250219')
        else:
            self.model_name = os.getenv('MODEL_NAME', 'gpt-4o')
        self.test_mode = test_mode
        
        # Initialize clients based on provider
        if self.provider == 'openai':
            openai.api_key = os.getenv('OPENAI_API_KEY')
            if not openai.api_key and not self.test_mode:
                raise ValueError("OpenAI API key not found in environment variables")
            self.client = openai.Client() if openai.api_key else None
        elif self.provider == 'anthropic':
            anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
            if not anthropic_api_key and not self.test_mode:
                raise ValueError("Anthropic API key not found in environment variables")
            self.client = anthropic.Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        
        logger.info(f"Initialized LLMEvaluator with provider: {self.provider}, model: {self.model_name}, test_mode: {test_mode}")
    
    def evaluate_response(self, response: Response) -> Evaluation:
        """
        Evaluate an advisor's response using the configured LLM.
        
        Args:
            response: The Response object containing the advisor's response text
            
        Returns:
            An Evaluation object with scores and feedback
        """
        logger.info(f"Evaluating response ID: {response.id}")
        
        # If in test mode and no API key is available, return a mock evaluation
        if self.test_mode and (not hasattr(self, 'client') or self.client is None):
            logger.info("Using mock evaluation for test mode")
            return Evaluation(
                empathy_score=8.5,
                positioning_score=7.5,
                persuasion_score=6.5,
                overall_score=7.5,
                strengths=["Acknowledges customer frustration", "Offers alternative solutions"],
                areas_for_improvement=["Could improve tone", "More proactive problem-solving"],
                feedback="Good empathy in acknowledging customer frustration. Consider offering more specific alternatives.",
                response_id=response.id
            )
        
        evaluation_prompt = self._create_evaluation_prompt(response)
        
        try:
            # Get evaluation from LLM
            if self.provider == 'openai':
                evaluation_result = self._evaluate_with_openai(evaluation_prompt)
            elif self.provider == 'anthropic':
                evaluation_result = self._evaluate_with_anthropic(evaluation_prompt)
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
            
            # Create and return evaluation object
            evaluation = Evaluation(
                empathy_score=evaluation_result['empathy_score'],
                positioning_score=evaluation_result['positioning_score'],
                persuasion_score=evaluation_result['persuasion_score'],
                overall_score=evaluation_result['overall_score'],
                strengths=evaluation_result['strengths'],
                areas_for_improvement=evaluation_result['areas_for_improvement'],
                feedback=evaluation_result['feedback'],
                response_id=response.id
            )
            
            logger.info(f"Evaluation complete for response ID: {response.id}")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating response: {str(e)}")
            # Return a default evaluation in case of error
            return Evaluation(
                empathy_score=5.0,
                positioning_score=5.0,
                persuasion_score=5.0,
                overall_score=5.0,
                strengths=["Unable to evaluate strengths due to an error"],
                areas_for_improvement=["Unable to evaluate areas for improvement due to an error"],
                feedback="There was an error processing your response. Please try again later.",
                response_id=response.id
            )
    
    def _create_evaluation_prompt(self, response: Response) -> str:
        """
        Create the prompt for the LLM to evaluate the response.
        
        Args:
            response: The Response object containing the advisor's response text
            
        Returns:
            A string containing the evaluation prompt
        """
        prompt = f"""You are a communication skills expert tasked with evaluating an advisor's response to a customer scenario. Your job is to analyze the response and provide scores and feedback on three key communication domains:

1. Empathy: How well does the advisor acknowledge and connect with the customer's emotions?
2. Positioning: How effectively does the advisor balance positive and negative sentiments, maintaining an appropriate tone?
3. Persuasion: How well does the advisor use persuasion techniques like foot-in-the-door, yes-set, social proof, and reciprocity?

Customer Scenario ID: {response.scenario_id}
Advisor Response: "{response.text}"

Please evaluate the response and provide the following:
1. Empathy Score (0-10): 
2. Positioning Score (0-10): 
3. Persuasion Score (0-10): 
4. Overall Score (0-10): 
5. Strengths (list 2-3 key strengths): 
6. Areas for Improvement (list 2-3 specific areas): 
7. Personalized Feedback (2-3 sentences of actionable advice): 

Format your response as a JSON object with the following keys: empathy_score, positioning_score, persuasion_score, overall_score, strengths (array), areas_for_improvement (array), and feedback (string).
"""
        return prompt
    
    def _evaluate_with_openai(self, prompt: str) -> Dict[str, Any]:
        """
        Evaluate a response using OpenAI's API.
        
        Args:
            prompt: The evaluation prompt
            
        Returns:
            A dictionary containing the evaluation results
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a communication skills expert who evaluates customer service responses."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            evaluation_result = json.loads(response.choices[0].message.content)
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error with OpenAI evaluation: {str(e)}")
            raise
    
    def _evaluate_with_anthropic(self, prompt: str) -> Dict[str, Any]:
        """
        Evaluate a response using Anthropic's API.
        
        Args:
            prompt: The evaluation prompt
            
        Returns:
            A dictionary containing the evaluation results
        """
        try:
            # Use the same system prompt as OpenAI for consistency
            system_prompt = "You are a communication skills expert who evaluates customer service responses. Provide your evaluation in valid JSON format."
            
            response = self.client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7,
                # Added to match standard Anthropic parameters
                anthropic_version="2023-06-01"
            )
            
            # Extract and parse the JSON response
            content = response.content[0].text
            # Find JSON in the response (in case there's additional text)
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                evaluation_result = json.loads(json_str)
                return evaluation_result
            else:
                raise ValueError("Could not find valid JSON in Anthropic response")
            
        except Exception as e:
            logger.error(f"Error with Anthropic evaluation: {str(e)}")
            raise
