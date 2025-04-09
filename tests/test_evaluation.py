import sys
import os
import pytest
from datetime import datetime

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.response import Response
from evaluator.llm_evaluator import LLMEvaluator

def test_response_creation():
    """Test that Response objects are created correctly."""
    response = Response(
        advisor_id="test_advisor",
        scenario_id=1,
        text="This is a test response",
        submitted_at=datetime.now()
    )
    
    assert response.advisor_id == "test_advisor"
    assert response.scenario_id == 1
    assert response.text == "This is a test response"
    assert response.id is not None

def test_evaluation_prompt_creation():
    """Test that the evaluation prompt is created correctly."""
    # Create a test evaluator with test_mode=True to bypass API key validation
    evaluator = LLMEvaluator(test_mode=True)
    
    # Create a test response
    response = Response(
        advisor_id="test_advisor",
        scenario_id=1,
        text="I understand you're frustrated about the refund denial. While our policy doesn't allow refunds for this app, I can suggest some alternatives that might help with your needs.",
        submitted_at=datetime.now()
    )
    
    # Get the evaluation prompt
    prompt = evaluator._create_evaluation_prompt(response)
    
    # Check that the prompt includes the response text
    assert response.text in prompt
    assert str(response.scenario_id) in prompt
    
    # Check that the prompt includes the evaluation criteria
    assert "Empathy" in prompt
    assert "Positioning" in prompt
    assert "Persuasion" in prompt

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not set")
def test_evaluation_with_openai():
    """
    Test the evaluation with OpenAI.
    This test is skipped if no API key is set.
    """
    # Create a test evaluator
    os.environ["LLM_PROVIDER"] = "openai"
    evaluator = LLMEvaluator(test_mode=True)
    
    # Create a test response with a good customer service approach
    response = Response(
        advisor_id="test_advisor",
        scenario_id=1,
        text="I completely understand how frustrating this situation is for you. While our policy doesn't allow refunds for this particular app, I'd like to help you find an alternative solution. Many customers in similar situations have found that [alternative approach] works well. Would you like me to guide you through those options instead?",
        submitted_at=datetime.now()
    )
    
    # Only run actual evaluation if API key is available
    if os.getenv("OPENAI_API_KEY"):
        # Evaluate the response
        evaluation = evaluator.evaluate_response(response)
        
        # Basic validation of the evaluation
        assert evaluation.empathy_score >= 0 and evaluation.empathy_score <= 10
        assert evaluation.positioning_score >= 0 and evaluation.positioning_score <= 10
        assert evaluation.persuasion_score >= 0 and evaluation.persuasion_score <= 10
        assert evaluation.overall_score >= 0 and evaluation.overall_score <= 10
        assert len(evaluation.strengths) > 0
        assert len(evaluation.feedback) > 0 