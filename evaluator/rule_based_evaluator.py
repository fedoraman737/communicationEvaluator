import re
from datetime import datetime
from models.response import Response
from models.evaluation import Evaluation

class RuleBasedEvaluator:
    """Evaluates advisor responses based on a set of predefined rules and patterns."""

    def __init__(self):
        # In the future, we might load rule sets or configurations here
        pass

    def evaluate_response(self, response: Response) -> Evaluation:
        """
        Evaluates a single advisor response against predefined communication criteria.
        """
        text_to_analyze = response.text.lower() # Convert to lowercase for case-insensitive matching

        # Initial default scores
        empathy_score = 5.0
        positioning_score = 5.0
        persuasion_score = 5.0
        overall_score = 5.0
        strengths = []
        areas_for_improvement = []
        feedback = "Initial rule-based evaluation."

        # Rule 1: Basic Acknowledgement (Empathy)
        acknowledgement_phrases = ["i understand", "i see", "i can see", "i hear you"]
        found_acknowledgement = False
        for phrase in acknowledgement_phrases:
            if phrase in text_to_analyze:
                found_acknowledgement = True
                break
        
        if found_acknowledgement:
            empathy_score = 7.0
            strengths.append("Acknowledged the customer's statement.")
            feedback += " Good job on acknowledging the customer."
        else:
            empathy_score = 3.0 # Lower empathy if no clear acknowledgement
            areas_for_improvement.append("Consider explicitly acknowledging the customer's situation (e.g., using phrases like 'I understand' or 'I see').")
            feedback += " Try to start by acknowledging the customer's point."

        # Rule 2: Penalize Blunt Negative Positioning
        blunt_negative_phrases = [
            "nothing we can do", "nothing i can do", "can't do that", "cannot do that", 
            "not possible", "unable to help", "can't help you with that"
        ]
        found_blunt_negative = False
        for phrase in blunt_negative_phrases:
            if phrase in text_to_analyze:
                found_blunt_negative = True
                break
        
        if found_blunt_negative:
            positioning_score = 1.0  # Harsh penalty for bluntness
            areas_for_improvement.append("Avoid using blunt negative language like 'nothing we can do'. Focus on what is possible or alternative solutions, even when delivering bad news.")
            feedback += " The language used to deliver the difficult news was too blunt."
        else:
            # If not blunt, it's not necessarily good yet, just not bad based on this specific rule.
            # We can add rules for positive positioning later to reward good phrasing.
            pass 

        # Adjust Overall Score Calculation (Simple Average for now)
        # We will refine this as more scores become meaningful
        overall_score = (empathy_score + positioning_score + persuasion_score) / 3
        overall_score = round(max(0.0, min(10.0, overall_score)), 1) # Ensure it's between 0-10 and rounded

        return Evaluation(
            response_id=response.id,
            empathy_score=empathy_score,
            positioning_score=positioning_score, # Placeholder
            persuasion_score=persuasion_score,   # Placeholder
            overall_score=overall_score,
            strengths=strengths,
            areas_for_improvement=areas_for_improvement,
            feedback=feedback,
            created_at=datetime.now(),
        )

    # We will add more methods here to detect other patterns for:
    # - Positive Positioning
    # - Persuasion techniques (e.g., yessets)
    # - Identifying summarization of issues
    # - Checking for clarity, tone (simple keyword-based), etc. 