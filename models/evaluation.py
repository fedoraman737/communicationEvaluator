from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import uuid

@dataclass
class Evaluation:
    """
    Model class representing the LLM evaluation of an advisor's response.
    """
    empathy_score: float
    positioning_score: float
    persuasion_score: float
    overall_score: float
    strengths: List[str]
    areas_for_improvement: List[str]
    feedback: str
    response_id: Optional[str] = None
    created_at: Optional[datetime] = None
    id: str = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.now() 