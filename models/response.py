from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import uuid

@dataclass
class Response:
    """
    Model class representing an advisor's response to a scenario.
    """
    advisor_id: str
    scenario_id: int
    text: str
    submitted_at: datetime
    id: str = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4()) 