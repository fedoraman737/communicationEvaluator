from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Scenario:
    """
    Model class representing a communication scenario to be evaluated.
    """
    id: int
    title: str
    description: str
    category: str
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now() 