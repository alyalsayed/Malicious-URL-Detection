"""
Pydantic request & response models for the Malicious URL Detection API.
"""

from pydantic import BaseModel, HttpUrl
from typing import Dict, Union


class URLRequest(BaseModel):
    url: str   


class URLPrediction(BaseModel):
    input_url: str
    predicted_class: str
    class_id: int
    probabilities: Union[Dict[str, float], str]
