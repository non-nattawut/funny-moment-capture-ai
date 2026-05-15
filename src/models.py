from pydantic import BaseModel, Field
from typing import List

class FunnyMoment(BaseModel):
    clip_start: float = Field(description="The start timestamp of the funny moment in seconds")
    clip_end: float = Field(description="The end timestamp of the funny moment in seconds")
    reason: str = Field(description="Explanation of why this moment is funny, considering Thai context and slang")

class FunnyMomentsList(BaseModel):
    moments: List[FunnyMoment] = Field(description="A list of identified funny moments from the transcript")
