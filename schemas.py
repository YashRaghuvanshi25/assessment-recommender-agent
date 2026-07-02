from typing import List, Optional, Literal

from pydantic import BaseModel, Field, ConfigDict


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    recommendations: Optional[List['Recommendation']] = None


class Recommendation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    id: str
    name: str
    link: str
    score: float
    test_type: Optional[str] = Field(default=None, alias="keys")
    duration: Optional[str] = None
    languages: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    recommendations: Optional[List[Recommendation]] = Field(default=None)