from ai_agents.langgraph_reflexion_agent.schemas.reflection_response import (
  ReflectionResponse,
)
from pydantic import BaseModel, Field


class AnswerQuestion(BaseModel):
  """Answer the question."""

  answer: str = Field(description='Max 250 word detailed answer to the question.')
  reflection: ReflectionResponse = Field(
    description='Your reflection on the initial answer.'
  )
  search_queries: list[str] = Field(
    description='1 to 3 search queries for researching improvements to address the critique of your current answer'  # noqa: E501
  )
