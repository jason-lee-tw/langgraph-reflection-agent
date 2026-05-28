from ai_agents.langgraph_reflexion_agent.schemas.answer_question import AnswerQuestion
from pydantic import Field


class ReviseAnswerQuestion(AnswerQuestion):
  """Revise your original answer to your question."""

  references: list[str] = Field(description='Citation motivating your updated answer.')
