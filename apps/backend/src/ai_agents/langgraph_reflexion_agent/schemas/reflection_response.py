from pydantic import BaseModel, Field


class ReflectionResponse(BaseModel):
  missing: str = Field(description='Critique of what is missing.')
  # Superfluous is something that is extra and adding no value in the article.
  superfluous: str = Field(description='Critique of what is superfluous')
