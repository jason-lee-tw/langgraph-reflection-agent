from pydantic import BaseModel


class GenerateGraphResDTO(BaseModel):
  generated_path: str
