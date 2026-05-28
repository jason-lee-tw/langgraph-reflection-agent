from fastapi import APIRouter
from modules.graph.dto.generate_graph_dto import GenerateGraphResDTO
from modules.graph.graph_service import (
  generate_langgraph_diagram,
  generate_reflexion_langgraph_diagram,
)

router = APIRouter(prefix='/graph', tags=['LangGraph Graph'])


@router.post('/')
def generate_graph() -> GenerateGraphResDTO:
  path = generate_langgraph_diagram()
  return GenerateGraphResDTO(generated_path=path)


@router.post('/reflexion')
def generate_reflexion_graph() -> GenerateGraphResDTO:
  path = generate_reflexion_langgraph_diagram()
  return GenerateGraphResDTO(generated_path=path)
