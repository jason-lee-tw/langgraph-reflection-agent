from enum import StrEnum

from langgraph.graph import END as GRAPH_END


class ReflexionFlowState(StrEnum):
  DRAFT = 'DRAFT'
  REVISOR = 'REVISOR'
  TOOL_EXECUTOR = 'TOOL_EXECUTOR'
  END = GRAPH_END
