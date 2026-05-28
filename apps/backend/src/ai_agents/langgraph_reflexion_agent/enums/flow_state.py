from enum import StrEnum

from langgraph.graph import END as GRAPH_END


class ReflexionFlowState(StrEnum):
  ACTOR = 'REFLEXION_ACTOR'
  REVISOR = 'REVISOR'
  TOOL_EXECUTOR = 'TOOL_EXECUTOR'
  REFLEXION_THINK = 'REFLEXION_THINK'
  END = GRAPH_END
