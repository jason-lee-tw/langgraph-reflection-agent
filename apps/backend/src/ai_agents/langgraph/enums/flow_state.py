from enum import StrEnum

from langgraph.graph import END as GRAPH_END


class FlowStateEnum(StrEnum):
  REFLECT = 'reflect'
  GENERATE = 'generate'
  FINALIZE = 'finalize'
  END = GRAPH_END
