from enum import StrEnum

from langgraph.graph import END as GRAPH_END


class FlowStateEnum(StrEnum):
  ACT = 'ACT'
  END = GRAPH_END
  AGENT_REASON = 'agent_reason'
