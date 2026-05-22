from ai_agents.langgraph.enums.flow_state import FlowStateEnum
from langgraph.graph import MessagesState, StateGraph


def _should_continue(state: MessagesState) -> str:
  LAST = -1
  if state['messages'][LAST].tool_calls:
    return FlowStateEnum.ACT
  return FlowStateEnum.END


def get_flow() -> StateGraph[MessagesState]:
  flow = StateGraph(MessagesState)

  # Nodes

  # Edges

  return flow
