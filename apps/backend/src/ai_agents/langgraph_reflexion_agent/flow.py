from ai_agents.langgraph_reflexion_agent.enums.flow_state import ReflexionFlowState
from ai_agents.langgraph_reflexion_agent.nodes.reflexion_actor import ReflexionActorNode
from ai_agents.langgraph_reflexion_agent.states.reflexion_agent_state import (
  ReflexionAgentState,
)
from langgraph.graph import StateGraph


def get_flow() -> StateGraph[ReflexionAgentState]:
  flow = StateGraph(state_schema=ReflexionAgentState)

  # Nodes
  reflexion_actor = ReflexionActorNode()

  flow.add_node(ReflexionFlowState.ACTOR, reflexion_actor.execute_node)

  # Edges
  flow.set_entry_point(ReflexionFlowState.ACTOR)

  return flow
