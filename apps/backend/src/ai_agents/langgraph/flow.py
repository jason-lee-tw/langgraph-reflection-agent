from ai_agents.langgraph.enums.flow_state import FlowStateEnum
from ai_agents.langgraph.nodes.finalize_node import FinalizeNode
from ai_agents.langgraph.nodes.generation import GenerationNode
from ai_agents.langgraph.nodes.reflection import ReflectionNode
from ai_agents.langgraph.states.message_graph import MessageGraphState
from langgraph.graph import StateGraph


def _should_continue(state: MessageGraphState) -> str:
  messages = state.get('messages', [])

  if len(messages) > 6:
    return FlowStateEnum.FINALIZE

  if messages[-1].content == 'PERFECT_TWEET':
    return FlowStateEnum.FINALIZE

  return FlowStateEnum.GENERATE


def get_flow() -> StateGraph[MessageGraphState]:
  flow = StateGraph(MessageGraphState)

  # Nodes
  generation_node = GenerationNode()
  reflection_node = ReflectionNode()
  finalize_node = FinalizeNode()

  flow.add_node(FlowStateEnum.GENERATE, generation_node.execute_node)
  flow.add_node(FlowStateEnum.REFLECT, reflection_node.execute_node)
  flow.add_node(FlowStateEnum.FINALIZE, finalize_node.execute_node)

  flow.set_entry_point(FlowStateEnum.GENERATE)
  flow.set_finish_point(FlowStateEnum.FINALIZE)

  # Edges
  flow.add_edge(FlowStateEnum.GENERATE, FlowStateEnum.REFLECT)
  flow.add_conditional_edges(
    FlowStateEnum.REFLECT,
    _should_continue,
    {
      FlowStateEnum.FINALIZE: FlowStateEnum.FINALIZE,
      FlowStateEnum.GENERATE: FlowStateEnum.GENERATE,
    },
  )

  return flow
