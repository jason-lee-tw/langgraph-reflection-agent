from typing import Literal

from ai_agents.langgraph_reflexion_agent.enums.flow_state import ReflexionFlowState
from ai_agents.langgraph_reflexion_agent.nodes.draft_actor import (
  DraftActorNode,
)
from ai_agents.langgraph_reflexion_agent.nodes.revise_actor import ReviseActorNode
from ai_agents.langgraph_reflexion_agent.nodes.tool_executor import ToolExecutorNode
from ai_agents.langgraph_reflexion_agent.states.reflexion_agent_state import (
  ReflexionAgentState,
)
from langchain.messages import ToolMessage
from langgraph.graph import StateGraph
from server_config.logger import Logger


def __event_loop(
  state: ReflexionAgentState,
) -> Literal[ReflexionFlowState.TOOL_EXECUTOR, ReflexionFlowState.END]:
  logger = Logger(__event_loop.__name__)
  logger.debug('🔄 Executing node...')

  messages = state.get('messages', [])
  tool_call_count = sum(isinstance(message, ToolMessage) for message in messages)

  logger.debug(f'number of messages: {len(messages)}')
  logger.debug(f'messages: {[message.__class__.__name__ for message in messages]}')
  logger.debug(f'tool_call_count: {tool_call_count}')

  if tool_call_count >= 2:
    return ReflexionFlowState.END

  return ReflexionFlowState.TOOL_EXECUTOR


def get_flow() -> StateGraph[ReflexionAgentState]:
  flow = StateGraph(state_schema=ReflexionAgentState)

  # Nodes
  draft_actor = DraftActorNode()
  revise_actor = ReviseActorNode()
  tool_execitor = ToolExecutorNode()

  flow.add_node(ReflexionFlowState.DRAFT, draft_actor.execute_node)
  flow.add_node(ReflexionFlowState.REVISOR, revise_actor.execute_node)
  flow.add_node(ReflexionFlowState.TOOL_EXECUTOR, tool_execitor.get_node())

  # Edges
  flow.set_entry_point(ReflexionFlowState.DRAFT)
  flow.add_edge(ReflexionFlowState.DRAFT, ReflexionFlowState.TOOL_EXECUTOR)
  flow.add_edge(ReflexionFlowState.TOOL_EXECUTOR, ReflexionFlowState.REVISOR)
  flow.add_conditional_edges(
    ReflexionFlowState.REVISOR,
    __event_loop,
    {
      ReflexionFlowState.END: ReflexionFlowState.END,
      ReflexionFlowState.TOOL_EXECUTOR: ReflexionFlowState.TOOL_EXECUTOR,
    },
  )

  return flow
