from ai_agents.langgraph.nodes.base_node import BaseNode
from ai_agents.langgraph.states.message_graph import MessageGraphState
from langchain.messages import AIMessage


class FinalizeNode(BaseNode):
  def __init__(self):
    super().__init__()

  def execute_node(self, state: MessageGraphState) -> MessageGraphState:
    messages = state.get('messages', [])

    if messages[-1].content == 'PERFECT_TWEET':
      last_message = messages[-2]
      messages = [AIMessage(last_message.content)]
    else:
      messages = []

    return MessageGraphState(messages=messages)
