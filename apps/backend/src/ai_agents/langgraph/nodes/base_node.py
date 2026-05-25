from abc import ABC, abstractmethod

from ai_agents.langgraph.states.message_graph import MessageGraphState
from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage, HumanMessage
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable


class BaseNode(ABC):
  def __init__(self):
    super().__init__()

  @abstractmethod
  def execute_node(self, state: MessageGraphState) -> MessageGraphState: ...


class BaseAgentNode(ABC):
  _model: BaseChatModel
  _prompt_template: ChatPromptTemplate
  _output_message_type: type[BaseMessage] = AIMessage

  def __init__(
    self,
    model: BaseChatModel,
    prompt_template: ChatPromptTemplate,
    output_message_type: type[BaseMessage],
  ):
    super().__init__()
    self._model = model
    self._prompt_template = prompt_template
    self._output_message_type = output_message_type

  @abstractmethod
  def _get_execution_chain(self) -> RunnableSerializable[dict, AIMessage]: ...

  def execute_node(self, state: MessageGraphState) -> MessageGraphState:
    messages = [message for message in state.get('messages', [])]

    if isinstance(messages[-1], AIMessage):
      original_message = messages[-1]
      messages[-1] = HumanMessage(original_message.content)

    model_response = self._get_execution_chain().invoke(
      {
        'messages': messages,
      }
    )
    parsed_response = self._output_message_type(model_response.content)
    return MessageGraphState(messages=[parsed_response])
