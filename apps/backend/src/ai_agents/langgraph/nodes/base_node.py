from abc import ABC, abstractmethod
from typing import TypedDict

from ai_agents.langgraph.states.message_graph import MessageGraphState
from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable


class BaseState(TypedDict):
  messages: list[BaseMessage]


class BaseNode[State: BaseState = MessageGraphState](ABC):
  def __init__(self):
    super().__init__()

  @abstractmethod
  def execute_node(self, state: State) -> State: ...


class BaseAgentNode[State: BaseState = MessageGraphState](ABC):
  _model: BaseChatModel
  _prompt_template: ChatPromptTemplate
  _output_message_type: type[BaseMessage]

  def __init__(
    self,
    model: BaseChatModel,
    prompt_template: ChatPromptTemplate,
    output_message_type: type[BaseMessage] = AIMessage,
  ):
    super().__init__()
    self._model = model
    self._prompt_template = prompt_template
    self._output_message_type = output_message_type

  @abstractmethod
  def _get_execution_chain(self) -> RunnableSerializable[dict, AnyMessage]: ...

  def _get_message_from_state(self, state: State) -> list[BaseMessage]:
    messages = [message for message in state.get('messages', [])]

    if isinstance(messages[-1], AIMessage):
      original_message = messages[-1]
      messages[-1] = HumanMessage(original_message.content)

    return messages

  def execute_node(self, state: State) -> BaseState:
    messages = self._get_message_from_state(state)

    model_response = self._get_execution_chain().invoke(
      {
        'messages': messages,
      }
    )
    parsed_response = self._output_message_type(model_response.content)
    return {'messages': [parsed_response]}
