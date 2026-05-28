import os

from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.output_parsers import ToolsOutputParser
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import BaseGenerationOutputParser
from pydantic import BaseModel, SecretStr


class BaseAgent:
  __model: BaseChatModel

  def __init__(self):
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
      raise ValueError(f'Invalid ANTHROPIC_API_KEY: {api_key}')

    self.__model = ChatAnthropic(
      api_key=SecretStr(api_key),
      temperature=0,
      model_name='claude-sonnet-4-6',
      timeout=None,
      stop=None,
    )

  def chat(self, messages: list[BaseMessage]) -> AIMessage:
    res = self.__model.invoke(input=messages)
    message = res.content
    return AIMessage(content=message)

  def get_model(self) -> BaseChatModel:
    return self.__model

  @staticmethod
  def get_parser(
    schemas: list[type[BaseModel]], return_single: bool = False
  ) -> BaseGenerationOutputParser:
    if len(schemas) == 0:
      raise ValueError('Schemas cannot be empty list.')

    return ToolsOutputParser(
      pydantic_schemas=schemas,
      first_tool_only=return_single,
    )
