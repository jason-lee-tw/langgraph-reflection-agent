from ai_agents.base_agent import BaseAgent
from ai_agents.langgraph.flow import get_flow
from fastapi import HTTPException
from langchain_core.messages import (
  AIMessage,
  BaseMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
)
from langgraph.graph import MessagesState
from modules.chat.dto.chat_dto import ChatHistoryDTO
from server_config.logger import Logger


def _convert_chat_list(chat_list: list[ChatHistoryDTO]) -> list[BaseMessage]:
  """Convert chat history list to proper message list for LLM"""
  result: list[BaseMessage] = []

  for chat in chat_list:
    role = chat.role
    content = chat.content
    message: BaseMessage

    if role == 'user':
      message = HumanMessage(content=content)
    elif role == 'system':
      message = SystemMessage(content=content)
    elif role == 'assistant':
      message = AIMessage(content=content)
    elif role == 'tool':
      message = ToolMessage(content=content)
    else:
      raise HTTPException(
        status_code=400, detail=f'Invalid role `{role}` in chat history.'
      )

    result.append(message)

  return result


def process_chat(chat_list: list[ChatHistoryDTO]) -> AIMessage:
  agent = BaseAgent()

  messages = _convert_chat_list(chat_list)
  agent_res = agent.chat(messages=messages)

  return agent_res


def process_chat_with_graph(chat_list: list[ChatHistoryDTO]) -> AIMessage:
  logger = Logger(__name__)
  messages = _convert_chat_list(chat_list)
  flow = get_flow().compile()
  state = MessagesState(messages=messages)

  res = flow.invoke(state)
  res_messages = res.get('messages')

  if not isinstance(res_messages, list) or not len(res_messages):
    logger.error(f'Invalid response message list.\nResponse:\n{res}')
    raise ValueError('Missing response message from LLM model.')

  latest_message = res_messages[-1]

  if not isinstance(latest_message, AIMessage):
    logger.error(f'Invalid last message. \nMessage:\n{latest_message}')
    raise ValueError('Invalid latest message object.')

  return latest_message
