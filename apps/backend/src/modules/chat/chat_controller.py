from json import dumps

from fastapi import APIRouter
from modules.chat.chat_service import (
  process_chat,
  process_chat_with_graph,
  process_reflexion_chat,
)
from modules.chat.dto.chat_dto import ChatHistoryDTO, ChatReqDTO, ChatResDTO

router = APIRouter(prefix='/chat')


@router.post('/')
def handle_chat(body: ChatReqDTO) -> ChatResDTO:
  message = body.message
  history = body.history

  chat_list = [chat for chat in history]
  chat_list.append(ChatHistoryDTO(role='user', content=message))

  result = process_chat(chat_list)

  return ChatResDTO(message=result.content, history=chat_list)


@router.post('/graph')
def handle_chat_with_graph(body: ChatReqDTO) -> ChatResDTO:
  message = body.message
  history = body.history

  chat_list = [chat for chat in history]
  chat_list.append(ChatHistoryDTO(role='user', content=message))

  result = process_chat_with_graph(chat_list)

  return ChatResDTO(message=result.content, history=chat_list)


@router.post('/reflexion')
def handle_reflexion_chat(body: ChatReqDTO) -> ChatResDTO:
  message = body.message
  history = body.history

  chat_list = [chat for chat in history]
  chat_list.append(ChatHistoryDTO(role='user', content=message))

  result = process_reflexion_chat(chat_list)
  res_message = __extract_message_content(result.content)

  return ChatResDTO(message=res_message, history=chat_list)


def __extract_message_content(obj: str | dict | list[dict]) -> str:
  if not isinstance(obj, list) and not isinstance(obj, dict):
    return str(obj)

  return dumps(obj)
