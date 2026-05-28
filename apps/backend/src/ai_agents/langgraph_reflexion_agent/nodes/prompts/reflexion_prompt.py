from langchain.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

__SYSTEM_PROMPT = SystemMessage(
  """You are an expert researcher.
  Current Time: {time}

  1. {first_instruction}
  2. Reflect and critique your answer. Be severe to maximum improvement.
  3. Recommend search queries to research information and improve your answer.
  """,
)

REFLEXION_PROMPT_TEMPLATE = ChatPromptTemplate(
  [
    __SYSTEM_PROMPT,
    SystemMessage(
      """Answer the user's question below using the required format.""",
    ),
    MessagesPlaceholder(variable_name='messages'),
  ]
)
