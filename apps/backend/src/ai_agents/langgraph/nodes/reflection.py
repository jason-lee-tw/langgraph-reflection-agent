from ai_agents.base_agent import BaseAgent
from ai_agents.langgraph.nodes.base_node import BaseAgentNode
from langchain.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class ReflectionNode(BaseAgentNode):
  def __init__(self):
    prompt_template = ChatPromptTemplate(
      [
        SystemMessage(
          """You are a viral twitter influencer grading a tweet. Generate critique and recommandations for the user's tweet.
          Always provide detailed recommendations, including requests for length, virality, style, etc.
          If you think the tweet is perfect, then you should just response with "PERFECT_TWEET".
          """,  # noqa: E501
        ),
        MessagesPlaceholder(variable_name='messages'),
      ]
    )
    model = BaseAgent().get_model()
    output_message_type = AIMessage

    super().__init__(
      model=model,
      prompt_template=prompt_template,
      output_message_type=output_message_type,
    )

  def _get_execution_chain(self):
    return self._prompt_template | self._model
