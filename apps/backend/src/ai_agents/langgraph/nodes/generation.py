from ai_agents.base_agent import BaseAgent
from ai_agents.langgraph.nodes.base_node import BaseAgentNode
from langchain.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class GenerationNode(BaseAgentNode):
  def __init__(self):
    prompt_template = ChatPromptTemplate(
      [
        SystemMessage(
          """You are a twitter techie influencer assistant tasked with writing excellent twitter posts.
          Generate the best twitter post possible for the user's request.
          If the user provides critique, response with a revised version of your previous attempts.
          
          ---

          Just response the revised tweet without any other messages.
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
