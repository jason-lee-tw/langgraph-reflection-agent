from datetime import datetime

from ai_agents.base_agent import BaseAgent
from ai_agents.langgraph.nodes.base_node import BaseAgentNode
from ai_agents.langgraph_reflexion_agent.nodes.prompts.reflexion_prompt import (
  REFLEXION_PROMPT_TEMPLATE,
)
from ai_agents.langgraph_reflexion_agent.schemas.revise_answer_question import (
  ReviseAnswerQuestion,
)
from ai_agents.langgraph_reflexion_agent.states.reflexion_agent_state import (
  ReflexionAgentState,
)
from langchain.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableSerializable
from server_config.logger import Logger


class ReviseActorNode(BaseAgentNode[ReflexionAgentState]):
  def __init__(self):
    model = BaseAgent().get_model()
    output_message_type = AIMessage

    instruction = """
    Revise your previous answer by using the new information.
      - You should use the previous critique to add important information into your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit.)
          - [1] https://example1.com
          - [2] https://example2.com
        - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words
    """  # noqa: E501

    prompt_template = REFLEXION_PROMPT_TEMPLATE.partial(
      time=lambda: datetime.now().isoformat(), first_instruction=instruction
    )

    super().__init__(
      model=model,
      prompt_template=prompt_template,
      output_message_type=output_message_type,
    )

  def _get_execution_chain(self) -> RunnableSerializable[dict, AIMessage | ToolMessage]:
    # pass the schema as a tool, so LLM will output the structured format
    # `strict` param ensure that the output format is strictly follow ReviseAnswerQuestion.  # noqa: E501
    model = self._model.bind_tools(
      tools=[ReviseAnswerQuestion],
      tool_choice='ReviseAnswerQuestion',
      strict=True,
    )

    return self._prompt_template | model

  def execute_node(self, state):
    logger = Logger(ReviseActorNode.__name__)
    logger.debug('🔄 Executing node...')

    messages = self._get_message_from_state(state)
    model_res = self._get_execution_chain().invoke(
      {
        'messages': messages,
      }
    )
    logger.debug('✅ Completed node...')
    return ReflexionAgentState(messages=[model_res])
