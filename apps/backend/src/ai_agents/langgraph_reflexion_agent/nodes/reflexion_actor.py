from datetime import datetime

from ai_agents.base_agent import BaseAgent
from ai_agents.langgraph.nodes.base_node import BaseAgentNode
from ai_agents.langgraph_reflexion_agent.nodes.prompts.reflexion_prompt import (
  REFLEXION_PROMPT_TEMPLATE,
)
from ai_agents.langgraph_reflexion_agent.schemas.answer_question import AnswerQuestion
from ai_agents.langgraph_reflexion_agent.states.reflexion_agent_state import (
  ReflexionAgentState,
)
from langchain.messages import AIMessage
from langchain_core.runnables import RunnableSerializable


class ReflexionActorNode(BaseAgentNode[ReflexionAgentState]):
  def __init__(self):
    model = BaseAgent().get_model()
    output_message_type = AIMessage
    prompt_template = REFLEXION_PROMPT_TEMPLATE.partial(
      time=lambda: datetime.now().isoformat(),
    )

    super().__init__(
      model=model,
      prompt_template=prompt_template,
      output_message_type=output_message_type,
    )

  def _get_execution_chain(self) -> RunnableSerializable[dict, AnswerQuestion]:
    parser = BaseAgent.get_parser(schemas=[AnswerQuestion], return_single=True)

    actor_prompt_template = self._prompt_template.partial(
      first_instruction='Provide a deailed answer with max 250 words.',
    )
    model = self._model.bind_tools(
      tools=[AnswerQuestion],
      tool_choice='AnswerQuestion',
      strict=True,
    )

    return actor_prompt_template | model | parser

  def execute_node(self, state):
    messages = self._get_message_from_state(state)

    model_res = self._get_execution_chain().invoke(
      {
        'messages': messages,
      }
    )
    model_res_json = {
      'type': 'AnswerQuestion',
      'response': model_res.model_dump_json(),
    }
    parsed_res = self._output_message_type(content=[model_res_json])
    return ReflexionAgentState(messages=[parsed_res])
