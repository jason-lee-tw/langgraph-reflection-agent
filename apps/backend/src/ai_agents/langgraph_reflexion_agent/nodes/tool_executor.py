from ai_agents.langgraph_reflexion_agent.schemas.answer_question import AnswerQuestion
from ai_agents.langgraph_reflexion_agent.schemas.revise_answer_question import (
  ReviseAnswerQuestion,
)
from ai_agents.tavily.tavily_client import TavilyClient
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode


class ToolExecutorNode:
  def __init__(self):
    super().__init__()

  # `search_queries` must be same name with the param in
  # the object (AnswerQuestion / ReviseAnswerQuestion) being passed into this ToolNode.
  # In this case, `search_queries` in `AnswerQuestion` will be passed into
  # this tool by ToolNode.
  def search_from_web(self, search_queries: list[str]):
    """Search from internet by using the generated search queries."""
    searcher = TavilyClient().searcher
    searcher.max_results = 5

    return searcher.batch(inputs=search_queries)  # type: ignore

  def get_node(self) -> ToolNode:
    return ToolNode(
      tools=[
        StructuredTool.from_function(
          self.search_from_web, name=AnswerQuestion.__name__
        ),
        StructuredTool.from_function(
          self.search_from_web, name=ReviseAnswerQuestion.__name__
        ),
      ],
    )
