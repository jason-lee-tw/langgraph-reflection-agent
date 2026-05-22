from datetime import datetime
from pathlib import Path

from ai_agents.langgraph.flow import get_flow
from server_config.logger import Logger


def generate_langgraph_diagram() -> str:
  logger = Logger(__name__)

  app = get_flow().compile()

  timestamp = datetime.now().isoformat()
  folder_path_obj = Path(__file__).parent.joinpath('../../../temp/graph/').resolve()
  file_name = f'flow-{"-".join(timestamp.split(":"))}.png'
  full_path = str(folder_path_obj.joinpath(file_name))
  logger.log(f'Graph image saved into {full_path}')

  # Create folder if not exist
  folder_path_obj.mkdir(parents=True, exist_ok=True)

  # Draw graph
  app.get_graph().draw_mermaid_png(output_file_path=full_path)

  return full_path
