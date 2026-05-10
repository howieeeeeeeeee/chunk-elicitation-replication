from .config import *
from utils.logging import setup_logger
from .llm_service import LLMService
from .instruction_setup_rf import *
import time

logger = setup_logger(__name__)

RESPONSE_FORMATTERS_LLM_CONFIG = {
    "llm_service": "openai",
    "temperature": 1,
    "frequency_penalty": 1,
    "model": "gpt-4o",
}


class ResponseFormatter:

    def __init__(self, simulation_config, instruction_config, content):
        self.simulation_config = simulation_config
        self.instruction_config = instruction_config
        self.content = content
        self.llm_service = LLMService(RESPONSE_FORMATTERS_LLM_CONFIG)

    def _prepare_messages(self):
        sys_prompt = get_sys_prompt()

        ask_for_output = get_output_format(
            simulation_mode=self.simulation_config["simulation_mode"],
            game_type=self.simulation_config["game_type"],
            explain_reasoning=self.instruction_config.get("explain_reasoning", False),
            batch_simulation_n=self.simulation_config.get("batch_simulation_n", 5),
        )
        ask_for_response = get_ask_for_response_message(
            simulation_mode=self.simulation_config["simulation_mode"],
            batch_simulation_n=self.simulation_config.get("batch_simulation_n", 5),
        )

        messages = [
            {
                "role": "system" if "o1" not in self.llm_service.model else "assistant",
                "content": sys_prompt,
            },
            {"role": "user", "content": ask_for_output},
            {"role": "user", "content": ask_for_response},
            {
                "role": "user",
                "content": f"Here is the content that I need to format, delimited by `---`: \n---\n {self.content} \n---\n",
            },
            {
                "role": "assistant",
                "content": f"Here is the output JSON:",
            },
        ]

        return messages

    def get_formatted_output(self):
        start_time = time.time()
        try:
            messages = self._prepare_messages()
            result, usage, extra = self.llm_service.get_completion(
                messages=messages,
                llm_response_format="json",
                require_response_formatter=False,
            )

            elapsed_time = time.time() - start_time
            return True, {
                "result": result,
                "extra": extra,
                "msg": "Success",
                "elapsed_time": elapsed_time,
                "token_usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                },
            }

        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Error: {e}")
            return False, {"result": {}, "msg": str(e), "elapsed_time": elapsed_time}
