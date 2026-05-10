from .config import *
from .instruction_setup import *
from utils.validation import validate_input
from utils.logging import setup_logger
from .llm_service import LLMService
from .response_formatter import ResponseFormatter
import time

logger = setup_logger(__name__)


class EconomicAgent:
    ALLOWED_SIMULATION_MODES = {"IterativeSimulate", "BatchSimulate"}
    ALLOWED_BATCH_MODES = {"Independent", "Appending"}
    ALLOWED_EXPLAIN_REASONING_MODES = {"basic", "split"}

    def __init__(self, simulation_config, instruction_config, llm_config):
        self.simulation_config = simulation_config
        self.instruction_config = instruction_config
        self.llm_config = llm_config
        self.llm_service = LLMService(llm_config)
        self.response_formatter = None
        self._split_messages = None
        self._split_chunks_done = 0
        self._split_total_chunks = 0
        self._validate_configs()

    def _validate_configs(self):
        validate_input(
            self.simulation_config["simulation_mode"],
            self.ALLOWED_SIMULATION_MODES,
            "simulation_mode",
        )
        validate_input(
            self.simulation_config["game_type"], GAME_DESCRIPTION, "game_type"
        )
        validate_input(
            self.simulation_config.get("batch_mode", "Independent"),
            self.ALLOWED_BATCH_MODES,
            "batch_mode",
        )
        explain_reasoning_mode = self.instruction_config.get(
            "explain_reasoning_mode", "basic"
        )
        validate_input(
            explain_reasoning_mode,
            self.ALLOWED_EXPLAIN_REASONING_MODES,
            "explain_reasoning_mode",
        )
        if explain_reasoning_mode == "split":
            split_n = self.instruction_config.get("split_n")
            batch_simulation_n = self.simulation_config.get("batch_simulation_n")
            if split_n is None:
                raise ValueError(
                    "split_n is required when explain_reasoning_mode='split'"
                )
            if not isinstance(split_n, int) or split_n <= 0:
                raise ValueError("split_n must be a positive integer")
            if not self.instruction_config.get("explain_reasoning", False):
                raise ValueError(
                    "explain_reasoning_mode='split' requires explain_reasoning=True"
                )
            if self.simulation_config["simulation_mode"] != "BatchSimulate":
                raise ValueError(
                    "explain_reasoning_mode='split' only supports BatchSimulate"
                )
            if self.simulation_config.get("batch_mode", "Independent") != "Independent":
                raise ValueError(
                    "explain_reasoning_mode='split' requires batch_mode='Independent'"
                )
            if not isinstance(batch_simulation_n, int) or batch_simulation_n <= 0:
                raise ValueError(
                    "batch_simulation_n must be a positive integer for split mode"
                )
            if batch_simulation_n % split_n != 0:
                raise ValueError(
                    f"split_n ({split_n}) must evenly divide batch_simulation_n "
                    f"({batch_simulation_n})"
                )
            self._split_total_chunks = batch_simulation_n // split_n

    def _prepare_messages(self):
        sys_prompt = get_sys_prompt(
            simulation_mode=self.simulation_config["simulation_mode"],
            background=self.instruction_config.get("background", ""),
            personality_traits=self.instruction_config.get("personality_traits", ""),
            batch_simulation_n=self.simulation_config.get("batch_simulation_n", 5),
            include_simulation_id=self.instruction_config.get(
                "include_simulation_id", True
            ),
            explain_reasoning_mode=self.instruction_config.get(
                "explain_reasoning_mode", "basic"
            ),
            split_n=self.instruction_config.get("split_n"),
        )

        instructions = get_game_instructions(
            game_type=self.simulation_config["game_type"],
            simulation_mode=self.simulation_config["simulation_mode"],
            theoretical_prediction=self.instruction_config.get(
                "theoretical_prediction", False
            ),
            focal_point=self.instruction_config.get("focal_point", False),
            context=self.instruction_config.get("context", ""),
            incentive_size=self.instruction_config.get("incentive_size", ""),
            privacy_treatment=self.instruction_config.get("privacy_treatment", ""),
            explain_reasoning=self.instruction_config.get("explain_reasoning", False),
            additional_instructions=self.instruction_config.get(
                "additional_instructions", []
            ),
        )

        messages = [
            {
                "role": "system" if "o1" not in self.llm_service.model else "assistant",
                "content": sys_prompt,
            },
            {
                "role": "user",
                "content": f"Here is the instruction for the game play, delimited by `---`: \n---\n {instructions} \n---\n",
            },
        ]

        output_format = self.simulation_config.get("output_format", "json")
        ask_for_output = get_output_format(
            simulation_mode=self.simulation_config["simulation_mode"],
            game_type=self.simulation_config["game_type"],
            explain_reasoning=self.instruction_config.get("explain_reasoning", False),
            batch_simulation_n=self.simulation_config.get("batch_simulation_n", 5),
            output_format=output_format,
            explain_reasoning_mode=self.instruction_config.get(
                "explain_reasoning_mode", "basic"
            ),
            split_n=self.instruction_config.get("split_n"),
        )
        messages.append({"role": "user", "content": ask_for_output})

        if self.simulation_config.get("batch_mode") == "Appending":
            self._append_previous_responses(messages)

        ask_for_response = get_ask_for_response_message(
            simulation_mode=self.simulation_config["simulation_mode"],
            batch_simulation_n=self.simulation_config.get("batch_simulation_n", 5),
        )
        messages.extend(
            [
                {"role": "user", "content": ask_for_response},
                {
                    "role": "assistant",
                    "content": f"Here is the output in the required format:",
                },
            ]
        )

        return messages

    def _append_previous_responses(self, messages):
        for response in self.simulation_config.get("previous_responses", []):
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": get_ask_for_response_message(
                            self.simulation_config["simulation_mode"],
                            self.simulation_config.get("batch_simulation_n", 5),
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": f"Here is the output in the required format:",
                    },
                    {"role": "assistant", "content": str(response)},
                ]
            )

    def reset_split_thread(self):
        self._split_messages = None
        self._split_chunks_done = 0

    def _prepare_split_initial_messages(self):
        messages = self._prepare_messages()
        # `_prepare_messages` ends with the batch "please make the choice..." turn
        # plus the assistant priming line. For split mode we replace the final
        # user turn with the chunk-ask so the first call only asks for split_n.
        split_n = self.instruction_config["split_n"]
        batch_simulation_n = self.simulation_config["batch_simulation_n"]
        chunk_ask = get_split_chunk_ask_message(
            split_n=split_n,
            chunk_index=1,
            total_chunks=self._split_total_chunks,
            batch_simulation_n=batch_simulation_n,
        )
        # Find and replace the last user turn (the ask-for-response one).
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                messages[i] = {"role": "user", "content": chunk_ask}
                break
        return messages

    def get_decisions_split_chunk(self):
        start_time = time.time()
        try:
            if self.instruction_config.get("explain_reasoning_mode") != "split":
                raise ValueError(
                    "get_decisions_split_chunk() requires explain_reasoning_mode='split'"
                )
            if self._split_chunks_done >= self._split_total_chunks:
                raise ValueError(
                    "All split chunks have already been produced for this thread; "
                    "call reset_split_thread() before starting a new super-session."
                )

            split_n = self.instruction_config["split_n"]
            batch_simulation_n = self.simulation_config["batch_simulation_n"]

            if self._split_messages is None:
                self._split_messages = self._prepare_split_initial_messages()
            else:
                chunk_ask = get_split_chunk_ask_message(
                    split_n=split_n,
                    chunk_index=self._split_chunks_done + 1,
                    total_chunks=self._split_total_chunks,
                    batch_simulation_n=batch_simulation_n,
                )
                self._split_messages.extend(
                    [
                        {"role": "user", "content": chunk_ask},
                        {
                            "role": "assistant",
                            "content": "Here is the output in the required format:",
                        },
                    ]
                )

            result, usage, extra = self.llm_service.get_completion(
                messages=self._split_messages,
                return_raw_response=self.llm_config.get("return_raw_response", False),
                save_messages_n_contents=self.simulation_config.get(
                    "save_messages_n_contents", False
                ),
                return_all_api_responses=self.llm_config.get(
                    "return_all_api_responses", False
                ),
            )

            self._split_messages.append({"role": "assistant", "content": str(result)})
            self._split_chunks_done += 1

            elapsed_time = time.time() - start_time
            return True, {
                "result": result,
                "extra": extra,
                "msg": "Success",
                "elapsed_time": elapsed_time,
                "split_chunk_index": self._split_chunks_done,
                "split_total_chunks": self._split_total_chunks,
                "token_usage": {
                    "raw_usage": str(usage),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                },
            }
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Error (split chunk): {e}")
            return False, {
                "result": {},
                "msg": str(e),
                "elapsed_time": elapsed_time,
                "split_chunk_index": self._split_chunks_done,
                "split_total_chunks": self._split_total_chunks,
            }

    def get_decisions(self):
        start_time = time.time()
        try:
            messages = self._prepare_messages()
            require_response_formatter = (
                self.simulation_config.get("output_format", "json") == "free_response"
            )  ## deprecated
            result, usage, extra = self.llm_service.get_completion(
                messages=messages,
                return_raw_response=self.llm_config.get("return_raw_response", False),
                # require_response_formatter=require_response_formatter,
                save_messages_n_contents=self.simulation_config.get(
                    "save_messages_n_contents", False
                ),
                return_all_api_responses=self.llm_config.get(
                    "return_all_api_responses", False
                ),
            )
            if require_response_formatter:
                self.response_formatter = ResponseFormatter(
                    simulation_config=self.simulation_config,
                    instruction_config=self.instruction_config,
                    content=result,
                )
                success, formatted_output = (
                    self.response_formatter.get_formatted_output()
                )
                if success:
                    result = formatted_output["result"]
                    extra["formatted_output"] = formatted_output
                else:
                    raise Exception(formatted_output["msg"])

            elapsed_time = time.time() - start_time
            return True, {
                "result": result,
                "extra": extra,
                "msg": "Success",
                "elapsed_time": elapsed_time,
                "token_usage": {
                    "raw_usage": str(usage),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                },
            }

        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Error: {e}")
            return False, {"result": {}, "msg": str(e), "elapsed_time": elapsed_time}
