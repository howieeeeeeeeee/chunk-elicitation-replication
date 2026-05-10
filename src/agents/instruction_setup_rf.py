from games.instructions import GAME_DECISION_ARRAY

SYS_PROMPT = """You are an expert in formatting the output of a LLM. \
You will be given a response from a LLM, and you need to format the response into a specific JSON format. \
You will be given the original response, and the format you need to format the response into. \
You need to return the formatted response. \
"""


def get_sys_prompt():
    return SYS_PROMPT


def get_output_format(
    simulation_mode,
    game_type,
    explain_reasoning,
    batch_simulation_n,
):

    decision_output = GAME_DECISION_ARRAY[game_type]

    if explain_reasoning:
        minimal_response_format = f"""[
        {decision_output},
        "reasoning process in 50 words"
    ]"""
    else:
        minimal_response_format = f"""{decision_output}"""

    if simulation_mode == "IterativeSimulate":
        format_str = (
            "Please return the decision as a JSON object with the following structure:\n"
            "```json\n"
            "{\n"
            f'    "response": {minimal_response_format}\n'
            "}\n"
            "```\n"
        )

    elif simulation_mode == "BatchSimulate":
        format_str = (
            f"Please return the decisions{batch_simulation_n} times as a JSON object with the following structure:\n"
            "```json\n"
            "{\n"
            f'   "number_of_decisions": {batch_simulation_n},\n'
            '    "all_responses": [\n'
            f"       {minimal_response_format},\n"
            f"       ...(repeated for {batch_simulation_n} decisions, if exist),\n"
            "    ]\n"
            "}\n"
            "```\n"
            f"(ideally, there will be {batch_simulation_n} decisions, if not, just return the number of decisions you can find)"
        )
    return format_str


def get_ask_for_response_message(simulation_mode, batch_simulation_n):
    ask_for_response_str = ""
    if simulation_mode == "IterativeSimulate":
        ask_for_response_str += (
            f"Show me the game decision in the required format, thanks!"
        )
    elif simulation_mode == "BatchSimulate":
        ask_for_response_str += f"Please make sure to reformat the output into the required format. Ideally, there will be {batch_simulation_n} decisions. If not, please make sure to return the correct number of decisions, thanks!"
    return ask_for_response_str
