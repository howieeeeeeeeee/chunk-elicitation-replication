from games.instructions import (
    GAME_DECISION_ARRAY,
    GAME_DECISION_ARRAY_DESCRIPTION,
    GAME_DESCRIPTION,
    GAME_FOCAL_POINTS,
    GAME_THEORETICAL_PREDICTIONS,
)
from games.treatments import (
    CONTEXT_TREATMENTS,
    INCENTIVE_SIZES,
    PRIVACY_TREATMENTS,
)
from utils.validation import validate_input
import uuid

SYS_PROMPT = """You will participate in an experiment and simulate the behavior of the particiapnt(s).  \
The user will give you the instructions, including the detailed game instructions, the decision(s) you need to make, and your other setting. \
Please follow the instructions and return the decision(s) in the required format."""


def get_sys_prompt(
    simulation_mode,
    background,
    personality_traits,
    batch_simulation_n=0,
    include_simulation_id=True,
    explain_reasoning_mode="basic",
    split_n=None,
):

    sys_prompt = SYS_PROMPT
    if include_simulation_id:
        sys_prompt += f"(Simulation ID: {str(uuid.uuid4())})"  ## Add simulation ID to bypass the cache layer from openai

    if simulation_mode == "BatchSimulate" and batch_simulation_n:
        if explain_reasoning_mode == "split" and split_n:
            sys_prompt += (
                f"\n***Important:*** You need to follow the instructions and play the game **independently** for {batch_simulation_n} times in total. "
                f"However, you will NOT produce all {batch_simulation_n} decisions at once. "
                f"Instead, I will ask you in chunks of {split_n} decisions at a time. Each time, I will tell you how many to generate for that chunk. "
                f"Treat each chunk as a continuation of the same experiment — do not repeat prior answers verbatim, and keep the variation across the full {batch_simulation_n} participants in mind.\n"
            )
        else:
            sys_prompt += (
                f"\n***Important:*** You need to follow the instructions and play the game **independently** for {batch_simulation_n} times. "
                f"Simulate the allocation decisions of {batch_simulation_n} participants, considering the variations among different individuals. "
                f"Ensure that all {batch_simulation_n} decisions are returned together in the required format.\n"
            )

    if background:
        sys_prompt += f"\n During the game play, the participant(s) have the following background: {background} \n"
    if personality_traits:
        sys_prompt += f"\n During the game play, the participant(s) have the following personlaity and trait: {personality_traits} \n"

    return sys_prompt


def get_output_format(
    simulation_mode,
    game_type,
    explain_reasoning,
    batch_simulation_n,
    output_format="json",
    explain_reasoning_mode="basic",
    split_n=None,
):

    decision_output = GAME_DECISION_ARRAY[game_type] if output_format == "json" else ""
    decision_output_description = (
        GAME_DECISION_ARRAY_DESCRIPTION[game_type] if output_format == "json" else ""
    )
    decision_output_len = len(decision_output_description)

    if explain_reasoning:
        if output_format == "json":
            minimal_response_format = f"""[
        {decision_output},
        "reasoning process in 50 words"
    ]"""
        elif output_format == "yaml":
            minimal_response_format = """- THE_DECISION_ARRAY
    - "reasoning process in 50 words"" """
        elif output_format == "free_response":
            minimal_response_format = """THE_DECISION_ARRAY and the reasoning process for each decision in 50 words"""
    else:
        if output_format == "json":
            minimal_response_format = f"""{decision_output}"""
        else:
            minimal_response_format = """the decision(s)"""

    if simulation_mode == "IterativeSimulate":
        if output_format == "json":
            format_str = (
                "Please return the decision you make as a JSON object with the following structure:\n"
                "```json\n"
                "{\n"
                f'    "response": {minimal_response_format}\n'
                "}\n"
                "```\n"
                "Make sure the response is a valid JSON object, and the first element in the response is also an array."
            )
        elif output_format == "yaml":
            format_str = (
                "Please return the decision you make as a YAML object with the following structure:\n"
                "```yaml\n"
                "response:\n"
                f"  {minimal_response_format}\n"
                "```\n"
            )
        elif output_format == "free_response":
            format_str = "Please return the decision you make as a free response with no specific format,\
                      but make sure to include all the required information, that is, {minimal_response_format}\n"

    elif simulation_mode == "BatchSimulate":
        if output_format == "json":
            format_str = (
                f"Please return the decisions you make {batch_simulation_n} times as a JSON object with the following structure:\n"
                "```json\n"
                "{\n"
                f'   "number_of_decisions": {batch_simulation_n},\n'
                '    "all_responses": [\n'
                f"       {minimal_response_format},\n"
                f"       ...(repeated for {batch_simulation_n} decisions),\n"
                "    ]\n"
                "}\n"
                "```\n"
                f"Make sure the `all_responses` array is a two-dimensional array with {batch_simulation_n} rows and {decision_output_len} columns.\n"
            )
        elif output_format == "yaml":
            format_str = (
                f"Please return the decisions you make {batch_simulation_n} times as a YAML object with the following structure:\n"
                "```yaml\n"
                "number_of_decisions: {batch_simulation_n}\n"
                "all_responses:\n"
                f"  - {minimal_response_format}\n"
                f"  - ... (repeated for {batch_simulation_n} decisions)\n"
                "```\n"
                f"Make sure the `all_responses` array is a two-dimensional array with {batch_simulation_n} rows and {decision_output_len} columns.\n"
            )
        elif output_format == "free_response":
            format_str = f"Please return the decisions you make for {batch_simulation_n} times, try to make the output as concise as possible.\
                    Make sure to include all the required information, that is, {minimal_response_format}\n"

        if explain_reasoning_mode == "split" and split_n:
            format_str += (
                f"\nIMPORTANT — split mode: On this call, only produce **{split_n}** decisions "
                f"(not {batch_simulation_n}). `all_responses` must contain exactly {split_n} entries. "
                f"I will ask you for the remaining chunks in follow-up turns until all {batch_simulation_n} decisions are collected.\n"
            )

    return format_str


def get_game_instructions(
    game_type,
    simulation_mode,
    theoretical_prediction,
    explain_reasoning,
    focal_point,
    context,
    incentive_size,
    privacy_treatment,
    additional_instructions,
):

    instruction_str = GAME_DESCRIPTION.get(game_type)

    instruction_str += (
        " Please give only one concrete choice."
        if simulation_mode == "IterativeSimulate"
        else " Please make the choice for this game INDEPENDENTLY for the required times."
    )

    if (
        theoretical_prediction
        or explain_reasoning
        or additional_instructions
        or focal_point
        or context
        or incentive_size
        or privacy_treatment
    ):
        instruction_str += (
            "\n Besides, here are some additional info you need to know: \n"
        )

    if theoretical_prediction:
        validate_input(game_type, GAME_THEORETICAL_PREDICTIONS, "game_type")
        if GAME_THEORETICAL_PREDICTIONS[game_type]:
            instruction_str += f"    - The theorotical prediction of this game is: {GAME_THEORETICAL_PREDICTIONS[game_type]} \n"
    if explain_reasoning:
        instruction_str += f"    - I need you to provide your reasoning process of making the decision(s). \n"
    if focal_point:
        validate_input(game_type, GAME_FOCAL_POINTS, "game_type")
        if GAME_FOCAL_POINTS[game_type]:
            instruction_str += f"    - When making your decisions, you may notice certain choices that naturally stand out. In similar situations, people often gravitate toward particular decisions even without discussing them with others: {GAME_FOCAL_POINTS[game_type]} \n"
    if context:
        validate_input(context, CONTEXT_TREATMENTS, "context")
        if CONTEXT_TREATMENTS[context]:
            instruction_str += f"    - {CONTEXT_TREATMENTS[context]} \n"
    if incentive_size and context:
        validate_input(incentive_size, INCENTIVE_SIZES, "incentive_size")
        validate_input(context, INCENTIVE_SIZES[incentive_size], "context")
        if INCENTIVE_SIZES[incentive_size][context]:
            instruction_str += f"    - {INCENTIVE_SIZES[incentive_size][context]} \n"
    if privacy_treatment and context:
        validate_input(privacy_treatment, PRIVACY_TREATMENTS, "privacy_treatment")
        if PRIVACY_TREATMENTS[privacy_treatment]:
            instruction_str += f"    - {PRIVACY_TREATMENTS[privacy_treatment]} \n"

    if additional_instructions:
        for i in additional_instructions:
            instruction_str += f"   - {i}\n"
    return instruction_str


def get_ask_for_response_message(simulation_mode, batch_simulation_n):
    ask_for_response_str = ""
    if simulation_mode == "IterativeSimulate":
        ask_for_response_str += (
            f"Show me the game decision in the required format, thanks!"
        )
    elif simulation_mode == "BatchSimulate":
        ask_for_response_str += f"Please make the choice for this game INDEPENDENTLY for the {batch_simulation_n} times. And show me the game decisions in the required format, thanks!"
    return ask_for_response_str


def get_split_chunk_ask_message(split_n, chunk_index, total_chunks, batch_simulation_n):
    return (
        f"Now please produce the next {split_n} independent decisions "
        f"(chunk {chunk_index}/{total_chunks} out of the total {batch_simulation_n} participants). "
        f"Make sure `all_responses` contains exactly {split_n} entries, each with the decision array and a short reasoning string. "
        f"Do not repeat the exact same decisions from earlier chunks — continue to reflect the variation across participants."
    )
