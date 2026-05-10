# from .import_setup import *
from games.instructions import *
from games.schemas import (
    GAME_DECISION_ARRAY_SCHEMA,
    GAME_DECISION_ARRAY_WITH_REASONING_SCHEMA,
)


def run_schema_check(data, simulation_config, instruction_config):
    # Determine which schema to use based on the instruction config
    game_type = simulation_config["game_type"]
    if instruction_config.get("explain_reasoning", False):
        schema_to_use = GAME_DECISION_ARRAY_WITH_REASONING_SCHEMA[game_type]
    else:
        schema_to_use = GAME_DECISION_ARRAY_SCHEMA[game_type]

    # Get the simulation mode
    simulation_mode = simulation_config.get("simulation_mode")
    batch_simulation_n = simulation_config.get("batch_simulation_n", 0)
    explain_reasoning_mode = instruction_config.get("explain_reasoning_mode", "basic")
    split_n = instruction_config.get("split_n")

    # Function to check if the data matches the schema
    def validate_data(data_to_validate, schema):
        try:
            schema.validate(data_to_validate)
            return True, "Data is valid."
        except Exception as e:
            return False, str(e)

    # If the simulation mode is BatchSimulate, validate each item in 'all_responses'
    if simulation_mode == "BatchSimulate":
        all_responses = data.get("all_responses", [])
        if explain_reasoning_mode == "split":
            if not isinstance(split_n, int) or split_n <= 0:
                return False, "split_n must be a positive integer in split mode."
            if len(all_responses) != split_n:
                return (
                    False,
                    f"Split mode: 'all_responses' length ({len(all_responses)}) must equal split_n ({split_n}).",
                )
        elif len(all_responses) < batch_simulation_n:
            return (
                False,
                f"The length of 'all_responses' ({len(all_responses)}) is less than 'batch_simulation_n' ({batch_simulation_n}).",
            )
        for response in all_responses:
            is_valid, message = validate_data(response, schema_to_use)
            if not is_valid:
                return False, message
        return True, "All responses are valid."

    # If the simulation mode is IterativeSimulate, validate the data directly
    elif simulation_mode == "IterativeSimulate":
        return validate_data(data["response"], schema_to_use)

    # If the simulation mode is not recognized, return an error
    else:
        return False, "Invalid simulation mode."


def post_process_simulated_data(data, simulation_config, instruction_config):
    ## run schema check
    ...
