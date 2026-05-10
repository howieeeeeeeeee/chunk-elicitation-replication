def validate_input(value, allowed_values, name):
    if value not in allowed_values:
        raise ValueError(
            f"Invalid {name}: {value}. Allowed values are: {allowed_values}"
        )
