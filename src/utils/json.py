import re
import json

def extract_json(string):
    # Find JSON code block
    match = re.search(r"```json(.+?)```", string, re.DOTALL)
    if match:
        json_string = match.group(1).strip()  # Extract the JSON string
        try:
            # Parse the JSON string
            json_object = json.loads(json_string)
            return [json_object]
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return []
    else:
        try:
            content = eval(string)
            if isinstance(content, list):
                return content
            else:
                return [content]
        except:
            return []
