"""Configuration class for xLAM client."""
class xLAMConfig:
    r"""
    Configuration class for xLAM client.

    Args:
        base_url (`str`):
            The base URL for the chat completion endpoint.
        model (`str`):
            The model name for within the xLAM series.

    Attributes:
        BASE_URL (`str`):
            The base URL for API requests.
        MODEL_NAME (`str`):
            The name of the xLAM model.
        TASK_INSTRUCTION (`str`):
            Instructions defining the task for the AI assistant.
        FORMAT_INSTRUCTION (`str`):
            Instructions on how to format the output.
    """
    TASK_INSTRUCTION = """
    Based on the previous context and API request history, generate an API request or a response as an AI assistant.""".strip()

    FORMAT_INSTRUCTION = """
    The output should be of the JSON format, which specifies a list of generated function calls. The example format is as follows, please make sure the parameter type is correct. If no function call is needed, please make 
    tool_calls an empty list "[]".
    ```
    {"thought": "the thought process, or an empty string", "tool_calls": [{"name": "api_name1", "arguments": {"argument1": "value1", "argument2": "value2"}}]}
    ```
    """.strip()
    
    def __init__(self, base_url: str, model: str):
        self.BASE_URL = base_url
        self.MODEL_NAME = model

