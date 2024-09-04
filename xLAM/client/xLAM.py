import json
from openai import OpenAI
from xLAM.client.config import xLAMConfig

class xLAMChatCompletion:
    r"""
    A class for handling chat completions using the xLAM model.

    Args:
        base_url (`str`):
            The base URL for the API endpoint.
        model_name (`str`):
            The name of the xLAM model to use.
        task_instruction (`str`):
            Instructions defining the task for the model.
        format_instruction (`str`):
            Instructions on how to format the output.

    Attributes:
        model_name (`str`):
            The name of the xLAM model to use.
        client (`OpenAI`):
            An OpenAI client instance for making API calls.
        task_instruction (`str`):
            Instructions defining the task for the model.
        format_instruction (`str`):
            Instructions on how to format the output.

    Methods:
        from_config(`xLAMConfig`):
            Class method to create an instance from an xLAMConfig object.
        completion(`List[Dict[str, str]]`, `Optional[List[Dict[str, Any]]]`, `**kwargs`):
            Generate a chat completion based on provided messages and tools.
    """

    def __init__(
        self, 
        base_url: str,
        model: str,
        task_instruction: str=xLAMConfig.TASK_INSTRUCTION, 
        format_instruction: str=xLAMConfig.FORMAT_INSTRUCTION
    ):
        self.model_name = model
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.task_instruction = task_instruction
        self.format_instruction = format_instruction
    
    @classmethod
    def from_config(cls, config: xLAMConfig):
        return cls(
            model=config.MODEL_NAME,
            base_url=config.BASE_URL,
            task_instruction=config.TASK_INSTRUCTION,
            format_instruction=config.FORMAT_INSTRUCTION
        )
    
    def completion(self, messages, tools=None, **kwargs):
        # Convert OpenAI-style functions to xLAM format
        if messages[0]['role'] == 'system':
            system_message = messages[0]
            messages = messages[1:]
        else:
            system_message = None
            
        xlam_tools = self.convert_to_xlam_tool(tools) if tools else []
        
        # Extract the user's query (last user message)
        query = next((msg['content'] for msg in reversed(messages) if msg['role'] == 'user'), "")
        
        # Build conversation history from messages
        conversation_history = self.build_conversation_history(messages)
        
        # Build the prompt
        content = self.build_prompt(self.task_instruction, self.format_instruction, xlam_tools, query, conversation_history, system_message)
        
        # Generate response
        inputs = [{'role': 'user', 'content': content}]
        outputs = self.client.chat.completions.create(messages=inputs, model=self.model_name, **kwargs)
        agent_action = outputs.choices[0].message.content
        
        # Parse the response
        thought, tool_calls = self.parse_agent_action(agent_action)
        
        return {
            'choices': [{
                'message': {
                    'role': 'assistant',
                    'content': thought,
                    'tool_calls': tool_calls if tool_calls else []
                }
            }]
        }

    def convert_to_xlam_tool(self, tools):
        '''Convert OpenAPI-specification functions to xLAM format'''
        if isinstance(tools, dict):
            return {
                "name": tools["name"],
                "description": tools["description"],
                "parameters": {k: v for k, v in tools["parameters"].get("properties", {}).items()}
            }
        elif isinstance(tools, list):
            return [self.convert_to_xlam_tool(tool) for tool in tools]
        else:
            return tools

    def build_prompt(self, task_instruction: str, format_instruction: str, tools: list, query: str, conversation_history: list, system_message: str):
        if system_message:
            prompt = f"{system_message}\n\n"
        else:
            prompt = ""
        prompt += f"[BEGIN OF TASK INSTRUCTION]\n{task_instruction}\n[END OF TASK INSTRUCTION]\n\n"
        prompt += f"[BEGIN OF AVAILABLE TOOLS]\n{json.dumps(tools)}\n[END OF AVAILABLE TOOLS]\n\n"
        prompt += f"[BEGIN OF FORMAT INSTRUCTION]\n{format_instruction}\n[END OF FORMAT INSTRUCTION]\n\n"
        prompt += f"[BEGIN OF QUERY]\n{query}\n[END OF QUERY]\n\n"
        
        if len(conversation_history) > 0: prompt += self.build_conversation_history_prompt(conversation_history)
        return prompt
        
    def build_conversation_history_prompt(self, conversation_history: str):
        parsed_history = []
        for step_data in conversation_history:
            parsed_history.append({
                "step_id": step_data["step_id"],
                "thought": step_data["thought"],
                "tool_calls": step_data["tool_calls"],
                "next_observation": step_data["next_observation"],
                "user_input": step_data['user_input']
            })
            
        history_string = json.dumps(parsed_history)
        return f"\n[BEGIN OF HISTORY STEPS]\n{history_string}\n[END OF HISTORY STEPS]\n"
        
    def parse_agent_action(self, agent_action: str):
        """
        Given an agent's action, parse it to add to conversation history
        """
        try: parsed_agent_action_json = json.loads(agent_action)
        except: return "", []
        
        if "thought" not in parsed_agent_action_json.keys(): thought = ""
        else: thought = parsed_agent_action_json["thought"]
        
        if "tool_calls" not in parsed_agent_action_json.keys(): tool_calls = []
        else: tool_calls = parsed_agent_action_json["tool_calls"]
        
        return thought, tool_calls

    def build_conversation_history(self, messages):
        history = []
        for msg in messages:
            if msg['role'] == 'tool':
                history[-1]['next_observation'] = msg['content']
            else:
                history.append({
                    'step_id': len(history) + 1,
                    'thought': msg.get('content', ''),
                    'tool_calls': [msg['tool_calls']] if 'tool_calls' in msg else [],
                    'next_observation': '',
                    'user_input': msg['content'] if msg['role'] == 'user' else ''
                })
        return history