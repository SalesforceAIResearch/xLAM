import json
import re
import sys
import time
import random
import os
import argparse


GLOBAL_FORMAT_INSTRUCTION_GENERAL = """
You are an assistant that can use tools. When making one or more tool calls, format them in a single JSON list, where each object represents a tool call. Do not output separate objects or multiple arrays. Use the following structure:

```
[{"name": "tool_call_name", "arguments": {"arg1": "value1", "arg2": "value2"}}, ... (additional parallel tool calls as needed)]
```

Do not provide any interpretation or response until the tool call results are returned, which you can then process for the user or proceed to use other tools.

If no tool call is necessary (e.g., need for clarification, general advice/explanations, casual conversation, or task completed), respond with plain text.
""".strip()

GLOBAL_FORMAT_INSTRUCTION = """
You are an assistant that can use tools. When making one or more tool calls, format them in a single JSON list after [TOOL_CALLS], where each object represents a tool call. Do not output separate objects or multiple arrays. Use the following structure:

```
[TOOL_CALLS][{"name": "tool_call_name", "arguments": {"arg1": "value1", "arg2": "value2"}}, ... (additional parallel tool calls as needed)]
```

Do not provide any interpretation or response until the tool call results are returned, which you can then process for the user or proceed to use other tools.

If no tool call is necessary (e.g., need for clarification, general advice/explanations, casual conversation, or task completed), respond with plain text.
""".strip()

def read_json(data_path):
    if os.path.exists(data_path):
        with open(data_path) as infile:
            ta_data = json.load(infile)
        return ta_data
    return None

def get_data_dict(data_path, data_keys, suffix):
    train_data_dict = {}
    for key in data_keys:
        if key == 'xlam-fc-60k':
            subkeys = ['train_part1.json', 'train_part2.json', 'train_part3.json']
            xlam_data = []
            for subkey in subkeys:
                train_data_path = os.path.join(data_path, key + '/train/' + subkey)
                ta_data = read_json(train_data_path)
                xlam_data += ta_data
            if xlam_data:
                train_data_dict[key] = xlam_data
            continue
        train_data_path = os.path.join(data_path, key + suffix)
        ta_data = read_json(train_data_path)
        if ta_data:
            train_data_dict[key] = ta_data
    return train_data_dict

def get_unified_data_dict(data_keys, unified_data_path, is_expand=False):
    train_data_dict = get_data_dict(unified_data_path, data_keys, '/train/train.json')
    return train_data_dict

def convert_data_dict_v2_to_train_model_free(uni_data_dict_v2_cot, is_mistral=True):
    converted_train_data_dict = {}
    for data_key, train_data in uni_data_dict_v2_cot.items():
        print(data_key)
        train_dps = convert_v2_to_train_model_free(data_key, train_data, is_mistral)
        converted_train_data_dict[data_key] = train_dps
        print(data_key, len(train_data), len(train_dps))
    return converted_train_data_dict

def get_format_instruction(data_key='other', is_mistral=True):
    if 'gorilla' in data_key:
        format_ins = GLOBAL_FORMAT_INSTRUCTION_GORILLA
        if not is_mistral:
            format_ins = GLOBAL_FORMAT_INSTRUCTION_GENERAL_GORILLA
    else:
        format_ins = GLOBAL_FORMAT_INSTRUCTION
        if not is_mistral:
            format_ins = GLOBAL_FORMAT_INSTRUCTION_GENERAL
    return format_ins

def convert_v2_to_train_model_free(data_key, train_data, is_mistral=True):
    train_dps = []
    error_count = 0
    format_ins = get_format_instruction(data_key, is_mistral)
    for dp in train_data:
        for i, msg in enumerate(dp['conversation']):
            if msg['role'] == 'assistant':
                train_dp = convert_v2_dp_to_train(dp, format_ins, i+1, is_mistral)
                if train_dp:
                    train_dps.append(train_dp)
                else:
                    error_count += 1
    print(len(train_dps), 'error count', error_count)
    return train_dps

def convert_toolcalls_to_output(tool_calls):
    new_tool_calls = []
    for tool_call in tool_calls:
        new_tc = tool_call['function']
        new_tool_calls.append(new_tc)
    return new_tool_calls

def convert_output_v2_to_train(message, is_prefix=True):
    assert message['role'] == 'assistant'
    if 'tool_calls' in message:
        tool_calls = convert_toolcalls_to_output(message['tool_calls'])
        tool_calls_str = json.dumps(tool_calls)
        if is_prefix:
            return "[TOOL_CALLS]" + tool_calls_str
        return tool_calls_str
    if not message['content']:
        return None
    if len(message['content'].strip()) == 0:
        return None
    if message['content'].strip().lower() in ['none', 'null']:
        return None
    return message['content']

def get_messages_from_dp_v2(dp, format_ins):
    task_instruction = dp['task_instruction']
    tools = dp['tools']
    messages = [
      {"role": "system", "content": task_instruction + '\n' + format_ins},
    ]
    messages += dp['conversation']
    
    return messages, tools

def convert_v2_dp_to_train(dp, format_ins=GLOBAL_FORMAT_INSTRUCTION, idx=-1, is_mistral=True):
    all_messages, tools = get_messages_from_dp_v2(dp, format_ins)
    messages = all_messages[:idx]
    chosen_msg = all_messages[idx]
    chosen = convert_output_v2_to_train(chosen_msg, is_mistral)
    if not chosen:
        return None
    has_toolcalls = False
    has_thought = False
    if 'tool_calls' in chosen_msg:
        has_toolcalls = True
    if 'tool_calls' in chosen_msg and 'content' in chosen_msg and chosen_msg['content']:
        has_thought = True
    train_dp = {
        'id':dp['unique_trajectory_id'] + '---' + str(idx),
        'messages':messages,
        'tools':dp['tools'],
        'chosen':chosen,
        'output_message':chosen_msg,
    }
    return train_dp

def save_data_dict_to_disk(converted_train_data_dict, v2_path):
    for data_key, train_dps in converted_train_data_dict.items(): 
        os.makedirs(os.path.join(v2_path, data_key, 'train'), exist_ok=True)
    for data_key, train_dps in converted_train_data_dict.items(): 
        print(data_key, len(train_dps))
        with open(os.path.join(v2_path, data_key, 'train', 'train.json'), 'w') as outfile:
            json.dump(train_dps, outfile, indent=4)
    return


def main():
    parser = argparse.ArgumentParser(description="A sample script.")
    parser.add_argument("--unified_data_path", type=str, default="../../datasets/unified_data/", help="unified_data_path")
    parser.add_argument("--output_data_path", type=str, default="../../datasets/train_data/", help="output_data_path")
    args = parser.parse_args()

    unified_data_path = args.unified_data_path
    train_data_path = args.output_data_path
    use_data_keys = ['actionstudio-98k'] # a list of dataset names 
    uni_data_dict = get_unified_data_dict(use_data_keys, unified_data_path, is_expand=False)
    converted_train_data_dict = convert_data_dict_v2_to_train_model_free(uni_data_dict)
    print(converted_train_data_dict.keys())
    save_data_dict_to_disk(converted_train_data_dict, train_data_path)

if __name__ == "__main__":
    main()
