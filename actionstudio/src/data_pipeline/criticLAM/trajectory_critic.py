import os
import time
import asyncio
import logging
import json
import argparse
import logging
from typing import Any
from tqdm import tqdm
from openai import AsyncOpenAI
from examples import example_trajectory, example_score, example_tools


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


try:       
    client = AsyncOpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
except Exception as e:
    logging.error(f"Error initializing OpenAI client: {e}")
    logging.error("Please check your OpenAI API key and environment variables. You can set it through 'export OPENAI_API_KEY=your_api_key'")
    exit(1)


async def dispatch_api_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float = 1.0,
) -> list[str]:
    """
    Dispatches requests to APIs asynchronously.
    
    Args:
        messages_list: List of messages to be sent to the ChatCompletion API.
        model: vLLM hosted model name and proprietary API name to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.

    Returns:
        List of responses from API.
    """
    if "o3-mini" in model:
        async_responses = [
            client.chat.completions.create(
                model=model,
                messages=x,
                # temperature=temperature,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
                store=False,
                # top_p=top_p,
            )
            for x in messages_list
        ]
    else:
        async_responses = [
            client.chat.completions.create(
                model=model,
                messages=x,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                store=False,
                # top_p=top_p,
            )
            for x in messages_list
        ]
    return await asyncio.gather(*async_responses)


if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="LLM-based evaluation.")
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-2024-11-20",  
        help="the API model name to call the LLM. The API name should come from vLLM host or proprietary API name."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="temperature to use for the LLM",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="the batch size to call the LLM."
    )
    parser.add_argument(
        "--target-data-dir",
        type=str,
        default="",
        help="the target dir path of all datasets to be evaluated by the LLM."
    )
    parser.add_argument(
        "--target-env-name",
        type=str,
        default=None,
        help="the name of the target dataset to be evaluated by the LLM."
    )
    parser.add_argument(
        "--num-target-processed-trajs-per-env",
        type=lambda x: int(x) if x.isdigit() else x,
        default="ALL",
        help="Number of target processed trajectories per environment, can be either an integer or a string. Default is process all trajectories for each env."
    )
    args = parser.parse_args()
    
    
    # Create corresponding output directory and file if it doesn't exist
    output_dir = os.path.join("outputs", args.llm_model)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except FileExistsError:
        # The directory already exists, no action needed
        pass
 
    output_json_file_name = f"{args.target_env_name}---{args.llm_model}.json"
    if os.path.isfile(os.path.join(output_dir, output_json_file_name)):
        print(f"❤️Attention❤️: {output_json_file_name} has already been filtered!\n")
        exit(1)
    
    actionstudio_data = json.load(open(os.path.join(args.target_data_dir, args.target_env_name, "train", "train.json"), "r"))
    if args.num_target_processed_trajs_per_env != "ALL":
        actionstudio_data = actionstudio_data[:args.num_target_processed_trajs_per_env]
        
    print(f"{args.target_env_name} data pairs: {len(actionstudio_data)}")
    
    # Create the system prompt with example content
    system_prompt = """
    ===
    To train a strong multi-turn agent model based on conversation trajectories, it's essential to assess the quality of each conversation—even if the trajectory consists of only a single turn. Each conversation will be accompanied by a list of provided tools.

    **Instructions:**

    As a professional conversational data evaluator, please:

    1. **Rate the quality** of the conversation on a scale from 1 to 5, where:
    - **5**: Best quality
    - **3**: Borderline quality
    - **1**: Bad quality

    2. **Justify your rating** with a clear and concise explanation, taking into account the use of the provided tools.
    
    3. **Present your response in the following JSON format:**

    ```json
    {
        "score": "<YOUR SCORE>",
        "explanation": "<YOUR EXPLANATION>"
    }
    ```
    """
    
    # Assemble example content for clarity and consistency.
    example_content = f"""
    [BEGIN OF PROVIDED TOOLS FOR THE ONE-SHOT EXAMPLE CONVERSATION]
    {example_tools.strip()}
    [END OF PROVIDED TOOLS FOR THE ONE-SHOT EXAMPLE CONVERSATION]

    [BEGIN OF ONE-SHOT EXAMPLE CONVERSATION]
    {example_trajectory.strip()}
    [END OF ONE-SHOT EXAMPLE CONVERSATION]

    === Here is the evaluator output for the above example:
    {example_score.strip()}
    """

    system_prompt = system_prompt.strip() + "\n"  + example_content
    print("*" * 50)
    print(system_prompt)
    print("*" * 50)
    
    messages_list = []
    for i in range(len(actionstudio_data)):
        eval_prompt = f"""[BEGIN OF PROVIDED TOOLS FOR THE CONVERSATION]\n{json.dumps(actionstudio_data[i]["tools"])}\n[END OF PROVIDED TOOLS FOR THE CONVERSATION]\n\n[BEGIN OF CONVERSATION]\n{json.dumps(actionstudio_data[i]["conversation"])}\n[END OF CONVERSATION]""".strip()
        
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": eval_prompt},
        ]
        messages_list.append(message)
    
    predictions = []
    idx = 0
    wait_base = 10
    retry = 0
    batch_size = args.batch_size
    skiped_ids = []
    pbar = tqdm(total=len(messages_list))
    while idx < len(messages_list):
        try:
            batch_predictions = asyncio.run(
                dispatch_api_requests(
                    messages_list[idx: idx+batch_size],
                    model=args.llm_model,
                    temperature=args.temperature,
                    max_tokens=args.max_new_tokens,
                    # top_p=1.0,
                )
            )
            predictions += batch_predictions
            idx += batch_size
            wait_base = 10
            pbar.update(batch_size)
        except Exception as e:
            retry += 1
            logging.error(f"Error: {e}") # This model's maximum context length is 128000 tokens.
            if "maximum context length is 128000 tokens" in str(e):
                logging.error("The model's maximum context length is 128000 tokens. Please reduce the length of the messages")
                if batch_size == 1: # skip this example
                    skiped_ids.append(actionstudio_data[idx]["unique_trajectory_id"])
                    idx += 1
                    continue
                else:
                    exit(1)
            logging.error("Batch error: {} to {}".format(idx, idx+batch_size))
            logging.info("Retry number: {} with Wait time: {}".format(retry, wait_base))
            time.sleep(wait_base)
            wait_base *= 2
    pbar.close()
    logger.info("All batches finished.\n\n")
    
    if len(skiped_ids) > 0:
        print(f"Skipped IDs count: {len(skiped_ids)}")
        print(f"Skipped IDs: {skiped_ids}")
        actionstudio_data = [example for example in actionstudio_data if example["unique_trajectory_id"] not in skiped_ids]

    # Further process the predictions
    final_saved_outputs = []
    for idx, prediction in tqdm(enumerate(predictions), total=len(predictions)):
        judge_score = prediction.choices[0].message.content
        triplet = f"""[BEGIN OF JSON DICT FILE]\n{json.dumps(messages_list[idx], indent=4)}\n[END OF JSON DICT FILE]\n"""
        meta_output = {
                    "unique_trajectory_id": actionstudio_data[idx]["unique_trajectory_id"],
                    "triplet":triplet,
                    "review":judge_score
                }
        final_saved_outputs.append(meta_output)
    
    
    # Save the outputs to a JSON file
    with open(os.path.join(output_dir, output_json_file_name), "w") as output_scores_file:
            json.dump(final_saved_outputs, output_scores_file, indent=4)
            
    logger.info("All outputs saved.")