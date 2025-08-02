from datasets import interleave_datasets
from actionstudio.src.foundation_modeling.data_handlers.derived_dataset import PromptAnswerDataset


def interleave_data(accelerator, data_objects, sample_probs=None, seed=None, return_type="prompt_answer", script_args=None, fc_mode=True, mask_prompt_loss=True, **kwargs):
    """
    interleave same type of datasets from different sources

    :param data_objects: data source classes
    :param return_type:
        "basic": directly interleave datasets
        "prompt_answer": PromptAnswerDataset of the interleaved datasets
    """
    assert isinstance(data_objects, list)
    
    if sample_probs is not None:
        assert len(data_objects) == len(sample_probs)

    train_datasets = []
    eval_datasets = []
    chars_per_token = 0

    for d_obj in data_objects:
        if accelerator.is_main_process:
            print(f"create datasets from ❤️{d_obj.dataset_name}❤️")
        train_dataset, eval_dataset = d_obj.create_datasets(return_type="basic", seed=seed)
        train_datasets.append(train_dataset)
        if eval_dataset is not None:
            eval_datasets.append(eval_dataset)
        chars_per_token = max(chars_per_token, d_obj.chars_per_token)

    if return_type == "basic":
        train_datasets = interleave_datasets(
            train_datasets,
            probabilities=sample_probs,
            seed=seed,
            stopping_strategy="all_exhausted"
        )
        
        if len(eval_datasets) > 0:
            eval_datasets = interleave_datasets(
                eval_datasets,
                probabilities=sample_probs,
                seed=seed,
                stopping_strategy="all_exhausted"
            )
        else: eval_datasets = None

    elif return_type == "prompt_answer":
        # Note: FC mode is only supported for prompt answer for now
        tokenizer = data_objects[0].tokenizer
        prepare_sample_text = data_objects[0].prepare_sample_text

        train_datasets = PromptAnswerDataset(
            accelerator,
            tokenizer,
            train_datasets,
            sample_probs,
            seed,
            script_args=script_args,
            formatting_func=prepare_sample_text,
            mask_prompt_loss=mask_prompt_loss,
            infinite=True,
            seq_length=kwargs["seq_length"],
            fc_mode=fc_mode
        )

        try:
            eval_datasets = PromptAnswerDataset(
                accelerator,
                tokenizer,
                eval_datasets,
                sample_probs,
                seed,
                script_args=script_args,
                formatting_func=prepare_sample_text,
                mask_prompt_loss=mask_prompt_loss,
                infinite=False,
                seq_length=kwargs["seq_length"],
                fc_mode=fc_mode
            )
        except: eval_datasets = None
    
    else: raise Exception("Only support basic or prompt_answer")

    if accelerator.is_main_process:
        print(f"Data ready! The random shuffle and interleave seeding is {seed} for this device")
    return train_datasets, eval_datasets


