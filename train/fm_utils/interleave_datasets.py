from datasets import interleave_datasets
from train.fm_utils.derived_dataset import ConstantLengthDataset, PromptAnswerDataset


def interleave_data(data_objects, sample_probs=None, seed=None, return_type="constant_length", **kwargs):
    """
    interleave same type of datasets from different sources

    :param data_objects: data source classes
    :param return_type:
        "basic": directly interleave datasets
        "constant_length": ConstantLengthDataset of the interleaved datasets
    """
    assert isinstance(data_objects, list)
    if sample_probs is not None:
        assert len(data_objects) == len(sample_probs)

    train_datasets = []
    eval_datasets = []
    chars_per_token = 0

    for d_obj in data_objects:
        print(f"create datasets from {d_obj.name}")
        train, eval = d_obj.create_datasets(return_type="basic", seed=seed)
        train_datasets.append(train)
        if eval is not None:
            eval_datasets.append(eval)
        chars_per_token = max(chars_per_token, d_obj.chars_per_token)

    if return_type == "basic":
        train_datasets = interleave_datasets(train_datasets, probabilities=sample_probs, seed=seed, stopping_strategy="all_exhausted")
        eval_datasets = interleave_datasets(eval_datasets, probabilities=sample_probs, seed=seed, stopping_strategy="all_exhausted")

    elif return_type == "constant_length":

        tokenizer = data_objects[0].tokenizer
        prepare_sample_text = data_objects[0].prepare_sample_text

        train_datasets = ConstantLengthDataset(
            tokenizer,
            train_datasets,
            sample_probs,
            seed,
            formatting_func=prepare_sample_text,
            infinite=True,
            seq_length=kwargs["seq_length"],
            chars_per_token=chars_per_token,
        )
        eval_datasets = ConstantLengthDataset(
            tokenizer,
            eval_datasets,
            sample_probs,
            seed,
            formatting_func=prepare_sample_text,
            infinite=False,
            seq_length=kwargs["seq_length"],
            chars_per_token=chars_per_token,
        )

    elif return_type == "prompt_answer":
        tokenizer = data_objects[0].tokenizer
        prepare_sample_text = data_objects[0].prepare_sample_text

        train_datasets = PromptAnswerDataset(
            tokenizer,
            train_datasets,
            sample_probs,
            seed,
            formatting_func=prepare_sample_text,
            infinite=True,
            seq_length=kwargs["seq_length"],
        )
        eval_datasets = PromptAnswerDataset(
            tokenizer,
            eval_datasets,
            sample_probs,
            seed,
            formatting_func=prepare_sample_text,
            infinite=False,
            seq_length=kwargs["seq_length"],
        )

    print(f"Data ready! The random shuffle and interleave seeding is {seed} for this device")
    return train_datasets, eval_datasets


