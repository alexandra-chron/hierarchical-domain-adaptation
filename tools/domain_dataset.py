import json
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

from .common import get_group_texts_function


def get_domain_dataset(
    tokenizer: GPT2Tokenizer,
    *,
    split: str = "test",
    block_size: int = 1024,
    num_workers: int = 1,
    domain: str = "",
    path: str = "./corpora",
) -> Dataset:
    dataset = load_dataset("text", data_files={split: [path + domain]}, split=split)

    def tokenize_function(example):
        return tokenizer(example["text"])

    dataset = dataset.map(  ## type: ignore[union-attr,call-arg]
        tokenize_function,
        batched=True,
        num_proc=num_workers,
        remove_columns=["text"],
        desc="Tokenizing dataset",
    )

    group_texts = get_group_texts_function(block_size)

    dataset = dataset.map(
        group_texts,
        batched=True,
        num_proc=num_workers,
        desc=f"Grouping texts into chunks of {block_size}",
    )

    return dataset  ## type: ignore[return-value]