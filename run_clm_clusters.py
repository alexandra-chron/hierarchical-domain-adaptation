#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=causal-lm
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import time
from collections import defaultdict

import numpy as np
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch

import datasets
import json
from datasets import load_dataset
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import transformers
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed, GPT2Tokenizer,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_pt_utils import find_batch_size
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

from clustering.gmm_clusters_and_hierarchical_clustering import fit_gmm_and_hierarchical
from models.modeling_gpt2 import GPT2LMHeadModel
from models.configuration_gpt2 import GPT2Config
from trainer import Trainer

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    average_single_adapters: Optional[bool] = field(
        default=False, metadata={"help": "Average single adapters."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default='./corpora/film.train.json', metadata={"help":
                                                                "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    truncate_train_samples_for_this_domain: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples of this "
            "domain if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    use_adapters: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Are we using adapters"
        },
    )
    num_domains: Optional[int] = field(
        default=None,
        metadata={
            "help": "How many domains do we want to fine-tune the model on. Remember to also define a domain_dict.json "
                    "with the domain hierarchy and the domain_names.json"
        },
    )
    adapter_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Size of each adapter layer"
        },
    )
    vocab_overlap: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Compute the vocabulary overlap of given domains"
        },
    )
    use_tree_structure: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Use tree structure for adapters. If it is false, use simple multi-task/single-task learning "
                    "based on the num_domains"
        },
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    config = GPT2Config.from_pretrained(model_args.model_name_or_path)
    config.use_adapters = data_args.use_adapters
    config.num_domains = data_args.num_domains
    config.adapter_size = data_args.adapter_size
    config.use_tree_structure = data_args.use_tree_structure
    config.vocab_overlap = data_args.vocab_overlap

    if config.num_domains:
        if config.use_tree_structure:
            with open('domain_dict.json', 'r') as f:
                config.domain_dict = {int(k): v for (k, v) in json.load(f).items()}
        with open('domain_names.json', 'r') as f:
            config.domains = []
            for (k, v) in json.load(f).items():
                config.domains.append(v)
        if config.use_tree_structure:
            assert config.num_domains == len(config.domains), "Make sure you have provided a domain_names.json that" \
                                                          " lists ALL domains (number" \
                                                          " of domains specified with the num_domains flag)!"
        else:
            if len(config.domains) > config.num_domains:
                print("There is {} set of adapters for {} domains!!!! We will train multi-domain adapters!".format(config.num_domains,
                                                                                                                   len(config.domains)))
    path = "/".join(data_args.train_file.split("/")[:2]) + "/"

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )
    else:
        domains = config.domains
        data_files = {}
        for domain in domains:
            data_files[domain] = {}
            if training_args.do_train:
                for split in ["train", "valid"]:
                    if split == "valid": temp_split = "val"
                    else: temp_split = "train"
                    data_files[domain][split] = path + domain + "." + temp_split + ".json"
            else:
                split = "valid"
                temp_split = "train"
                data_files[domain][split] = path + domain + "." + temp_split + ".json"

        raw_datasets = {}
        for domain in domains:
            raw_datasets[domain] = {}
        for domain in domains:
            if training_args.do_train:
                for split in ["train", "valid"]:
                    raw_datasets[domain][split] = load_dataset("text", data_files={split: data_files[domain][split]},
                                                               split=split, cache_dir=model_args.cache_dir)
            else:
                raw_datasets[domain]["valid"] = load_dataset("text", data_files={"valid": data_files[domain]["valid"]},
                                                           split="valid", cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    tokenizer = GPT2Tokenizer.from_pretrained(model_args.model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(model_args.model_name_or_path, config=config,
                                            cache_dir=model_args.cache_dir
                                            )

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets[domains[0]]["valid"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = {}
        for domain in domains:
            tokenized_datasets[domain] = {}
            if training_args.do_train:
                for split in ["train", "valid"]:
                    tokenized_datasets[domain][split] = raw_datasets[domain][split].map(
                        tokenize_function,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer on dataset",
                    )
            else:
                split = "valid"
                tokenized_datasets[domain][split] = raw_datasets[domain][split].map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = {}
    for domain in domains:
        lm_datasets[domain] = {}
        split = "valid"
        with training_args.main_process_first(desc="grouping texts together"):
            lm_datasets[domain][split] = tokenized_datasets[domain][split].map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
                )

    if config.vocab_overlap:
        vocab = {}
        for i, domain in enumerate(domains):
            vocab[domain] = set()
            # for row in range(min(data_args.max_train_samples, len(lm_datasets[domain]["train"].data.columns[1]))):
            for row in range(len(lm_datasets[domain]['train'].data.columns[1])):
                for token_id in lm_datasets[domain]['train'].data.columns[1][row]:
                    if token_id not in vocab:
                        vocab[domain].add(int(str((token_id))))
        for i, current_domain in enumerate(domains):
            for next_domain in domains[i+1:]:
                if len(domains[i+1:]) > 0:
                    vocab_overlap = len(vocab[current_domain].intersection(vocab[next_domain]))
                    logger.warning("The vocabulary overlap between {} and {} is {}.".format(current_domain,
                                                                                   next_domain,
                                                                                   vocab_overlap))
    train_datasets = []
    eval_datasets = []
    if training_args.do_eval:
        for domain in domains:
            if "valid" not in tokenized_datasets[domain]:
                raise ValueError("--do_eval requires a validation dataset")
            eval_datasets.append(lm_datasets[domain]["valid"])

        if data_args.max_eval_samples is not None:
            for i in range(len(eval_datasets)):
                eval_datasets[i] = eval_datasets[i].select(range(data_args.max_eval_samples))
        for i, domain in enumerate(domains):
            logger.info("Eval dataset of domain {} length: {} rows.".format(domain, len(eval_datasets[i])))

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_datasets if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
    )
    model_to_domain_to_encodings_new = []
    num_clusters = 0
    max_size = 1000

    # Evaluation
    if training_args.do_eval:
        logger.info("*** encoding with gpt2 ***")

        with torch.no_grad():
            dataloaders = trainer.only_get_dataloader()
            observed_num_examples = 0
            model.eval()
            model_name = 'gpt2'

            for ind, dataloader in enumerate(dataloaders):
                start = time.time()
                model_to_states = {}
                model_to_states[model_name] = {}
                model_to_states[model_name]['states'] = []
                model_to_states[model_name]['sents'] = []
                current_size = 0
                for step, inputs in enumerate(dataloader):
                    current_size += 1
                    # Update the observed num examples
                    observed_batch_size = find_batch_size(inputs)
                    if observed_batch_size is not None:
                        observed_num_examples += observed_batch_size
                    inputs['attention_mask'] = inputs['attention_mask'].cuda()
                    inputs['input_ids'] = inputs['input_ids'].cuda()
                    inputs['labels'] = inputs['labels'].cuda()
                    outputs = model(**inputs, output_hidden_states=True)
                    last_hidden_states = outputs.hidden_states[-1]
                    # shape (batch_size, sequence_length, hidden_dim)
                    # avg pool last hidden layer
                    squeezed = last_hidden_states.squeeze(dim=0)
                    masked = squeezed[:inputs['input_ids'].shape[1], :]
                    avg_pooled = masked.mean(dim=0)
                    model_to_states[model_name]['states'].append(avg_pooled.cpu())
                    if current_size == max_size:
                        break
                end = time.time()
                print('encoded with {} in {} seconds'.format(model_name, end - start))
                np_tensors = [np.array(tensor) for tensor in model_to_states[model_name]['states']]
                model_to_states[model_name]['states'] = np.stack(np_tensors)
                model_to_domain_to_encodings_new.extend(model_to_states[model_name]['states'])
                num_clusters += 1


    # cluster the new split dev data
    first_principal = 1
    last_principal = 50
    num_experiments = 1
    use_pca = True

    model_to_accuracies = defaultdict(list)
    for i in range(num_experiments):
        if i == num_experiments - 1:
            plot = True
            confusion = True
        else:
            plot = False
            confusion = False

        accuracy = fit_gmm_and_hierarchical(model_to_domain_to_encodings_new, domains,
                                            first_principal_component_shown=first_principal,
                                            last_principal_component_shown=last_principal,
                                            clusters=num_clusters,
                                            pca=use_pca, confusion=confusion,
                                            examples_per_class=max_size)
        model_to_accuracies[model_name].append(accuracy)

    for model_name in model_to_accuracies:
        print('{0}\t{1:.2f} (±{2:.2f})'.format(model_name,
                                               np.mean(np.array(model_to_accuracies[model_name])),
                                               np.std(np.array(model_to_accuracies[model_name]))))

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
