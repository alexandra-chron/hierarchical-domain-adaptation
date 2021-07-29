"""
Evaluate pretrained GPT-2 models on standard datasets.
Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm_no_trainer.py.
"""  # noqa: E501

import math
from typing import Optional

import click
from click_help_colors import HelpColorsCommand
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    GPT2Tokenizer,
    default_data_collator,
    DataCollatorForLanguageModeling,
)
from tqdm import tqdm
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from tools.mmap_dataset import get_mmap_dataset
from tools.openwebtext_dataset import get_openwebtext_dataset
from tools.wikitext_dataset import get_wikitext_dataset

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2Tokenizer),
}


@click.command(
    cls=HelpColorsCommand,
    help_options_color="green",
    help_headers_color="yellow",
    context_settings={"max_content_width": 115},
)
@click.option("--model-name", default="gpt2")
@click.option(
    "--dataset",
    default="wikitext2",
    type=click.Choice(["wikitext2", "openwebtext", "mmap"]),
    show_choices=True,
    show_default=True,
    help="The dataset to evaluate on.",
)
@click.option(
    "--dataset-path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Path to the memory-mapped dataset (only valid when --dataset=mmap)",
)
@click.option(
    "--block-size",
    default=1024,
    show_default=True,
    help="""Input texts are blocked together into blocks of this size.
    This should probably match the max input size of the model.""",
)
@click.option(
    "--batch-size",
    default=32,
    show_default=True,
    help="The batch size to use for evaluation.",
)
@click.option(
    "--model-type",
    default='gpt2',
    show_default=True,
    help="""What model are we using""",
)
@click.option(
    "--use-adapters",
    default=True,
    show_default=True,
    help="""Should we use adapters""",
)
@click.option(
    "--adapter-size",
    default=256,
    show_default=True,
    help="""Size of bottleneck dimension""",
)
def main(
    model_name: str,
    dataset: str,
    dataset_path: Optional[str],
    block_size: int,
    batch_size: int,
    model_type: str,
    use_adapters: bool,
    adapter_size: int,
):
    """
    Evaluate a GPT-2 model on a dataset.
    """
    # Validate params.
    if dataset != "mmap" and dataset_path is not None:
        raise click.UsageError("'--dataset-path' only valid when '--dataset=mmap'")
    if dataset == "mmap" and dataset_path is None:
        raise click.UsageError("'--dataset-path' is required for this dataset type")

    click.secho("[1/3] Loading tokenizer and model...", fg="green")
    config_class, tokenizer_class = MODEL_CLASSES[model_type]

    config = config_class.from_pretrained(model_type)
    config.use_adapters = use_adapters
    config.adapter_size = adapter_size

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
    # model = model.cuda()
    print(model)
    click.secho("\n[2/3] Preprocessing data...", fg="green")

    dataset_object: Dataset
    collator = default_data_collator
    if dataset == "wikitext2":
        dataset_object = get_wikitext_dataset(
            tokenizer,
            split="test",
            block_size=block_size,
            num_workers=1,
        )
    elif dataset == "openwebtext":
        dataset_object = get_openwebtext_dataset(
            tokenizer,
            block_size=block_size,
            num_workers=8,
        )
    elif dataset == "mmap":
        assert dataset_path is not None
        dataset_object = get_mmap_dataset(tokenizer, dataset_path)
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    else:
        raise ValueError(f"Unexpected dataset '{dataset}'")

    click.secho("\n[3/3] Evaluating model on data...", fg="green")

    dataloader: DataLoader = DataLoader(
        dataset_object, collate_fn=collator, batch_size=32  # type: ignore[arg-type]
    )

    model.eval()

    losses = []
    total_batches = len(dataloader)
    with tqdm(dataloader, desc="Evaluating", total=total_batches) as batch_iterator:
        for i, batch in enumerate(batch_iterator):
            batch = {k: v for k, v in batch.items()}
            with torch.inference_mode():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(loss.unsqueeze(0))

            if i % 50 == 0 or i == total_batches - 1:
                mean_loss = torch.mean(torch.cat(losses))
                ppl = math.exp(mean_loss)
                batch_iterator.set_postfix(loss=mean_loss.item(), ppl=ppl)

    mean_loss = torch.mean(torch.cat(losses))
    ppl = math.exp(mean_loss)

    click.secho(f"\nDone! Final perplexity: {ppl:.4f}", fg="green", bold=True)


if __name__ == "__main__":
    main()