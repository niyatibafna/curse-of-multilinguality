from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.common import str_to_bool, training_dir


def main(
    train_file: str | None = None,
    tokenizer_path: str | None = None,
    output_dir: str | None = None,
    num_train_epochs: float = 3.0,
    per_device_train_batch_size: int = 64,
    gradient_accumulation_steps: int = 1,
    max_seq_length: int = 128,
    preprocessing_num_workers: int = 8,
    save_steps: int = 10_000,
    logging_steps: int = 100,
    seed: int = 13,
    fp16: bool = True,
) -> None:
    try:
        from datasets import load_dataset
        from transformers import (
            AutoTokenizer,
            BertConfig,
            BertForMaskedLM,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError("Install `datasets`, `torch`, and `transformers` to train MLM checkpoints.") from exc

    tokenizer_dir = Path(tokenizer_path) if tokenizer_path else training_dir() / "tokenizers" / "bert_wordpiece_50000"
    corpus = Path(train_file) if train_file else training_dir() / "corpora" / "fixed" / "n1.jsonl"
    output = Path(output_dir) if output_dir else training_dir() / "checkpoints" / corpus.parent.name / corpus.stem
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    dataset = load_dataset("json", data_files=str(corpus), split="train")

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(batch["text"], return_special_tokens_mask=True, truncation=False)

    tokenized = dataset.map(
        tokenize,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=dataset.column_names,
        desc="tokenizing",
    )
    grouped = tokenized.map(
        lambda batch: group_texts(batch, max_seq_length),
        batched=True,
        num_proc=preprocessing_num_workers,
        desc=f"grouping into {max_seq_length}-token chunks",
    )

    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=512,
        type_vocab_size=2,
    )
    model = BertForMaskedLM(config)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    args = TrainingArguments(
        output_dir=str(output),
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        seed=seed,
        fp16=fp16,
        report_to="none",
        save_total_limit=2,
    )
    trainer = Trainer(model=model, args=args, data_collator=collator, train_dataset=grouped)
    trainer.train()
    trainer.save_model(output)
    tokenizer.save_pretrained(output)
    print(output)


def group_texts(batch: dict[str, list[list[int]]], block_size: int) -> dict[str, list[list[int]]]:
    concatenated = {key: sum(batch[key], []) for key in batch}
    total_length = (len(concatenated["input_ids"]) // block_size) * block_size
    return {
        key: [values[index : index + block_size] for index in range(0, total_length, block_size)]
        for key, values in concatenated.items()
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file")
    parser.add_argument("--tokenizer_path")
    parser.add_argument("--output_dir")
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--preprocessing_num_workers", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=10_000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--fp16", type=str_to_bool, default=True)
    main(**vars(parser.parse_args()))
