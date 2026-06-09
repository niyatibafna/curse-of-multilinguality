from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.common import training_dir


def main(
    input_files: str | list[str] | None = None,
    output_dir: str | None = None,
    vocab_size: int = 50_000,
    min_frequency: int = 2,
    lowercase: bool = False,
) -> None:
    try:
        from tokenizers import BertWordPieceTokenizer
        from transformers import BertTokenizerFast
    except ImportError as exc:
        raise ImportError("Install `tokenizers` and `transformers` to train the tokenizer.") from exc

    files = normalize_files(input_files)
    output = Path(output_dir) if output_dir else training_dir() / "tokenizers" / f"bert_wordpiece_{vocab_size}"
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = BertWordPieceTokenizer(lowercase=lowercase)
    tokenizer.train(
        files=files,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )
    tokenizer.save_model(str(output))

    fast = BertTokenizerFast(
        vocab_file=str(output / "vocab.txt"),
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        do_lower_case=lowercase,
    )
    fast.save_pretrained(output)
    print(output)


def normalize_files(input_files: str | list[str] | None) -> list[str]:
    if input_files is None:
        default = training_dir() / "corpora" / "tokenizer" / "n100.txt"
        return [str(default)]
    if isinstance(input_files, str):
        return [item.strip() for item in input_files.split(",") if item.strip()]
    return [str(item) for item in input_files]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files")
    parser.add_argument("--output_dir")
    parser.add_argument("--vocab_size", type=int, default=50_000)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--lowercase", action="store_true")
    main(**vars(parser.parse_args()))
