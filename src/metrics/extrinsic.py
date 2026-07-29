from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any


@dataclass
class MaskedLanguageModelLoss:
    model_name_or_path: str
    texts_by_language: dict[str, list[str]]
    device: str | None = None
    batch_size: int = 32
    max_seq_length: int = 128
    mlm_probability: float = 0.15
    mask_seed: int = 0
    fp16: bool = True

    def compute(self) -> dict[str, Any]:
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Install `torch` and `transformers` to compute MLM loss."
            ) from exc

        if not self.texts_by_language:
            raise ValueError("MaskedLanguageModelLoss requires at least one language.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.max_seq_length <= 2:
            raise ValueError("max_seq_length must be greater than 2.")
        if not 0.0 < self.mlm_probability < 1.0:
            raise ValueError("mlm_probability must be between 0 and 1.")

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        model = AutoModelForMaskedLM.from_pretrained(self.model_name_or_path)
        model.to(device)
        model.eval()

        language_results = []
        total_loss_sum = 0.0
        total_loss_tokens = 0
        for language, texts in self.texts_by_language.items():
            result = self._language_loss(torch, tokenizer, model, language, texts, device)
            language_results.append(result)
            total_loss_sum += result["loss_sum"]
            total_loss_tokens += result["num_loss_tokens"]

        losses = [row["loss"] for row in language_results if row["loss"] is not None]
        mean_loss = float(sum(losses) / len(losses)) if losses else None
        token_weighted_loss = (
            float(total_loss_sum / total_loss_tokens)
            if total_loss_tokens > 0
            else None
        )
        return {
            "score": mean_loss,
            "mean_loss": mean_loss,
            "mean_perplexity": self._perplexity(mean_loss),
            "token_weighted_loss": token_weighted_loss,
            "token_weighted_perplexity": self._perplexity(token_weighted_loss),
            "num_languages": len(language_results),
            "languages": [row["language"] for row in language_results],
            "num_loss_tokens": total_loss_tokens,
            "batch_size": self.batch_size,
            "max_seq_length": self.max_seq_length,
            "mlm_probability": self.mlm_probability,
            "mask_seed": self.mask_seed,
            "language_results": language_results,
        }

    def _language_loss(
        self,
        torch: Any,
        tokenizer: Any,
        model: Any,
        language: str,
        texts: list[str],
        device: str,
    ) -> dict[str, Any]:
        loss_sum = 0.0
        num_loss_tokens = 0
        num_sequences = 0
        generator = self._torch_generator(torch, device, language)

        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start:start + self.batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            special_tokens_mask = encoded["special_tokens_mask"].to(device).bool()
            special_tokens_mask |= attention_mask == 0

            masked_inputs, labels, masked = self._mask_inputs(
                torch,
                tokenizer,
                input_ids,
                special_tokens_mask,
                generator,
                device,
            )
            masked_count = int(masked.sum().item())
            if masked_count == 0:
                continue

            autocast_device = "cuda" if device.startswith("cuda") else "cpu"
            with torch.no_grad():
                with torch.autocast(device_type=autocast_device, enabled=self.fp16 and device.startswith("cuda")):
                    output = model(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels)

            loss_sum += float(output.loss.item()) * masked_count
            num_loss_tokens += masked_count
            num_sequences += input_ids.shape[0]

        loss = float(loss_sum / num_loss_tokens) if num_loss_tokens > 0 else None
        return {
            "language": language,
            "loss": loss,
            "perplexity": self._perplexity(loss),
            "loss_sum": loss_sum,
            "num_loss_tokens": num_loss_tokens,
            "num_sequences": num_sequences,
            "num_texts": len(texts),
        }

    def _mask_inputs(
        self,
        torch: Any,
        tokenizer: Any,
        input_ids: Any,
        special_tokens_mask: Any,
        generator: Any,
        device: str,
    ) -> tuple[Any, Any, Any]:
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability, device=device)
        probability_matrix.masked_fill_(special_tokens_mask, 0.0)
        masked = torch.bernoulli(probability_matrix, generator=generator).bool()

        for row_index in range(masked.shape[0]):
            if masked[row_index].any():
                continue
            candidates = (~special_tokens_mask[row_index]).nonzero(as_tuple=False).flatten()
            if len(candidates) == 0:
                continue
            choice = torch.randint(len(candidates), (1,), generator=generator, device=device)
            masked[row_index, candidates[choice]] = True

        labels[~masked] = -100
        masked_inputs = input_ids.clone()

        replace_probs = torch.rand(labels.shape, generator=generator, device=device)
        mask_token_mask = masked & (replace_probs < 0.8)
        masked_inputs[mask_token_mask] = tokenizer.mask_token_id

        random_token_mask = masked & (replace_probs >= 0.8) & (replace_probs < 0.9)
        random_words = torch.randint(
            len(tokenizer),
            labels.shape,
            dtype=torch.long,
            generator=generator,
            device=device,
        )
        masked_inputs[random_token_mask] = random_words[random_token_mask]
        return masked_inputs, labels, masked

    def _torch_generator(self, torch: Any, device: str, language: str) -> Any:
        generator_device = device if str(device).startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device)
        digest = hashlib.sha256(f"{self.mask_seed}:{language}".encode("utf-8")).hexdigest()
        generator.manual_seed(int(digest[:16], 16) % (2**63 - 1))
        return generator

    def _perplexity(self, loss: float | None) -> float | None:
        if loss is None:
            return None
        if loss > 100:
            return float("inf")
        return float(math.exp(loss))
