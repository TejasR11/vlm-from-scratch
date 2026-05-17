"""Vision-Language Model — §5.

You implement: VisionLanguageModel.

Three injection strategies to support:
  - "cls":          Single visual token (the ViT's CLS embedding) prepended.
  - "all_patches":  All N+1 visual tokens (CLS + patches) prepended.
  - "interleaved":  A special <image> token in the prompt is replaced by the
                    sequence of patch embeddings at runtime.

Two attention masking strategies to support (Problem `masking`):
  - "causal":         Fully causal across the whole sequence.
  - "image_bidir":    Bidirectional within the image block, causal everywhere
                      else. Use vlm.masking.build_image_bidir_mask().
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from vlm.masking import build_image_bidir_mask

InjectionMode = Literal["cls", "all_patches", "interleaved"]
MaskMode = Literal["causal", "image_bidir"]


class VisionLanguageModel(nn.Module):
    """ViT image encoder + projector + pretrained causal LM decoder.

    Args:
        vit:       Your CLIP-pretrained ViT from §3.
        projector: vlm.projector.VisionLanguageProjector instance.
        decoder:   HuggingFace causal LM (e.g., SmolLM2-360M-Instruct) loaded
                   in bf16 with FlashAttention-2.
        tokenizer: Matching HF tokenizer.
        image_token_id: Token ID corresponding to the special <image> placeholder
                        in interleaved mode (None for cls / all_patches modes).

    Forward:
        images:         (B, 3, H, W) float tensor.
        input_ids:      (B, T) tokenized text.
        attention_mask: (B, T) text attention mask from the tokenizer.
        labels:         (B, T) for loss computation, or None for inference.
                        Visual-token positions must be set to -100 in labels
                        before being passed in (so they're masked out by HF's
                        loss).
        injection:      One of "cls", "all_patches", "interleaved".
        mask_mode:      One of "causal", "image_bidir".

    Returns:
        A dict with at least:
          - "loss":   scalar (only if labels was provided).
          - "logits": (B, T_total, vocab_size).
    """

    def __init__(
        self,
        vit: nn.Module,
        projector: nn.Module,
        decoder: nn.Module,
        tokenizer,
        image_token_id: int | None = None,
    ) -> None:
        super().__init__()
        self.vit = vit
        self.projector = projector
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id

    def _encode_visual_tokens(
        self, images: torch.Tensor, injection: InjectionMode
    ) -> torch.Tensor:
        if injection == "cls":
            image_features = self.vit(images)
        elif injection in ("all_patches", "interleaved"):
            image_features = self.vit(images, return_all_tokens=True)
        else:
            raise ValueError(f"Unknown injection mode: {injection}")
        return self.projector(image_features)

    def _decoder_embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder.get_input_embeddings()(input_ids)

    def _prepare_labels(
        self, labels: torch.Tensor | None, attention_mask: torch.Tensor
    ) -> torch.Tensor | None:
        if labels is None:
            return None
        labels = labels.clone()
        labels = labels.masked_fill(attention_mask == 0, -100)
        return labels

    def _prefix_inputs(
        self,
        visual_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
        B, n_visual, _ = visual_embeds.shape
        visual_mask = torch.ones(
            B,
            n_visual,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        stitched_attention_mask = torch.cat([visual_mask, attention_mask], dim=1)

        adjusted_labels = None
        if labels is not None:
            visual_labels = torch.full(
                (B, n_visual),
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )
            adjusted_labels = torch.cat([visual_labels, labels], dim=1)
        return inputs_embeds, stitched_attention_mask, adjusted_labels, n_visual

    def _interleaved_inputs(
        self,
        visual_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, list[int]]:
        if self.image_token_id is None:
            raise ValueError("image_token_id must be set for interleaved injection")

        B, n_visual, _ = visual_embeds.shape
        stitched_embeds = []
        stitched_masks = []
        stitched_labels = [] if labels is not None else None
        visual_starts: list[int] = []

        for b in range(B):
            image_positions = (input_ids[b] == self.image_token_id).nonzero(as_tuple=False)
            if image_positions.numel() != 1:
                raise ValueError(
                    "interleaved injection expects exactly one <image> token per example"
                )
            pos = int(image_positions.item())
            visual_starts.append(pos)

            stitched_embeds.append(
                torch.cat(
                    [
                        text_embeds[b, :pos],
                        visual_embeds[b],
                        text_embeds[b, pos + 1 :],
                    ],
                    dim=0,
                )
            )
            stitched_masks.append(
                torch.cat(
                    [
                        attention_mask[b, :pos],
                        torch.ones(
                            n_visual,
                            device=attention_mask.device,
                            dtype=attention_mask.dtype,
                        ),
                        attention_mask[b, pos + 1 :],
                    ],
                    dim=0,
                )
            )
            if labels is not None and stitched_labels is not None:
                stitched_labels.append(
                    torch.cat(
                        [
                            labels[b, :pos],
                            torch.full(
                                (n_visual,),
                                -100,
                                device=labels.device,
                                dtype=labels.dtype,
                            ),
                            labels[b, pos + 1 :],
                        ],
                        dim=0,
                    )
                )

        inputs_embeds = torch.stack(stitched_embeds, dim=0)
        stitched_attention_mask = torch.stack(stitched_masks, dim=0)
        adjusted_labels = (
            torch.stack(stitched_labels, dim=0)
            if stitched_labels is not None
            else None
        )
        return inputs_embeds, stitched_attention_mask, adjusted_labels, visual_starts

    def _interleaved_image_bidir_mask(
        self,
        visual_starts: list[int],
        n_visual: int,
        attention_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        B, seq_len = attention_mask.shape
        min_value = torch.finfo(dtype).min
        masks = []
        causal = torch.full(
            (seq_len, seq_len),
            min_value,
            device=attention_mask.device,
            dtype=dtype,
        )
        causal = torch.triu(causal, diagonal=1)

        for b in range(B):
            mask = causal.clone()
            start = visual_starts[b]
            end = start + n_visual
            mask[start:end, start:end] = 0
            mask = mask.masked_fill(attention_mask[b][None, :] == 0, min_value)
            masks.append(mask)
        return torch.stack(masks, dim=0).unsqueeze(1)

    def _decoder_attention_mask(
        self,
        injection: InjectionMode,
        mask_mode: MaskMode,
        n_visual: int,
        n_text: int,
        attention_mask: torch.Tensor,
        inputs_dtype: torch.dtype,
        visual_starts: list[int] | None = None,
    ) -> torch.Tensor:
        if mask_mode == "causal":
            return attention_mask
        if mask_mode != "image_bidir":
            raise ValueError(f"Unknown mask_mode: {mask_mode}")

        min_value = torch.finfo(inputs_dtype).min
        if injection == "interleaved":
            if visual_starts is None:
                raise ValueError("visual_starts are required for interleaved masking")
            return self._interleaved_image_bidir_mask(
                visual_starts, n_visual, attention_mask, inputs_dtype
            )

        mask = build_image_bidir_mask(
            n_visual=n_visual,
            n_text=n_text,
            device=attention_mask.device,
            dtype=inputs_dtype,
        )
        mask = mask.expand(attention_mask.shape[0], -1, -1, -1).clone()
        return mask.masked_fill(attention_mask[:, None, None, :] == 0, min_value)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        injection: InjectionMode = "cls",
        mask_mode: MaskMode = "causal",
    ) -> dict:
        visual_embeds = self._encode_visual_tokens(images, injection)
        text_embeds = self._decoder_embed_tokens(input_ids)
        visual_embeds = visual_embeds.to(dtype=text_embeds.dtype)
        text_labels = self._prepare_labels(labels, attention_mask)

        if injection in ("cls", "all_patches"):
            inputs_embeds, stitched_attention_mask, adjusted_labels, n_visual = (
                self._prefix_inputs(
                    visual_embeds, text_embeds, attention_mask, text_labels
                )
            )
            visual_starts = None
            n_text = input_ids.shape[1]
        elif injection == "interleaved":
            inputs_embeds, stitched_attention_mask, adjusted_labels, visual_starts = (
                self._interleaved_inputs(
                    visual_embeds,
                    text_embeds,
                    input_ids,
                    attention_mask,
                    text_labels,
                )
            )
            n_visual = visual_embeds.shape[1]
            n_text = stitched_attention_mask.shape[1] - n_visual
        else:
            raise ValueError(f"Unknown injection mode: {injection}")

        decoder_attention_mask = self._decoder_attention_mask(
            injection=injection,
            mask_mode=mask_mode,
            n_visual=n_visual,
            n_text=n_text,
            attention_mask=stitched_attention_mask,
            inputs_dtype=inputs_embeds.dtype,
            visual_starts=visual_starts,
        )

        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=decoder_attention_mask,
            labels=adjusted_labels,
        )
        result = {"logits": outputs.logits}
        if labels is not None:
            result["loss"] = outputs.loss
        return result

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompts: list[str],
        injection: InjectionMode = "cls",
        max_new_tokens: int = 32,
        **gen_kwargs,
    ) -> list[str]:
        """Generate text continuations conditioned on images + prompts.

        Useful for §5's qualitative evaluation problem (vlm_qualitative).
        """
        device = images.device
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        visual_embeds = self._encode_visual_tokens(images, injection)
        text_embeds = self._decoder_embed_tokens(input_ids)
        visual_embeds = visual_embeds.to(dtype=text_embeds.dtype)

        if injection in ("cls", "all_patches"):
            inputs_embeds, stitched_attention_mask, _, _ = self._prefix_inputs(
                visual_embeds, text_embeds, attention_mask, None
            )
        elif injection == "interleaved":
            inputs_embeds, stitched_attention_mask, _, _ = self._interleaved_inputs(
                visual_embeds, text_embeds, input_ids, attention_mask, None
            )
        else:
            raise ValueError(f"Unknown injection mode: {injection}")

        generated = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=stitched_attention_mask,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
