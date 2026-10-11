"""Autoregressive audio-token decoding for the XTTS GPT.

Reproduces `transformers` `generate()` sampling (same logits processors, order and RNG use) on top of public APIs
only, so it is independent of `generate()` internals and also yields the per-step latents for streaming.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from transformers import DynamicCache, LogitsProcessorList
from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from TTS.tts.layers.xtts import static_decoder

if TYPE_CHECKING:
    from TTS.tts.layers.xtts.gpt import GPT

logger = logging.getLogger(__name__)

PREFIX_TOKEN_ID = 1
"""Placeholder id for prefix (conditioning + text) positions; `generate()` sees these in its repetition penalty."""


@dataclass
class Sampling:
    """Decoding settings; defaults match `transformers.GenerationConfig`. As in `generate()`, `None` turns a
    setting off (top_k=None: no top-k filtering)."""

    do_sample: bool | None = False
    temperature: float | None = 1.0
    top_k: int | None = 50
    top_p: float | None = 1.0
    repetition_penalty: float | None = 1.0

    def __post_init__(self):
        off = {"do_sample": False, "temperature": 1.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0}
        for name, value in off.items():
            if getattr(self, name) is None:
                setattr(self, name, value)

    @classmethod
    def from_kwargs(cls, kwargs: dict) -> "Sampling":
        """Pop the supported settings from `kwargs`; missing ones keep the defaults."""
        return cls(**{k: kwargs.pop(k) for k in cls.__dataclass_fields__ if k in kwargs})

    def processors(self) -> LogitsProcessorList:
        processors = LogitsProcessorList()
        if self.repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=float(self.repetition_penalty)))
        if self.do_sample:
            if self.temperature != 1.0:
                processors.append(TemperatureLogitsWarper(float(self.temperature)))
            if self.top_k:
                processors.append(TopKLogitsWarper(top_k=int(self.top_k), min_tokens_to_keep=1))
            if self.top_p < 1.0:
                processors.append(TopPLogitsWarper(top_p=float(self.top_p), min_tokens_to_keep=1))
        return processors


def pick_tokens(scores: torch.Tensor, do_sample: bool) -> torch.Tensor:
    """Next token per row: multinomial sample of the softmax, or argmax."""
    if not do_sample:
        return scores.argmax(dim=-1)
    probs = scores.softmax(dim=-1)
    zero_rows = probs.sum(dim=-1) == 0
    if zero_rows.any():  # all logits filtered or NaN: multinomial would raise a device-side assert
        logger.warning("No valid token to sample in %d row(s); sampling uniformly", int(zero_rows.sum()))
        probs[zero_rows] = 1.0 / probs.shape[-1]
    return torch.multinomial(probs, num_samples=1).squeeze(1)


class EagerStepper:
    """Decode steps through the `transformers` GPT-2 modules with a dynamic KV cache."""

    def __init__(self, gpt: "GPT"):
        self.gpt = gpt
        self.cache = DynamicCache()
        self.mask: torch.Tensor | None = None
        self.mel_pos = 0

    def _forward(self, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.gpt.gpt(inputs_embeds=emb, past_key_values=self.cache, attention_mask=self.mask, use_cache=True)
        latents = self.gpt.final_norm(out.last_hidden_state[:, -1])
        return latents, self.gpt.mel_head(latents)

    def prefill(self, emb: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.mask = mask
        return self._forward(emb)

    def step(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.mel_pos += 1
        self.mask = torch.cat([self.mask, self.mask.new_ones(self.mask.shape[0], 1)], dim=1)
        pos_emb = self.gpt.mel_pos_embedding.get_fixed_embedding(self.mel_pos, tokens.device)
        return self._forward(self.gpt.mel_embedding(tokens[:, None]) + pos_emb)


@torch.inference_mode()
def decode(
    gpt: "GPT",
    prefix_emb: torch.Tensor,
    prefix_mask: torch.Tensor | None = None,
    *,
    sampling: Sampling,
    max_new_tokens: int | None = None,
    cuda_graph: bool = False,
    active: torch.Tensor | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Generate audio tokens after a conditioning + text prefix, one step at a time.

    Args:
        gpt: XTTS GPT, prepared with `init_gpt_for_inference()`.
        prefix_emb: `[B, P, D]` prefix embeddings, left-padded when rows differ in length.
        prefix_mask: `[B, P]`, 0 at padding. `None` means no padding.
        sampling: Decoding settings.
        max_new_tokens: Token limit per row, including the stop token. Defaults to `gpt.max_gen_mel_tokens`.
        cuda_graph: Decode with a static KV cache, replaying each step as a CUDA graph. Much faster on CUDA; the
            logits match the default path up to float rounding, so sampled tokens may diverge.
        active: `[B]` bool, False for filler rows that only keep the batch shape; they emit the stop token.

    Yields:
        `(tokens [B], latents [B, D])` per step. `latents` is the normed hidden state that predicted `tokens`, as fed
        to the HiFi-GAN decoder. Finished rows repeat the stop token. Stops once every row emitted the stop token.
    """
    batch, prefix_len, _ = prefix_emb.shape
    device = prefix_emb.device
    stop = gpt.stop_audio_token
    max_new_tokens = gpt.max_gen_mel_tokens if max_new_tokens is None else max_new_tokens
    processors = sampling.processors()

    start = torch.full((batch, 1), gpt.start_audio_token, dtype=torch.long, device=device)
    seq_ids = torch.cat([torch.full((batch, prefix_len), PREFIX_TOKEN_ID, device=device), start], dim=1)
    mask = torch.ones_like(seq_ids)
    if prefix_mask is not None:
        mask[:, :prefix_len] = prefix_mask
    token_emb = gpt.mel_embedding(start) + gpt.mel_pos_embedding.get_fixed_embedding(0, device)
    emb = torch.cat([prefix_emb.to(token_emb.dtype), token_emb], dim=1)

    if cuda_graph and static_decoder.supported(gpt):
        stepper = static_decoder.StaticDecoder.for_gpt(gpt, batch, emb.shape[1] + max_new_tokens)
    else:
        stepper = EagerStepper(gpt)
    latents, logits = stepper.prefill(emb, mask)
    unfinished = torch.ones(batch, dtype=torch.bool, device=device) if active is None else active.clone()
    for _ in range(max_new_tokens):
        scores = processors(seq_ids, logits.float())
        tokens = torch.where(unfinished, pick_tokens(scores, sampling.do_sample), stop)
        yield tokens, latents.clone()

        unfinished &= tokens != stop
        if not unfinished.any():
            return
        seq_ids = torch.cat([seq_ids, tokens[:, None]], dim=1)
        latents, logits = stepper.step(tokens)
