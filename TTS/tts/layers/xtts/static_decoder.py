"""Single-token GPT-2 decoding over a preallocated KV cache, replayed as a CUDA graph.

The XTTS GPT emits one audio token per forward pass, so eager decoding is bound by kernel launch overhead (hundreds of
launches per token, most of the GPU idle). Capturing one step and replaying it removes that overhead. The step reuses
the GPT-2 block weights and covers the configuration XTTS builds: self-attention only, plain `1/sqrt(head_dim)` scaling.
"""

import math
import weakref
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from TTS.tts.layers.tortoise.autoregressive import LearnedPositionEmbeddings

if TYPE_CHECKING:
    from TTS.tts.layers.xtts.gpt import GPT

CACHE_ROUND = 256
"""Cache length granularity, so similar requests reuse one captured graph."""
MAX_CACHED = 2
"""Decoders (one per batch size) kept per GPT, e.g. single-sentence streaming plus one batch size."""


def weight_pointers(gpt: "GPT") -> tuple:
    """Storage of every weight the step reads: a captured graph is only valid while these stay put."""
    modules = (gpt.gpt, gpt.mel_embedding, gpt.mel_pos_embedding, gpt.final_norm, gpt.mel_head)
    return tuple((p.data_ptr(), p.dtype, p.device) for module in modules for p in module.parameters())


def supported(gpt: "GPT") -> bool:
    """Whether the static step reproduces this GPT's attention."""
    attn = gpt.gpt.h[0].attn
    return (
        isinstance(gpt.mel_pos_embedding, LearnedPositionEmbeddings)
        and not gpt.mel_pos_embedding.relative
        and attn.scale_attn_weights
        and not attn.scale_attn_by_inverse_layer_idx
        and not attn.reorder_and_upcast_attn
        and not attn.is_cross_attention
    )


class StaticDecoder:
    """KV-cached GPT decoder with static shapes: `prefill()` the prompt eagerly, then `step()` one token at a time.

    On CUDA the step is captured as a graph on its first call and replayed afterwards. Outputs of `step()` live in
    static buffers that the next step overwrites. Get one with `acquire()` and hand it back with `release()`: a
    decoder serves one decode at a time.
    """

    def __init__(self, gpt: "GPT", batch: int, max_len: int, *, use_graph: bool = True):
        param = next(gpt.gpt.parameters())
        config = gpt.gpt.config
        self._gpt = weakref.ref(gpt)  # the GPT caches its decoders: no reference cycle
        self.weights = weight_pointers(gpt)
        self.batch = batch
        self.max_len = max_len
        self.dtype = param.dtype
        self.device = param.device
        self.heads = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.use_graph = use_graph and self.device.type == "cuda"

        cache_shape = (len(gpt.gpt.h), batch, self.heads, max_len, self.head_dim)
        self.keys = torch.zeros(cache_shape, dtype=self.dtype, device=self.device)
        self.values = torch.zeros(cache_shape, dtype=self.dtype, device=self.device)
        self.valid = torch.zeros(batch, 1, 1, max_len, dtype=torch.bool, device=self.device)
        self.write_pos = torch.zeros(1, dtype=torch.long, device=self.device)
        self.mel_pos = torch.zeros(1, dtype=torch.long, device=self.device)
        self.tokens = torch.zeros(batch, 1, dtype=torch.long, device=self.device)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.latents: torch.Tensor | None = None
        self.logits: torch.Tensor | None = None

    @property
    def gpt(self) -> "GPT":
        gpt = self._gpt()
        if gpt is None:
            msg = "The GPT of this decoder was deleted"
            raise RuntimeError(msg)
        return gpt

    @classmethod
    def acquire(cls, gpt: "GPT", batch: int, length: int) -> "StaticDecoder":
        """Take `gpt`'s cached decoder for this batch size, or build one if it is missing, busy in another decode,
        too short for `length`, or its weights moved (`.to()`, dtype change)."""
        cached: dict[int, StaticDecoder] = gpt.__dict__.setdefault("_static_decoders", {})
        decoder = cached.pop(batch, None)
        if decoder is None or decoder.max_len < length or decoder.weights != weight_pointers(gpt):
            del decoder  # free the old buffers before allocating
            decoder = cls(gpt, batch, math.ceil(length / CACHE_ROUND) * CACHE_ROUND)
        return decoder

    def release(self) -> None:
        """Return the decoder to its GPT's cache, which keeps the `MAX_CACHED` most recent batch sizes."""
        gpt = self._gpt()
        if gpt is None:
            return
        cached: dict[int, StaticDecoder] = gpt.__dict__.setdefault("_static_decoders", {})
        cached.pop(self.batch, None)  # replaced by this one
        while len(cached) >= MAX_CACHED:
            cached.pop(next(iter(cached)))
        cached[self.batch] = self

    def _qkv(self, block, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, dim = hidden.shape
        qkv = block.attn.c_attn(block.ln_1(hidden)).view(batch, length, 3, self.heads, self.head_dim)
        return qkv.permute(2, 0, 3, 1, 4).unbind(0)  # 3 x [B, H, L, Dh]

    def _finish_block(self, block, hidden: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        batch, _, length, _ = attn.shape
        hidden = hidden + block.attn.c_proj(attn.transpose(1, 2).reshape(batch, length, -1))
        return hidden + block.mlp(block.ln_2(hidden))

    def _head(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.gpt.final_norm(self.gpt.gpt.ln_f(hidden[:, -1]))
        return latents, self.gpt.mel_head(latents)

    @torch.inference_mode()
    def prefill(self, emb: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the prompt `[B, L, D]` (left-padded, `mask` 0 at padding) and reset the cache to it.

        Returns the latents `[B, D]` and logits `[B, V]` predicting the first token.
        """
        length = emb.shape[1]
        if emb.shape[0] != self.batch or length >= self.max_len:
            msg = f"Prompt {tuple(emb.shape)} does not fit the decoder (batch {self.batch}, length {self.max_len})"
            raise ValueError(msg)
        mask = mask.to(torch.bool)
        eye = torch.eye(length, dtype=torch.bool, device=self.device)
        causal = torch.ones(length, length, dtype=torch.bool, device=self.device).tril()
        attn_mask = (causal & mask[:, None, None, :]) | eye  # padding queries see themselves: no empty softmax

        hidden = emb.to(self.dtype)
        for i, block in enumerate(self.gpt.gpt.h):
            query, key, value = self._qkv(block, hidden)
            self.keys[i, :, :, :length] = key
            self.values[i, :, :, :length] = value
            attn = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
            hidden = self._finish_block(block, hidden, attn)

        self.valid.zero_()
        self.valid[:, 0, 0, :length] = mask
        self.write_pos.fill_(length)
        self.mel_pos.fill_(1)
        return self._head(hidden)

    def _step(self) -> tuple[torch.Tensor, torch.Tensor]:
        gpt = self.gpt
        hidden = gpt.mel_embedding(self.tokens) + gpt.mel_pos_embedding.emb(self.mel_pos)[None]
        self.valid.index_fill_(3, self.write_pos, True)
        for i, block in enumerate(gpt.gpt.h):
            query, key, value = self._qkv(block, hidden)
            self.keys[i].index_copy_(2, self.write_pos, key)
            self.values[i].index_copy_(2, self.write_pos, value)
            attn = F.scaled_dot_product_attention(query, self.keys[i], self.values[i], attn_mask=self.valid)
            hidden = self._finish_block(block, hidden, attn)
        self.write_pos += 1
        self.mel_pos += 1
        return self._head(hidden)

    def _capture(self) -> None:
        state = (self.write_pos.clone(), self.mel_pos.clone(), self.valid.clone())
        stream = torch.cuda.Stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(2):  # warm up kernels and allocator; writes beyond write_pos are overwritten later
                self._step()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        for buffer, saved in zip((self.write_pos, self.mel_pos, self.valid), state):
            buffer.copy_(saved)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.latents, self.logits = self._step()

    @torch.inference_mode()
    def step(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Feed `tokens [B]`; returns the latents `[B, D]` and logits `[B, V]` predicting the next ones."""
        self.tokens.copy_(tokens[:, None])
        if not self.use_graph:
            return self._step()
        with torch.cuda.device(self.device):  # capture and replay on the model's GPU, not the current one
            if self.graph is None:
                self._capture()
            self.graph.replay()
        return self.latents, self.logits
