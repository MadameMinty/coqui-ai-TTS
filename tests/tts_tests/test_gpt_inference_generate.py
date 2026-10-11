"""Run cached autoregressive generation for Tortoise and XTTS with tiny random models.

transformers >= 5.18 drops an all-ones attention_mask in generate(), so the cached
(single-token) forward steps of both GPT2InferenceModel classes receive attention_mask=None.
"""

import pytest
import torch

from TTS.tts.layers.tortoise.autoregressive import UnifiedVoice
from TTS.tts.layers.xtts.gpt import GPT

MAX_NEW_TOKENS = 8
SAMPLING = {"do_sample": True, "top_p": 0.8, "temperature": 0.8}


@pytest.mark.parametrize("num_return_sequences", [1, 2])
def test_tortoise_inference_speech_with_kv_cache(num_return_sequences):
    torch.manual_seed(0)
    model = UnifiedVoice(
        layers=2,
        model_dim=64,
        heads=2,
        max_text_tokens=40,
        max_mel_tokens=60,
        max_conditioning_inputs=1,
        number_text_tokens=255,
        start_text_token=255,
        checkpointing=False,
    )
    model.post_init_gpt2_config(kv_cache=True)
    model.eval()
    with torch.inference_mode():
        codes = model.inference_speech(
            torch.randn(1, 64),
            torch.randint(0, 255, (1, 12)),
            num_return_sequences=num_return_sequences,
            max_generate_length=MAX_NEW_TOKENS,
            **SAMPLING,
        )
    assert codes.shape == (num_return_sequences, MAX_NEW_TOKENS)


def test_xtts_gpt_generate_with_kv_cache():
    torch.manual_seed(0)
    model = GPT(
        layers=2,
        model_dim=64,
        heads=2,
        max_text_tokens=40,
        max_mel_tokens=60,
        max_prompt_tokens=10,
        number_text_tokens=300,
    )
    model.init_gpt_for_inference(kv_cache=True)
    model.eval()
    with torch.inference_mode():
        codes = model.generate(
            torch.randn(1, 3, 64),
            torch.randint(0, 255, (1, 12)),
            max_new_tokens=MAX_NEW_TOKENS,
            **SAMPLING,
        )
    assert codes.shape == (1, MAX_NEW_TOKENS)
