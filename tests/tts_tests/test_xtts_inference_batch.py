"""Batched XTTS inference must match one-by-one inference for greedy decoding."""

import numpy as np
import pytest
import torch

from TTS.tts.configs.xtts_config import XttsArgs, XttsConfig
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
from TTS.tts.models.xtts import Xtts

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
TEXTS = ["Hello world.", "A much longer sentence, with a comma in it.", "Short.", "Medium length text here.", "Hi."]


@pytest.fixture(scope="module", params=DEVICES)
def xtts(request):
    torch.manual_seed(0)
    tokenizer = VoiceBpeTokenizer(vocab_file="tests/inputs/xtts_vocab.json")
    args = XttsArgs(
        gpt_layers=2,
        gpt_n_model_channels=64,
        gpt_n_heads=2,
        gpt_number_text_tokens=tokenizer.tokenizer.get_vocab_size(),
        gpt_num_audio_tokens=40,
        gpt_start_audio_token=38,
        gpt_stop_audio_token=39,
        gpt_max_audio_tokens=40,
        gpt_use_perceiver_resampler=True,
        decoder_input_dim=64,
    )
    model = Xtts(XttsConfig(model_args=args))
    model.tokenizer = tokenizer
    model.gpt.init_gpt_for_inference(kv_cache=True)
    model = model.eval().to(request.param)
    voice = (torch.randn(1, 32, 64, device=request.param), torch.randn(1, 512, 1, device=request.param))
    return model, voice


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_greedy_batch_matches_sequential(xtts, batch_size):
    model, voice = xtts
    with torch.inference_mode():
        expected = [model.inference(text, "en", *voice, do_sample=False)["wav"] for text in TEXTS]
        outputs = model.inference_batch(TEXTS, "en", *voice, batch_size=batch_size, do_sample=False)
    assert len(outputs) == len(TEXTS)
    for out, want in zip(outputs, expected):
        assert out["wav"].shape == want.shape
        np.testing.assert_allclose(out["wav"], want, rtol=1e-4, atol=1e-4)


def test_cuda_graph_batch_runs(xtts):
    model, voice = xtts
    with torch.inference_mode():
        outputs = model.inference_batch(TEXTS, "en", *voice, batch_size=2, cuda_graph=True)
    assert [o["wav"].ndim for o in outputs] == [1] * len(TEXTS)
    assert all(np.isfinite(o["wav"]).all() for o in outputs)


def test_unsupported_options_fall_back_to_sequential(xtts, monkeypatch):
    model, voice = xtts
    calls = []
    monkeypatch.setattr(model, "inference", lambda text, *a, **k: calls.append((text, k)) or {"wav": text})
    outputs = model.inference_batch(TEXTS[:2], "en", *voice, num_beams=2)
    assert [o["wav"] for o in outputs] == TEXTS[:2]
    assert all(k["num_beams"] == 2 for _, k in calls)
