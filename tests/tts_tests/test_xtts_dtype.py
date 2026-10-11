"""XTTS with a half-precision GPT: only the GPT changes dtype, and every path still runs."""

import pytest
import torch

from TTS.tts.configs.xtts_config import XttsArgs, XttsConfig
from TTS.tts.layers.tortoise.autoregressive import NullPositionEmbeddings
from TTS.tts.layers.xtts.decoding import Sampling, decode
from TTS.tts.models.xtts import Xtts

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def make_xtts(device="cpu"):
    torch.manual_seed(0)
    args = XttsArgs(
        gpt_layers=2,
        gpt_n_model_channels=64,
        gpt_n_heads=2,
        gpt_number_text_tokens=300,
        gpt_num_audio_tokens=40,
        gpt_start_audio_token=38,
        gpt_stop_audio_token=39,
        gpt_max_audio_tokens=60,
        gpt_use_perceiver_resampler=True,
        decoder_input_dim=64,
    )
    model = Xtts(XttsConfig(model_args=args))
    model.gpt.init_gpt_for_inference(kv_cache=True)
    return model.eval().to(device)


def test_null_position_embeddings_follow_dtype_without_state():
    module = NullPositionEmbeddings(8)
    assert module(torch.zeros(2, 3, dtype=torch.long)).shape == (2, 3, 8)
    assert module.to(torch.bfloat16)(torch.zeros(1, 1, dtype=torch.long)).dtype == torch.bfloat16
    assert module.state_dict() == {}


def test_set_gpt_dtype_casts_only_the_gpt():
    model = make_xtts()
    model.set_gpt_dtype("bf16")
    gpt = model.gpt
    assert {p.dtype for p in gpt.gpt.parameters()} == {torch.bfloat16}
    assert gpt.mel_embedding.weight.dtype == gpt.mel_head.weight.dtype == torch.bfloat16
    assert gpt.gpt.wpe(torch.zeros(1, 1, dtype=torch.long)).dtype == torch.bfloat16
    assert {p.dtype for p in gpt.conditioning_encoder.parameters()} == {torch.float32}
    assert {p.dtype for p in model.hifigan_decoder.parameters()} == {torch.float32}
    model.set_gpt_dtype(None)
    assert {p.dtype for p in gpt.gpt.parameters()} == {torch.float32}


def test_set_gpt_dtype_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown GPT dtype"):
        make_xtts().set_gpt_dtype("int8")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("cuda_graph", [False, True])
def test_half_gpt_generates_latents_and_audio(device, dtype, cuda_graph):
    if device == "cpu" and dtype == "fp16":
        pytest.skip("fp16 matmul on CPU is slow/partial")
    model = make_xtts(device)
    model.set_gpt_dtype(dtype)
    cond = torch.randn(1, 32, 64, device=device)  # float32, as the perceiver returns them
    text = torch.randint(0, 255, (1, 10), device=device)
    with torch.inference_mode():
        steps = list(
            decode(model.gpt, model.gpt.prefix_embeddings(cond, text), sampling=Sampling(), cuda_graph=cuda_graph)
        )
        codes = torch.stack([t for t, _ in steps], dim=1)
        latents = model.gpt(
            text,
            torch.tensor([text.shape[-1]], device=device),
            codes,
            torch.tensor([codes.shape[-1] * model.gpt.code_stride_len], device=device),
            cond_latents=cond,
            return_latent=True,
        )
        wav = model.hifigan_decoder(latents, g=torch.randn(1, 512, 1, device=device))
    assert latents.dtype == model.gpt.mel_head.weight.dtype
    assert torch.isfinite(latents).all()
    assert wav.dtype == torch.float32 and torch.isfinite(wav).all()
