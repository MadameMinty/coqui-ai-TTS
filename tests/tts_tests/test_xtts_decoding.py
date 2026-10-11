"""The owned XTTS decode loop must reproduce `transformers` `generate()` token for token."""

import pytest
import torch

from TTS.tts.layers.xtts.decoding import Sampling, decode
from TTS.tts.layers.xtts.gpt import GPT

SAMPLING = [
    {"do_sample": False},
    {"do_sample": False, "repetition_penalty": 10.0},
    {"do_sample": True, "temperature": 0.75, "top_k": 50, "top_p": 0.85, "repetition_penalty": 10.0},
    {"do_sample": True, "temperature": 1.3, "top_k": 5, "top_p": 1.0, "repetition_penalty": 2.0},
]


@pytest.fixture(scope="module")
def gpt():
    torch.manual_seed(0)
    model = GPT(
        layers=2,
        model_dim=64,
        heads=2,
        max_text_tokens=40,
        max_mel_tokens=60,
        max_prompt_tokens=10,
        number_text_tokens=300,
        num_audio_tokens=40,
        start_audio_token=38,
        stop_audio_token=39,
    )
    model.init_gpt_for_inference(kv_cache=True)
    return model.eval()


def run_decode(gpt, cond, text, sampling, **kwargs):
    prefix = gpt.prefix_embeddings(cond, text)
    steps = list(decode(gpt, prefix, sampling=Sampling(**sampling), **kwargs))
    return torch.stack([t for t, _ in steps], dim=1), torch.stack([lat for _, lat in steps], dim=1)


@pytest.mark.parametrize("sampling", SAMPLING)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_decode_matches_generate(gpt, sampling, seed):
    torch.manual_seed(100 + seed)
    cond, text = torch.randn(1, 3, 64), torch.randint(0, 255, (1, 12))
    with torch.inference_mode():
        torch.manual_seed(seed)
        expected = gpt.generate(cond, text, use_owned_decoder=False, **sampling)
        torch.manual_seed(seed)
        tokens, _ = run_decode(gpt, cond, text, sampling)
    assert torch.equal(tokens, expected)


def test_decode_latents_match_generate_hidden_states(gpt):
    torch.manual_seed(3)
    cond, text = torch.randn(1, 3, 64), torch.randint(0, 255, (1, 12))
    with torch.inference_mode():
        torch.manual_seed(0)
        _, gen = gpt.generate(
            cond,
            text,
            use_owned_decoder=False,
            do_sample=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        expected = torch.stack([gpt.final_norm(h[-1][:, -1]) for h in gen.hidden_states], dim=1)
        torch.manual_seed(0)
        _, latents = run_decode(gpt, cond, text, {"do_sample": True})
    torch.testing.assert_close(latents, expected)


def test_decode_left_padded_batch_matches_single_rows(gpt):
    torch.manual_seed(4)
    cond = torch.randn(1, 3, 64)
    texts = [torch.randint(0, 255, (1, n)) for n in (5, 12, 9)]
    with torch.inference_mode():
        singles = [run_decode(gpt, cond, t, {"do_sample": False}, max_new_tokens=20)[0][0] for t in texts]
        prefixes = [gpt.prefix_embeddings(cond, t)[0] for t in texts]
        width = max(p.shape[0] for p in prefixes)
        emb = torch.stack([torch.nn.functional.pad(p, (0, 0, width - p.shape[0], 0)) for p in prefixes])
        mask = torch.stack([(torch.arange(width) >= width - p.shape[0]).long() for p in prefixes])
        steps = list(decode(gpt, emb, mask, sampling=Sampling(), max_new_tokens=20))
    batch = torch.stack([t for t, _ in steps], dim=1)
    for row, single in zip(batch, singles):
        assert torch.equal(row[: len(single)], single)
        assert (row[len(single) :] == gpt.stop_audio_token).all()


def test_decode_stops_at_max_new_tokens(gpt, monkeypatch):
    monkeypatch.setattr(gpt, "stop_audio_token", 10_000)  # never sampled
    cond, text = torch.randn(1, 3, 64), torch.randint(0, 255, (1, 4))
    with torch.inference_mode():
        tokens, latents = run_decode(gpt, cond, text, {"do_sample": True}, max_new_tokens=3)
    assert tokens.shape == (1, 3)
    assert latents.shape == (1, 3, 64)


def test_sampling_from_kwargs_pops_known_keys():
    kwargs = {"temperature": 0.5, "top_k": None, "num_beams": 1}
    sampling = Sampling.from_kwargs(kwargs)
    assert sampling.temperature == 0.5
    assert sampling.top_k == 50
    assert kwargs == {"num_beams": 1}
