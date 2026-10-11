"""The static-cache decoder must reproduce the eager `transformers` decode steps up to float rounding."""

import pytest
import torch

from TTS.tts.layers.xtts import static_decoder
from TTS.tts.layers.xtts.decoding import EagerStepper, Sampling, decode
from TTS.tts.layers.xtts.gpt import GPT

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def make_gpt(device):
    torch.manual_seed(0)
    model = GPT(
        layers=3,
        model_dim=64,
        heads=4,
        max_text_tokens=40,
        max_mel_tokens=60,
        max_prompt_tokens=10,
        number_text_tokens=300,
        num_audio_tokens=40,
        start_audio_token=38,
        stop_audio_token=39,
    )
    model.init_gpt_for_inference(kv_cache=True)
    return model.eval().to(device)


def prompt(gpt, text_lengths, device):
    """Left-padded prompt embeddings (prefix + start token) and mask for texts of the given lengths."""
    cond = torch.randn(1, 3, 64, device=device)
    rows = [gpt.prefix_embeddings(cond, torch.randint(0, 255, (1, n), device=device))[0] for n in text_lengths]
    start = gpt.mel_embedding(torch.tensor([gpt.start_audio_token], device=device))
    rows = [torch.cat([row, start + gpt.mel_pos_embedding.get_fixed_embedding(0, device)[0]]) for row in rows]
    width = max(row.shape[0] for row in rows)
    emb = torch.stack([torch.nn.functional.pad(row, (0, 0, width - row.shape[0], 0)) for row in rows])
    mask = torch.stack([(torch.arange(width, device=device) >= width - row.shape[0]).long() for row in rows])
    return emb, mask


def teacher_forced(stepper, emb, mask, tokens):
    outputs = [stepper.prefill(emb, mask)]
    for column in tokens.T:
        outputs.append(tuple(t.clone() for t in stepper.step(column)))
    return torch.stack([lat for lat, _ in outputs]), torch.stack([log for _, log in outputs])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("text_lengths", [[12], [5, 12, 9]])
def test_static_matches_eager(device, text_lengths):
    gpt = make_gpt(device)
    torch.manual_seed(1)
    emb, mask = prompt(gpt, text_lengths, device)
    tokens = torch.randint(0, 38, (len(text_lengths), 15), device=device)
    with torch.inference_mode():
        expected = teacher_forced(EagerStepper(gpt), emb, mask, tokens)
        decoder = static_decoder.StaticDecoder(gpt, len(text_lengths), 64)
        actual = teacher_forced(decoder, emb, mask, tokens)
    assert decoder.use_graph == (device == "cuda")
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def acquire(gpt, batch, length):
    return static_decoder.StaticDecoder.acquire(gpt, batch, length)


@pytest.mark.parametrize("device", DEVICES)
def test_decoder_is_reused_across_prompts(device):
    gpt = make_gpt(device)
    first = None
    with torch.inference_mode():
        for seed in range(3):
            torch.manual_seed(seed)
            emb, mask = prompt(gpt, [4 + 5 * seed], device)
            tokens = torch.randint(0, 38, (1, 10), device=device)
            expected = teacher_forced(EagerStepper(gpt), emb, mask, tokens)
            decoder = acquire(gpt, 1, 70)
            first = first or decoder
            assert decoder is first
            actual = teacher_forced(decoder, emb, mask, tokens)
            decoder.release()
            for got, want in zip(actual, expected):
                torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
    assert (first.graph is not None) == (device == "cuda")


def test_acquire_builds_when_needed():
    gpt = make_gpt("cpu")
    decoder = acquire(gpt, 1, 10)
    assert decoder.max_len == static_decoder.CACHE_ROUND
    assert acquire(gpt, 1, 10) is not decoder  # busy until released
    decoder.release()
    assert acquire(gpt, 1, decoder.max_len) is decoder
    decoder.release()
    longer = acquire(gpt, 1, decoder.max_len + 1)
    assert longer.max_len == 2 * static_decoder.CACHE_ROUND
    longer.release()
    for batch in (2, 3):  # keeps the most recent MAX_CACHED batch sizes
        acquire(gpt, batch, 10).release()
    assert set(gpt._static_decoders) == {2, 3}


def test_acquire_rebuilds_after_weights_move():
    gpt = make_gpt("cpu")
    decoder = acquire(gpt, 1, 10)
    decoder.release()
    gpt.to(torch.bfloat16).to(torch.float32)  # same dtype and device, new storage
    assert acquire(gpt, 1, 10) is not decoder


def test_decoder_does_not_keep_the_gpt_alive():
    import gc
    import weakref

    gpt = make_gpt("cpu")
    acquire(gpt, 1, 10).release()
    alive = weakref.ref(gpt)
    gc.disable()
    try:
        del gpt
        assert alive() is None
    finally:
        gc.enable()


@pytest.mark.parametrize("device", DEVICES)
def test_interleaved_decodes_do_not_share_a_cache(device):
    gpt = make_gpt(device)
    cond = torch.randn(1, 3, 64, device=device)
    texts = [torch.randint(0, 255, (1, n), device=device) for n in (6, 11)]
    sampling = Sampling(do_sample=False)

    def run(text):
        return decode(gpt, gpt.prefix_embeddings(cond, text), sampling=sampling, cuda_graph=True, max_new_tokens=12)

    with torch.inference_mode():
        expected = [torch.stack([lat for _, lat in run(t)]) for t in texts]
        first, second = run(texts[0]), run(texts[1])
        latents = [[], []]
        for pair in zip(first, second):  # step both decodes in lockstep
            for out, (_, lat) in zip(latents, pair):
                out.append(lat)
    for got, want in zip(latents, expected):
        n = min(len(got), len(want))
        torch.testing.assert_close(torch.stack(got[:n]), want[:n], rtol=1e-4, atol=1e-4)


def test_prefill_rejects_oversized_prompt():
    gpt = make_gpt("cpu")
    emb, mask = prompt(gpt, [12], "cpu")
    with pytest.raises(ValueError, match="does not fit"):
        static_decoder.StaticDecoder(gpt, 1, emb.shape[1]).prefill(emb, mask)


@pytest.mark.parametrize("device", DEVICES)
def test_decode_with_cuda_graph_flag_runs_to_stop(device):
    gpt = make_gpt(device)
    cond, text = torch.randn(1, 3, 64, device=device), torch.randint(0, 255, (1, 8), device=device)
    with torch.inference_mode():
        steps = list(decode(gpt, gpt.prefix_embeddings(cond, text), sampling=Sampling(do_sample=True), cuda_graph=True))
    tokens = torch.stack([t for t, _ in steps], dim=1)
    assert tokens.shape[1] <= gpt.max_gen_mel_tokens
    assert tokens[0, -1] == gpt.stop_audio_token or tokens.shape[1] == gpt.max_gen_mel_tokens
    assert (tokens[0, :-1] != gpt.stop_audio_token).all()


def test_unsupported_gpt_falls_back(monkeypatch):
    gpt = make_gpt("cpu")
    monkeypatch.setattr(gpt.mel_pos_embedding, "relative", True)
    assert not static_decoder.supported(gpt)


def test_clear_static_decoders():
    gpt = make_gpt("cpu")
    acquire(gpt, 1, 10).release()
    gpt.clear_static_decoders()
    assert not getattr(gpt, "_static_decoders", None)
    gpt.clear_static_decoders()  # idempotent
