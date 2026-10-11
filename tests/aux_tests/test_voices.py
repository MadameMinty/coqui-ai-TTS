import logging

import numpy as np
import pytest
import torch

from TTS.tts.models.bark import Bark
from TTS.utils.voices import CloningMixin


class DummyCloner(CloningMixin):
    def _clone_voice(self, speaker_wav, **kwargs):
        return {"emb": torch.ones(2)}, {"name": "dummy"}


@pytest.fixture
def cloner():
    return DummyCloner()


@pytest.fixture
def bark():
    # Voice file handling doesn't depend on model state, so skip loading weights
    return Bark.__new__(Bark)


@pytest.mark.parametrize("speaker_id", ["my.voice", "Craig Gutsy", "Zoë", "../escape"])
def test_clone_and_load_voice(cloner, tmp_path, speaker_id):
    cloner.clone_voice("x.wav", speaker_id, tmp_path)
    voice = cloner.clone_voice(None, speaker_id, tmp_path)
    assert torch.equal(voice["emb"], torch.ones(2))


@pytest.mark.parametrize("load_id", ["my.voice", "my_voice", "my voice"])
def test_load_voice_equivalent_ids(cloner, tmp_path, load_id):
    cloner.clone_voice("x.wav", "my.voice", tmp_path)
    voice = cloner.load_voice_file(load_id, tmp_path)
    assert torch.equal(voice["emb"], torch.ones(2))


def test_clone_voice_overwrites_same_slug(cloner, tmp_path, caplog):
    cloner.clone_voice("x.wav", "my.voice", tmp_path)
    with caplog.at_level(logging.INFO, logger="TTS.utils.voices"):
        cloner.clone_voice("x.wav", "my voice", tmp_path)
    assert "already exists" in caplog.text
    assert list(cloner.get_voices(tmp_path)) == ["my_voice"]


def test_clone_voice_file_and_metadata(cloner, tmp_path):
    voice_dir = tmp_path / "voices"
    cloner.clone_voice("x.wav", "my.voice", voice_dir)
    voice_fn = voice_dir / "my_voice.pth"
    assert voice_fn.is_file()
    voice = torch.load(voice_fn, weights_only=True)
    assert voice["metadata"]["speaker_id"] == "my_voice"
    assert voice["metadata"]["source_files"] == ["x.wav"]

    cloner.clone_voice("x.wav", "../escape", voice_dir)
    assert (voice_dir / "escape.pth").is_file()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["voices"]


def test_load_voice_file_not_found(cloner, tmp_path):
    with pytest.raises(FileNotFoundError, match=r"`my_voice\.pth` for speaker `my\.voice`"):
        cloner.load_voice_file("my.voice", tmp_path)


def test_bark_load_npz_voice(bark, tmp_path):
    prompts = {
        "semantic_prompt": np.arange(3),
        "coarse_prompt": np.ones((2, 3)),
        "fine_prompt": np.zeros((2, 3)),
    }
    np.savez(tmp_path / "my_voice.npz", **prompts)
    voice = bark.load_voice_file("my.voice", tmp_path)
    assert set(voice) == set(prompts)
    for key, value in prompts.items():
        assert torch.equal(voice[key], torch.tensor(value))


def test_bark_load_pth_voice(bark, tmp_path):
    torch.save({"semantic_prompt": torch.arange(3)}, tmp_path / "my_voice.pth")
    voice = bark.load_voice_file("my.voice", tmp_path)
    assert torch.equal(voice["semantic_prompt"], torch.arange(3))


def test_bark_load_voice_file_not_found(bark, tmp_path):
    with pytest.raises(FileNotFoundError, match=r"`my_voice\.pth` or \.npz for speaker `my\.voice`"):
        bark.load_voice_file("my.voice", tmp_path)


class CountingCloner(DummyCloner):
    def __init__(self):
        self.calls = 0

    def _clone_voice(self, speaker_wav, **kwargs):
        self.calls += 1
        return super()._clone_voice(speaker_wav, **kwargs)


def test_clone_voice_reuses_same_reference_audio(tmp_path):
    wav = tmp_path / "ref.wav"
    wav.write_bytes(b"one")
    cloner = CountingCloner()
    first = cloner.clone_voice(str(wav), gpt_cond_len=6)
    second = cloner.clone_voice(wav, gpt_cond_len=6)
    assert cloner.calls == 1
    assert torch.equal(first["emb"], second["emb"]) and first is not second
    cloner.clone_voice(wav, gpt_cond_len=3)  # other settings
    assert cloner.calls == 2
    wav.write_bytes(b"changed")  # other size
    cloner.clone_voice(wav, gpt_cond_len=6)
    assert cloner.calls == 3


def test_clone_voice_cache_still_saves_voice_file(tmp_path):
    wav = tmp_path / "ref.wav"
    wav.write_bytes(b"one")
    cloner = CountingCloner()
    cloner.clone_voice(wav)
    cloner.clone_voice(wav, "cached", tmp_path / "voices")
    assert cloner.calls == 1
    assert (tmp_path / "voices" / "cached.pth").is_file()


def test_clone_voice_cache_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr("TTS.utils.voices.CLONE_CACHE_SIZE", 2)
    cloner = CountingCloner()
    wavs = []
    for i in range(3):
        wavs.append(tmp_path / f"{i}.wav")
        wavs[-1].write_bytes(b"x")
        cloner.clone_voice(wavs[-1])
    cloner.clone_voice(wavs[0])  # evicted
    cloner.clone_voice(wavs[2])  # kept
    assert cloner.calls == 4


class WeightedCloner(CountingCloner, torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        CountingCloner.__init__(self)
        self.weight = torch.nn.Parameter(torch.zeros(2))


def test_clone_voice_cache_follows_weight_updates(tmp_path):
    wav = tmp_path / "ref.wav"
    wav.write_bytes(b"one")
    cloner = WeightedCloner()
    cloner.clone_voice(wav)
    cloner.clone_voice(wav)
    assert cloner.calls == 1
    with torch.no_grad():
        cloner.weight.add_(1)  # an optimizer step or load_state_dict
    cloner.clone_voice(wav)
    assert cloner.calls == 2
    with torch.inference_mode():
        cloner.weight = torch.nn.Parameter(torch.ones(2))  # inference tensor: no version counter
        cloner.clone_voice(wav)
        cloner.clone_voice(wav)
    assert cloner.calls == 3


def test_models_can_opt_out_of_the_clone_cache(tmp_path):
    wav = tmp_path / "ref.wav"
    wav.write_bytes(b"one")
    cloner = CountingCloner()
    cloner.cache_cloned_voices = False
    cloner.clone_voice(wav)
    cloner.clone_voice(wav)
    assert cloner.calls == 2
