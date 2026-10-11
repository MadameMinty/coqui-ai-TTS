"""Synthesizer batching with a stub model: same waveforms as one-by-one, regrouped per text."""

from types import SimpleNamespace

import numpy as np
import pytest

from TTS.utils.synthesizer import PAD_SILENCE_SAMPLES, Synthesizer


class StubModel:
    """A sentence becomes len(text) samples, each equal to len(text)."""

    def __init__(self, batched: bool):
        self.calls = []
        if batched:
            self.synthesize_batch = self._synthesize_batch

    @staticmethod
    def _wav(text):
        return {"wav": np.full(len(text), float(len(text)))}

    def synthesize(self, text, **kwargs):
        self.calls.append(("one", text, kwargs))
        return self._wav(text)

    def _synthesize_batch(self, texts, batch_size, **kwargs):
        self.calls.append(("batch", texts, batch_size, kwargs))
        return [self._wav(text) for text in texts]


class StubSegmenter:
    def segment(self, text):
        return [s.strip() + "." for s in text.split(".") if s.strip()]


def make_synthesizer(batched: bool) -> Synthesizer:
    synth = object.__new__(Synthesizer)
    synth.tts_model = StubModel(batched)
    synth.vocoder_model = None
    synth.use_cuda = False
    synth.voice_dir = None
    synth.tts_config = SimpleNamespace(audio={"sample_rate": 100})
    synth.segmenter = {"en": StubSegmenter()}
    return synth


def expected_wav(text):
    out = []
    for sentence in StubSegmenter().segment(text):
        out += [float(len(sentence))] * len(sentence) + [0] * PAD_SILENCE_SAMPLES
    return out


TEXTS = ["One. Three three.", "Two two."]


@pytest.mark.parametrize("batched", [True, False])
def test_tts_batch_regroups_per_text(batched):
    synth = make_synthesizer(batched)
    wavs = synth.tts_batch(TEXTS, language_name="en", batch_size=4, temperature=0.5)
    assert wavs == [expected_wav(t) for t in TEXTS]
    if batched:
        [(kind, sentences, batch_size, kwargs)] = synth.tts_model.calls
        assert (kind, sentences, batch_size) == ("batch", ["One.", "Three three.", "Two two."], 4)
        assert kwargs["temperature"] == 0.5 and kwargs["language"] == "en"
    else:
        assert [c[0] for c in synth.tts_model.calls] == ["one"] * 3


@pytest.mark.parametrize("batch_size", [1, 3])
def test_tts_batch_size_matches_sequential(batch_size):
    synth = make_synthesizer(batched=True)
    assert synth.tts(TEXTS[0], language_name="en", batch_size=batch_size) == expected_wav(TEXTS[0])
    assert synth.tts_model.calls[0][0] == ("batch" if batch_size > 1 else "one")
