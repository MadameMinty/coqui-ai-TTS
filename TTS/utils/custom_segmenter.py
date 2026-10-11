"""Sentence segmenter for XTTS: yasbd sentences, sub-split until each fits the
model's per-language character limit. Deployed beside TTS/utils/synthesizer.py
as custom_segmenter.py; Synthesizer._set_segmenters builds one per language."""

import re

from yasbd import BoundaryDetector, get_supported_langs

# XTTS input limits per language (TTS/tts/layers/xtts/tokenizer.py, VoiceBpeTokenizer.char_limits)
CHAR_LIMITS: dict[str, int] = {
    "en": 250,
    "de": 253,
    "fr": 273,
    "es": 239,
    "it": 213,
    "pt": 203,
    "pl": 224,
    "zh": 82,
    "ar": 166,
    "cs": 186,
    "ru": 182,
    "nl": 251,
    "tr": 226,
    "ja": 71,
    "hu": 224,
    "ko": 95,
    "hi": 150,
}
DEFAULT_LIMIT: int = 250
SEPARATORS: list[str] = ["... ", ". ", "; ", ": "]  # tried in order on over-long sentences
COMMAS: str = ",，、"
# ". x" / ",. x" after a word; skips ellipses ("wait... what") and dotted abbreviations ("e.g. this")
MID_PERIOD: re.Pattern = re.compile(r"(?<=[^\W\d_]|,)(?<!\.\w)\. [^\W\d_]")


def subsegment(string: str, sep: str = ". ") -> list[str]:
    """Splits on `sep`, keeping the separator on each piece."""
    terminate = string.endswith((sep, sep.strip()))
    sentences = string.strip(sep).split(sep)
    last = sentences[-1]
    return [s + sep for s in sentences[:-1]] + [last + sep if terminate else last]


def subsegment_loop(sentences: list[str], separators: list[str], limit: int) -> list[str]:
    for sep in separators:
        sentences = flatten([subsegment(s, sep) if len(s) > limit and sep.strip() in s else s for s in sentences])
    return sentences


def middle_comma_split(sentence: str, limit: int) -> list[str]:
    """Halves a sentence at its middle comma until each piece fits `limit` or has no inner comma."""
    comma_positions = [pos for pos, char in enumerate(sentence[:-1]) if char in COMMAS]
    if len(sentence) <= limit or not comma_positions:
        return [sentence]
    split_index = comma_positions[len(comma_positions) // 2] + 1
    return middle_comma_split(sentence[:split_index], limit) + middle_comma_split(sentence[split_index:], limit)


def flatten(partially_nested_list: list[str | list[str]]) -> list[str]:
    flat: list[str] = []
    for item in partially_nested_list:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


class Segmenter:
    """yasbd segmenter for `language` ("pl", "en-US", "zh-cn", ...) capped at the XTTS limit.

    A line wrapped mid-sentence stays one sentence; a line without terminal punctuation
    (heading) is its own. Languages unknown to yasbd fall back to its English rules,
    unknown limits to DEFAULT_LIMIT.
    """

    def __init__(self, language: str = "en", limit: int | None = None) -> None:
        self.language: str = language.split("-")[0].lower()
        self.limit: int = CHAR_LIMITS.get(self.language, DEFAULT_LIMIT) if limit is None else limit
        rules = self.language if self.language in get_supported_langs() else "en"
        self.detector: BoundaryDetector = BoundaryDetector(lang=rules)

    def segment(self, text: str) -> list[str]:
        # dashes read as pauses; turn them into sentence boundaries
        text = text.replace(" - ", ". ").replace(" – ", ". ").replace(" — ", ". ")
        # capitalize after a mid-text period (lexicon dash, "sword. pale") so yasbd splits there:
        # XTTS often stops generating at a second period within one call. Inaudible, XTTS lowercases.
        text = MID_PERIOD.sub(lambda m: m[0].upper(), text)

        # whitespace runs (wrapped lines) collapse inside a sentence
        sentences = [re.sub(r"\s+", " ", s).strip() for s in self.detector.segment(text)]
        sentences = [s for s in sentences if s]
        sentences = subsegment_loop(sentences, SEPARATORS, self.limit)
        # last resort: halve at the middle comma
        sentences = flatten([middle_comma_split(s, self.limit) for s in sentences])

        return [s.strip() + " " for s in sentences if s.strip()]
