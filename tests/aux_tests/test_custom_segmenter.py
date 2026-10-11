import pytest

from TTS.utils.custom_segmenter import CHAR_LIMITS, DEFAULT_LIMIT, Segmenter, middle_comma_split, subsegment


@pytest.mark.parametrize(
    ("language", "limit"),
    [("en", 250), ("en-US", 250), ("zh-cn", 82), ("PL", 224), ("xx", DEFAULT_LIMIT)],
)
def test_limit_per_language(language, limit):
    assert Segmenter(language).limit == limit


def test_limit_override():
    assert Segmenter("en", limit=10).limit == 10


def test_char_limits_match_xtts_tokenizer():
    tokenizer = pytest.importorskip("TTS.tts.layers.xtts.tokenizer")
    assert CHAR_LIMITS == {k.split("-")[0]: v for k, v in tokenizer.VoiceBpeTokenizer().char_limits.items()}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", []),
        ("Hello. Two sentences", ["Hello. ", "Two sentences "]),
        ("This line wraps\nmid sentence.", ["This line wraps mid sentence. "]),
        ("A heading\nBody text.", ["A heading ", "Body text. "]),
        ("Wait - what?", ["Wait. ", "What? "]),
        ("Wait — what?", ["Wait. ", "What? "]),
        ("Use e.g. this one.", ["Use e.g. this one. "]),
    ],
)
def test_segment(text, expected):
    assert Segmenter("en").segment(text) == expected


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("en", "Long clause, " * 60 + "end."),
        ("en", "One; two; " * 40 + "three."),
        ("zh-cn", "今天天气很好，" * 30 + "明天见。"),
    ],
)
def test_segment_fits_limit(language, text):
    segmenter = Segmenter(language)
    sentences = segmenter.segment(text)
    assert all(len(s.strip()) <= segmenter.limit for s in sentences)
    assert "".join(sentences).replace(" ", "") == text.replace(" ", "")


def test_segment_unsplittable_kept():
    assert Segmenter("en").segment("a" * 300) == ["a" * 300 + " "]


def test_subsegment_keeps_separator():
    assert subsegment("a. b. c", ". ") == ["a. ", "b. ", "c"]
    assert subsegment("a. b.", ". ") == ["a. ", "b. "]


def test_middle_comma_split_trailing_comma():
    assert middle_comma_split("a" * 20 + ",", 10) == ["a" * 20 + ","]
