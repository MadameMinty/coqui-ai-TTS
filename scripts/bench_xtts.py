"""Benchmark XTTS inference: real-time factor, streaming latency, peak VRAM and optional ASR word error rate.

python scripts/bench_xtts.py --out bench.json
python scripts/bench_xtts.py --load gpt_dtype=bf16 --infer temperature=0.7 --batch-size 4
python scripts/bench_xtts.py --asr openai/whisper-large-v3-turbo --wav-dir out/
"""

import argparse
import json
import logging
import statistics
import string
import time
from pathlib import Path
from typing import Any

import torch

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.custom_segmenter import Segmenter
from TTS.utils.manage import ModelManager

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
SAMPLE_RATE = 24000

TEXTS = {
    "en-short": ("en", "The quick brown fox jumps over the lazy dog."),
    "en-medium": (
        "en",
        "It was a bright cold day in April, and the clocks were striking thirteen. Winston Smith, his chin nuzzled "
        "into his breast in an effort to escape the vile wind, slipped quickly through the glass doors of Victory "
        "Mansions.",
    ),
    "en-long": (
        "en",
        "The village lay at the bottom of a valley, hidden from the road by a line of old oaks. Every morning the "
        "baker opened his shutters before dawn, and the smell of fresh bread drifted across the square. Children "
        "ran to school along the river, counting the ducks as they went. In the afternoons the old men played "
        "chess under the chestnut tree, arguing about moves they had made forty years ago. Nobody there was in a "
        "hurry, and nobody wanted to be.",
    ),
    "de-medium": (
        "de",
        "Am Abend zog ein Gewitter über die Stadt. Die Straßen leerten sich schnell, und nur ein einzelner Hund "
        "lief noch über den nassen Marktplatz.",
    ),
}


def parse_kv(pairs: list[str]) -> dict[str, Any]:
    """`key=value` pairs; values parsed as JSON when possible (numbers, booleans), else kept as strings."""
    out = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def normalize_words(text: str) -> list[str]:
    table = str.maketrans("", "", string.punctuation + "„“”‚‘’«»…–—")
    return text.lower().translate(table).split()


def wer(ref: str, hyp: str) -> float:
    r, h = normalize_words(ref), normalize_words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(len(r), 1)


def load_model(model_dir: Path | None, device: str, load_kwargs: dict[str, Any]) -> Xtts:
    if model_dir is None:
        model_path, _, _ = ModelManager(progress_bar=False).download_model(MODEL_NAME)
        model_dir = model_path if model_path.is_dir() else model_path.parent
    config = XttsConfig()
    config.load_json(str(Path(model_dir) / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=str(model_dir), eval=True, **load_kwargs)
    return model.to(device)


def synth(model: Xtts, sentences: list[str], language: str, latents, batch_size: int, seed: int, infer: dict):
    torch.manual_seed(seed)
    if batch_size > 1:
        outputs = model.inference_batch(sentences, language, *latents, batch_size=batch_size, **infer)
        wavs = [out["wav"] for out in outputs]
    else:
        wavs = [model.inference(s, language, *latents, **infer)["wav"] for s in sentences]
    return torch.cat([torch.as_tensor(w) for w in wavs]).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, help="XTTS checkpoint dir; default: download/cache the zoo model")
    parser.add_argument("--speaker-wav", type=Path, default=Path(__file__).parents[1] / "tests/inputs/example_1.wav")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--runs", type=int, default=3, help="timed runs per text, after one warm-up")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--texts", nargs="*", default=list(TEXTS), choices=list(TEXTS))
    parser.add_argument("--batch-size", type=int, default=1, help=">1 uses Xtts.inference_batch")
    parser.add_argument("--load", nargs="*", default=[], metavar="K=V", help="extra Xtts.load_checkpoint kwargs")
    parser.add_argument("--infer", nargs="*", default=[], metavar="K=V", help="extra inference kwargs")
    parser.add_argument("--no-stream", action="store_true", help="skip the streaming latency test")
    parser.add_argument("--asr", help="HF ASR model id for word error rate, e.g. openai/whisper-large-v3-turbo")
    parser.add_argument("--wav-dir", type=Path, help="write the last run's audio per text here")
    parser.add_argument("--out", type=Path, help="write results as JSON")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    load_kwargs, infer = parse_kv(args.load), parse_kv(args.infer)
    cuda = args.device.startswith("cuda")
    results: dict[str, Any] = {
        "device": torch.cuda.get_device_name() if cuda else args.device,
        "torch": torch.__version__,
        "load": load_kwargs,
        "infer": infer,
        "batch_size": args.batch_size,
        "texts": {},
    }

    t0 = time.perf_counter()
    model = load_model(args.model_dir, args.device, load_kwargs)
    sync(args.device)
    results["load_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    latents = model.get_conditioning_latents(audio_path=str(args.speaker_wav))
    sync(args.device)
    results["conditioning_s"] = time.perf_counter() - t0

    audio: dict[str, Any] = {}
    for name in args.texts:
        language, text = TEXTS[name]
        sentences = Segmenter(language).segment(text)
        synth(model, sentences, language, latents, args.batch_size, args.seed, infer)  # warm-up
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        times, durations = [], []
        for run in range(args.runs):
            sync(args.device)
            t0 = time.perf_counter()
            wav = synth(model, sentences, language, latents, args.batch_size, args.seed + run, infer)
            sync(args.device)
            times.append(time.perf_counter() - t0)
            durations.append(len(wav) / SAMPLE_RATE)
        audio[name] = wav
        results["texts"][name] = {
            "sentences": len(sentences),
            "time_s": statistics.median(times),
            "times_s": times,
            "audio_s_runs": durations,
            "audio_s": statistics.median(durations),
            "rtf": statistics.median(t / d for t, d in zip(times, durations)),
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if cuda else None,
        }

    if not args.no_stream:
        language, text = TEXTS["en-medium"]
        stream_infer = {k: v for k, v in infer.items() if k not in ("batch_size",)}
        ttfb = []
        for run in range(args.runs + 1):
            torch.manual_seed(args.seed + run)
            sync(args.device)
            t0 = time.perf_counter()
            for _chunk in model.inference_stream(text, language, *latents, **stream_infer):
                sync(args.device)
                ttfb.append(time.perf_counter() - t0)
                break
        results["stream_ttfb_s"] = statistics.median(ttfb[1:])  # first is warm-up

    if args.wav_dir:
        import soundfile as sf

        args.wav_dir.mkdir(parents=True, exist_ok=True)
        for name, wav in audio.items():
            sf.write(args.wav_dir / f"{name}.wav", wav, SAMPLE_RATE)

    if args.asr:
        import librosa
        from transformers import pipeline

        del model
        if cuda:
            torch.cuda.empty_cache()
        asr = pipeline("automatic-speech-recognition", model=args.asr, device=args.device)
        for name, wav in audio.items():
            language, text = TEXTS[name]
            wav16 = librosa.resample(wav.astype("float32"), orig_sr=SAMPLE_RATE, target_sr=16000)
            hyp = asr(wav16, generate_kwargs={"language": language}, return_timestamps=True)["text"]
            results["texts"][name]["wer"] = wer(text, hyp)
            results["texts"][name]["asr"] = hyp.strip()

    print(
        f"{results['device']} | torch {results['torch']} | load {load_kwargs} | infer {infer} | batch {args.batch_size}"
    )
    print(f"load {results['load_s']:.1f}s  conditioning {results['conditioning_s'] * 1000:.0f}ms", end="")
    print(f"  stream TTFB {results['stream_ttfb_s'] * 1000:.0f}ms" if "stream_ttfb_s" in results else "")
    print(f"{'text':<10} {'sent':>4} {'time':>7} {'audio':>7} {'RTF':>6} {'VRAM':>6} {'WER':>6}")
    for name, r in results["texts"].items():
        vram = f"{r['peak_vram_gb']:.2f}" if r["peak_vram_gb"] is not None else "-"
        w = f"{r['wer']:.3f}" if "wer" in r else "-"
        print(
            f"{name:<10} {r['sentences']:>4} {r['time_s']:>7.2f} {r['audio_s']:>7.2f} {r['rtf']:>6.3f} {vram:>6} {w:>6}"
        )
    if args.out:
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
