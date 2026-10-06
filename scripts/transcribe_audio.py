#!/usr/bin/env python3
"""Transcribe an audio or video file into a UTF-8 transcript for voice cloning.

Uses faster-whisper to produce the reference transcript that high-quality
zero-shot cloning needs. Any container PyAV can decode works, including MP4,
so the transcript can come straight from the source video.

faster-whisper is an optional dependency: pip install -e .[transcribe]
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "large-v3"
DEFAULT_LANGUAGE = "auto"


class TranscribeError(RuntimeError):
    """Expected user-facing error."""


@dataclass(slots=True, frozen=True)
class Segment:
    start: float
    end: float
    text: str


def _import_whisper_model() -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscribeError(
            "faster-whisper is not installed. Install the optional dependency with "
            "'pip install -e .[transcribe]'."
        ) from exc
    return WhisperModel


def _device_attempts(device: str) -> list[str]:
    return ["cuda", "cpu"] if device == "auto" else [device]


@dataclass(slots=True)
class TranscriptionOutcome:
    segments: list[Segment]
    language: str | None
    language_probability: float | None
    device_label: str
    fallback_notes: list[str]


_DLL_DIRECTORY_HANDLES: list[Any] = []


def _expose_torch_cuda_libraries() -> str | None:
    """Expose PyTorch's bundled CUDA DLLs to CTranslate2 on Windows.

    CTranslate2 loads cuBLAS and cuDNN at run time but does not look inside
    PyTorch's bundled library directory, so a CUDA-capable machine can still
    fail with errors such as "cublas64_12.dll is not found or cannot be loaded".
    """
    if os.name != "nt":
        return None
    try:
        import torch
    except ImportError:
        return None

    lib_dir = Path(torch.__file__).parent / "lib"
    if not lib_dir.is_dir():
        return None
    try:
        handle = os.add_dll_directory(str(lib_dir))
    except OSError:
        return None
    _DLL_DIRECTORY_HANDLES.append(handle)
    return str(lib_dir)


def transcribe_file(
    input_path: Path,
    *,
    model_name: str,
    device: str,
    compute_type: str | None,
    language: str | None,
    vad: bool,
    model_class: Any = None,
) -> TranscriptionOutcome:
    """Load the whisper model and transcribe, trying each candidate device in order."""
    whisper_model_class = model_class or _import_whisper_model()
    attempts = _device_attempts(device)
    if any(candidate.startswith("cuda") for candidate in attempts):
        _expose_torch_cuda_libraries()

    fallback_notes: list[str] = []
    for candidate in attempts:
        effective_type = compute_type or ("float16" if candidate == "cuda" else "int8")
        try:
            model = whisper_model_class(
                model_name, device=candidate, compute_type=effective_type
            )
            segments, info = transcribe_media(model, input_path, language, vad)
        except Exception as exc:
            fallback_notes.append(f"{candidate}: {exc}")
            continue
        return TranscriptionOutcome(
            segments=segments,
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
            device_label=f"{candidate}, {effective_type}",
            fallback_notes=fallback_notes,
        )

    detail = "\n".join(f"  - {note}" for note in fallback_notes)
    raise TranscribeError(f"Could not transcribe {input_path}:\n{detail}")


def transcribe_media(
    model: Any, input_path: Path, language: str | None, vad: bool
) -> tuple[list[Segment], Any]:
    segments_iterator, info = model.transcribe(
        str(input_path), language=language, vad_filter=vad, beam_size=5
    )
    segments = [
        Segment(
            start=float(segment.start),
            end=float(segment.end),
            text=segment.text.strip(),
        )
        for segment in segments_iterator
    ]
    return segments, info


def segments_to_text(segments: list[Segment]) -> str:
    """Join segment texts with single spaces, normalizing whitespace."""
    return " ".join(" ".join(segment.text.split()) for segment in segments).strip()


def format_timestamp(seconds: float) -> str:
    total = max(int(seconds), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_segments(segments: list[Segment]) -> str:
    return "\n".join(
        f"  [{format_timestamp(segment.start)} - {format_timestamp(segment.end)}] {segment.text}"
        for segment in segments
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe_audio.py",
        description="Transcribe a reference audio or video file into a UTF-8 transcript for voice cloning.",
        epilog="Example: python scripts/transcribe_audio.py reference.wav --language en",
    )
    parser.add_argument(
        "input", type=Path, help="Input audio or video file, for example a WAV or MP4."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output text file. Defaults to <input stem>.txt.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "faster-whisper model name or local model path, for example tiny.en, base, "
            f"small, medium, large-v3, or large-v3-turbo. Default: {DEFAULT_MODEL}."
        ),
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="Spoken language code such as en or zh. Default: auto-detect.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device such as auto, cpu, cuda, or cuda:0. Default: auto (CUDA first, then CPU).",
    )
    parser.add_argument(
        "--compute-type",
        help="CTranslate2 compute type such as float16, int8_float16, or int8. Default: float16 on CUDA, int8 on CPU.",
    )
    parser.add_argument(
        "--vad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use voice activity detection to skip silence and avoid hallucinated text. Enabled by default; use --no-vad to disable.",
    )
    parser.add_argument(
        "--show-segments",
        action="store_true",
        help="Print sentence timestamps, useful for choosing a --start/--duration cut for extract_audio.py.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    return parser


def run(args: argparse.Namespace, *, model_class: Any = None) -> int:
    input_path: Path = args.input
    if not input_path.exists():
        raise TranscribeError(f"Input file does not exist: {input_path}")
    if not input_path.is_file():
        raise TranscribeError(f"Input path is not a file: {input_path}")

    output_path: Path = args.output or input_path.with_name(f"{input_path.stem}.txt")
    if output_path.exists() and not args.force:
        raise TranscribeError(
            f"Output file already exists: {output_path}. Use --force to overwrite."
        )

    language = None if args.language.lower() == "auto" else args.language

    print(
        f"Transcribing {input_path} with whisper model '{args.model}' (device={args.device})...",
        flush=True,
    )
    outcome = transcribe_file(
        input_path,
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=language,
        vad=args.vad,
        model_class=model_class,
    )
    for note in outcome.fallback_notes:
        print(f"Note: {note}", file=sys.stderr)

    text = segments_to_text(outcome.segments)
    if not text:
        raise TranscribeError(
            "No speech was detected. If the file does contain speech, retry with "
            "--no-vad or a larger --model."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding="utf-8")

    print(f"Wrote transcript: {output_path}")
    if outcome.language:
        if isinstance(outcome.language_probability, float):
            print(
                f"Detected language: {outcome.language} ({outcome.language_probability:.0%})"
            )
        else:
            print(f"Detected language: {outcome.language}")
    if args.show_segments:
        print("Segments:")
        print(format_segments(outcome.segments))
    print("Transcript:")
    print(text)
    print(
        "Next: verify the transcript matches the clip word for word, then pass it "
        "as --reference-text-file."
    )
    return 0


def main(argv: list[str] | None = None, *, model_class: Any = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args, model_class=model_class)
    except TranscribeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
