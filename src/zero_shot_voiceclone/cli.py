from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .backends.registry import get_backend_names, get_default_model
from .domain import CLIError, GenerationSettings
from .engine import (
    VoiceCloneEngine,
    build_metadata,
    validate_reference_audio,
    write_audio_file,
    write_metadata_file,
)

DEFAULT_BACKEND = "qwen"
DEFAULT_MODEL = get_default_model(DEFAULT_BACKEND)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voiceclone",
        description="Clone a voice from a reference audio sample using a selectable backend.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    synth = subparsers.add_parser(
        "synth", help="Render a target text file in the cloned voice."
    )

    shared = synth.add_argument_group("Shared synthesis options")
    qwen_group = synth.add_argument_group("Qwen backend options")
    xtts_group = synth.add_argument_group("XTTS-v2 backend options")

    shared.add_argument(
        "--reference-audio",
        required=True,
        type=Path,
        help="Path to the source voice sample.",
    )
    shared.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=get_backend_names(),
        help="Voice cloning backend to use.",
    )
    shared.add_argument(
        "--reference-text-file",
        type=Path,
        help="UTF-8 transcript of the reference audio. Required by some backends and ignored by others.",
    )
    shared.add_argument(
        "--target-text-file",
        required=True,
        type=Path,
        help="UTF-8 text file to synthesize.",
    )
    shared.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output audio path, for example out.wav.",
    )
    shared.add_argument(
        "--model",
        help="Backend-specific model identifier. Defaults to the selected backend's default model.",
    )
    shared.add_argument(
        "--language",
        default="Auto",
        help="Target language. Qwen accepts 'Auto'; XTTS expects an explicit code such as 'en' or 'zh-cn'.",
    )
    shared.add_argument(
        "--chunk-max-chars",
        type=int,
        default=240,
        help="Maximum characters per synthesized text chunk.",
    )
    shared.add_argument(
        "--silence-ms",
        type=int,
        default=250,
        help="Silence inserted between synthesized chunks.",
    )
    shared.add_argument(
        "--metadata-path",
        type=Path,
        help="Optional path for the JSON sidecar. Defaults to <output>.json.",
    )
    shared.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Do not write a JSON metadata sidecar.",
    )
    shared.add_argument(
        "--confirm-rights-to-voice",
        action="store_true",
        help="Required acknowledgement that you have rights to clone the reference voice.",
    )

    qwen_group.add_argument(
        "--qwen-device",
        dest="qwen_device",
        default="auto",
        help="Qwen backend option: model device map, for example auto, cpu, or cuda:0.",
    )
    qwen_group.add_argument(
        "--device",
        dest="qwen_device",
        help=argparse.SUPPRESS,
    )
    qwen_group.add_argument(
        "--qwen-dtype",
        dest="qwen_dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Qwen backend option: torch dtype for model inference.",
    )
    qwen_group.add_argument(
        "--dtype",
        dest="qwen_dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        help=argparse.SUPPRESS,
    )
    qwen_group.add_argument(
        "--qwen-attn-implementation",
        dest="qwen_attn_implementation",
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager", "none"],
        help="Qwen backend option: attention backend passed to the model loader.",
    )
    qwen_group.add_argument(
        "--attn-implementation",
        dest="qwen_attn_implementation",
        choices=["auto", "flash_attention_2", "sdpa", "eager", "none"],
        help=argparse.SUPPRESS,
    )
    qwen_group.add_argument(
        "--qwen-x-vector-only",
        dest="qwen_x_vector_only",
        action="store_true",
        help="Qwen backend option: allow transcript-free cloning with lower similarity.",
    )
    qwen_group.add_argument(
        "--x-vector-only",
        dest="qwen_x_vector_only",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    xtts_group.add_argument(
        "--xtts-device",
        default="auto",
        help="XTTS-v2 backend option: compute target such as auto, cpu, or cuda.",
    )
    xtts_group.add_argument(
        "--xtts-split-sentences",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="XTTS-v2 backend option: use Coqui's internal sentence splitting in addition to the CLI chunking.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "synth":
            return run_synth(args)
        raise CLIError(f"Unsupported command: {args.command}")
    except CLIError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


def run_synth(args: argparse.Namespace) -> int:
    if not args.confirm_rights_to_voice:
        raise CLIError("Refusing to run without --confirm-rights-to-voice.")

    reference_audio = args.reference_audio
    reference_text_file = args.reference_text_file
    target_text_file = args.target_text_file
    output_path = args.output

    reference_transcript = (
        read_utf8_text(reference_text_file) if reference_text_file else None
    )
    target_text = read_utf8_text(target_text_file)
    if not target_text.strip():
        raise CLIError("Target text file is empty.")

    audio_inspection = validate_reference_audio(reference_audio)
    settings = GenerationSettings(
        backend_name=args.backend,
        model_name=args.model or get_default_model(args.backend),
        language=args.language,
        chunk_max_chars=args.chunk_max_chars,
        silence_ms=args.silence_ms,
        backend_options=build_backend_options(args),
    )

    engine = VoiceCloneEngine(settings)
    audio, sample_rate, chunks = engine.synthesize_document(
        target_text=target_text,
        reference_audio=reference_audio,
        reference_transcript=reference_transcript,
    )
    write_audio_file(output_path, audio, sample_rate)

    if hasattr(audio, "shape"):
        output_samples = int(audio.shape[0])
    else:
        output_samples = len(audio)
    output_duration_seconds = output_samples / sample_rate

    metadata_path = args.metadata_path or Path(f"{output_path}.json")
    if not args.skip_metadata:
        metadata = build_metadata(
            settings=settings,
            reference_audio=reference_audio,
            reference_transcript_path=reference_text_file,
            target_text_path=target_text_file,
            output_path=output_path,
            audio_inspection=audio_inspection,
            chunk_count=len(chunks),
            output_duration_seconds=output_duration_seconds,
        )
        write_metadata_file(metadata_path, metadata)

    print(f"Wrote cloned audio to {output_path}")
    if not args.skip_metadata:
        print(f"Wrote metadata to {metadata_path}")
    return 0


def read_utf8_text(path: Path | None) -> str:
    if path is None:
        raise CLIError("A required text file path was not provided.")
    if not path.exists():
        raise CLIError(f"Text file does not exist: {path}")
    if not path.is_file():
        raise CLIError(f"Text file path is not a file: {path}")

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise CLIError(f"Text file must be UTF-8 encoded: {path}") from exc


def build_backend_options(args: argparse.Namespace) -> dict[str, object]:
    if args.backend == "qwen":
        return {
            "device": args.qwen_device,
            "dtype": args.qwen_dtype,
            "attn_implementation": args.qwen_attn_implementation,
            "x_vector_only_mode": args.qwen_x_vector_only,
        }

    if args.backend == "xtts":
        options: dict[str, object] = {
            "device": args.xtts_device,
        }
        if args.xtts_split_sentences is not None:
            options["split_sentences"] = args.xtts_split_sentences
        return options

    return {}
