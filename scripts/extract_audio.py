#!/usr/bin/env python3
"""Extract a clean audio reference clip from a video (or any media) file.

Decodes an audio stream from an MP4 or another container into a mono PCM WAV
suitable for zero-shot voice cloning, with optional trimming and loudness
normalization, followed by a reference-quality report.

FFmpeg is located on PATH, through the FFMPEG_BINARY environment variable, or
through the optional imageio-ffmpeg package. The quality report needs numpy and
soundfile, which are regular project dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FFMPEG_ENV_VAR = "FFMPEG_BINARY"
FFPROBE_ENV_VAR = "FFPROBE_BINARY"
LOUDNESS_FILTER = "loudnorm=I=-16:TP=-1.5:LRA=11"
_AUDIO_INFO_RE = re.compile(r"Stream #\d+:\d+[^:]*: Audio: [^,]+, (\d+) Hz")
SILENCE_FLOOR_DBFS = -50.0
SILENCE_FRAME_SECONDS = 0.025

# Mirrors the reference-audio thresholds used by zero_shot_voiceclone.engine.
MIN_DURATION_SECONDS = 1.0
RECOMMENDED_MIN_SECONDS = 3.0
RECOMMENDED_MAX_SECONDS = 10.0
LONG_REFERENCE_SECONDS = 30.0


class ExtractError(RuntimeError):
    """Expected user-facing error."""


@dataclass(slots=True)
class AudioReport:
    duration_seconds: float
    sample_rate: int
    channels: int
    subtype: str
    peak_dbfs: float | None
    rms_dbfs: float | None
    clipped_samples: int
    dc_offset: float
    silence_ratio: float


def parse_time(value: str) -> float:
    """Parse seconds or [[HH:]MM:]SS into seconds."""
    text = value.strip()
    if not text:
        raise ValueError(f"Invalid time value: {value!r}")

    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"Invalid time value: {value!r}")

    total = 0.0
    for part in parts:
        total = total * 60.0 + float(part)

    if not math.isfinite(total) or total < 0.0:
        raise ValueError(f"Invalid time value: {value!r}")
    return total


def _format_seconds(seconds: float) -> str:
    text = f"{seconds:.3f}".rstrip("0").rstrip(".")
    return text or "0"


def _optional_time(value: str | None, option: str) -> float | None:
    if value is None:
        return None
    try:
        return parse_time(value)
    except ValueError as exc:
        raise ExtractError(f"Invalid value for {option}: {value!r}") from exc


def _first_available(candidates: tuple[str | None, ...]) -> str | None:
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _resolve_explicit(candidate: str, option: str, label: str) -> str:
    resolved = shutil.which(candidate)
    if not resolved:
        raise ExtractError(
            f"{label} was not found at the path given to {option}: {candidate}"
        )
    return resolved


def resolve_ffmpeg(explicit: str | None = None) -> str | None:
    if explicit:
        return _resolve_explicit(explicit, "--ffmpeg", "FFmpeg")

    from_env = os.environ.get(FFMPEG_ENV_VAR)
    resolved = _first_available((from_env, "ffmpeg"))
    if resolved:
        return resolved
    if from_env:
        raise ExtractError(f"FFmpeg from {FFMPEG_ENV_VAR} was not found: {from_env}")

    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def resolve_ffprobe(
    explicit: str | None = None, ffmpeg_path: str | None = None
) -> str | None:
    if explicit:
        return _resolve_explicit(explicit, "--ffprobe", "ffprobe")

    from_env = os.environ.get(FFPROBE_ENV_VAR)
    resolved = _first_available((from_env, "ffprobe"))
    if resolved:
        return resolved
    if from_env:
        raise ExtractError(f"ffprobe from {FFPROBE_ENV_VAR} was not found: {from_env}")

    if ffmpeg_path:
        suffix = ".exe" if os.name == "nt" else ""
        sibling = Path(ffmpeg_path).with_name(f"ffprobe{suffix}")
        if sibling.exists():
            return str(sibling)
    return None


def build_extract_command(
    ffmpeg: str,
    input_path: Path,
    output_path: Path,
    *,
    stream_index: int = 0,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    channels: int = 1,
    sample_rate: int | None = None,
    normalize: bool = True,
) -> list[str]:
    command = [ffmpeg, "-hide_banner", "-nostdin", "-y", "-loglevel", "error"]
    if start_seconds is not None:
        command += ["-ss", _format_seconds(start_seconds)]
    command += ["-i", str(input_path), "-vn", "-map", f"0:a:{stream_index}"]
    if duration_seconds is not None:
        command += ["-t", _format_seconds(duration_seconds)]
    if normalize:
        command += ["-af", LOUDNESS_FILTER]
    command += ["-ac", str(channels)]
    if sample_rate is not None:
        command += ["-ar", str(sample_rate)]
    command += ["-c:a", "pcm_s16le", str(output_path)]
    return command


def build_copy_command(
    ffmpeg: str,
    input_path: Path,
    output_path: Path,
    *,
    stream_index: int = 0,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
) -> list[str]:
    command = [ffmpeg, "-hide_banner", "-nostdin", "-y", "-loglevel", "error"]
    if start_seconds is not None:
        command += ["-ss", _format_seconds(start_seconds)]
    command += ["-i", str(input_path), "-vn", "-map", f"0:a:{stream_index}"]
    if duration_seconds is not None:
        command += ["-t", _format_seconds(duration_seconds)]
    command += ["-c:a", "copy", str(output_path)]
    return command


def run_command(command: list[str]) -> None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise ExtractError(f"Failed to run {command[0]}: {exc}") from exc

    if result.returncode == 0:
        return

    detail = (result.stderr or result.stdout or "").strip()
    if "matches no streams" in detail:
        raise ExtractError(
            "The input file has no audio stream matching the requested --stream-index."
        )

    message = f"FFmpeg failed with exit code {result.returncode}."
    message += f"\nCommand: {shlex.join(command)}"
    if detail:
        message += f"\n{detail}"
    raise ExtractError(message)


def _import_audio_stack() -> tuple[Any, Any] | None:
    try:
        import numpy
        import soundfile
    except ImportError:
        return None
    return numpy, soundfile


def analyze_audio_file(path: Path) -> AudioReport | None:
    """Return quality metrics for a WAV file, or None if numpy/soundfile are missing."""
    stack = _import_audio_stack()
    if stack is None:
        return None
    np, sf = stack

    info = sf.info(str(path))
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    frames, channels = data.shape
    duration_seconds = frames / sample_rate if sample_rate else 0.0

    if frames == 0:
        return AudioReport(
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            channels=channels,
            subtype=info.subtype or "",
            peak_dbfs=None,
            rms_dbfs=None,
            clipped_samples=0,
            dc_offset=0.0,
            silence_ratio=1.0,
        )

    mono = data.mean(axis=1)
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))

    frame_size = max(int(sample_rate * SILENCE_FRAME_SECONDS), 1)
    frame_count = frames // frame_size
    if frame_count == 0:
        silence_ratio = 0.0
    else:
        windowed = mono[: frame_count * frame_size].reshape(frame_count, frame_size)
        frame_rms = np.sqrt(np.mean(np.square(windowed, dtype=np.float64), axis=1))
        threshold = 10.0 ** (SILENCE_FLOOR_DBFS / 20.0)
        silence_ratio = float(np.count_nonzero(frame_rms < threshold) / frame_count)

    return AudioReport(
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        channels=channels,
        subtype=info.subtype or "",
        peak_dbfs=_dbfs(peak),
        rms_dbfs=_dbfs(rms),
        clipped_samples=int(np.count_nonzero(np.abs(data) >= 0.9995)),
        dc_offset=float(mono.mean()),
        silence_ratio=silence_ratio,
    )


def _dbfs(amplitude: float) -> float | None:
    if amplitude <= 0.0:
        return None
    return 20.0 * math.log10(amplitude)


def reference_warnings(report: AudioReport, *, normalize: bool) -> list[str]:
    notes: list[str] = []
    duration = report.duration_seconds

    if duration < MIN_DURATION_SECONDS:
        notes.append(
            f"Clip is only {duration:.2f} s long; most backends need at least 1 s of speech."
        )
    elif duration > LONG_REFERENCE_SECONDS:
        notes.append(
            f"Clip is {duration:.1f} s long; a shorter 3-10 s excerpt is usually better for cloning quality."
        )
    elif duration < RECOMMENDED_MIN_SECONDS or duration > RECOMMENDED_MAX_SECONDS:
        notes.append(
            f"Clip is outside the recommended {RECOMMENDED_MIN_SECONDS:.0f}-{RECOMMENDED_MAX_SECONDS:.0f} s reference range."
        )

    if report.clipped_samples:
        notes.append(
            f"{report.clipped_samples} clipped sample(s) detected; the source may be distorted."
        )
    if report.peak_dbfs is None:
        notes.append("Track appears to be digital silence.")
    elif report.peak_dbfs < -30.0 and not normalize:
        notes.append("Recording is very quiet; consider leaving normalization enabled.")
    if report.silence_ratio > 0.5:
        notes.append(
            f"{report.silence_ratio:.0%} of the clip is near silence; trim to the speech segment."
        )
    if abs(report.dc_offset) > 0.01:
        notes.append(f"DC offset of {report.dc_offset:+.4f} detected.")
    return notes


def format_report(report: AudioReport, *, normalize: bool) -> str:
    normalization = (
        "loudnorm I=-16 LUFS, TP=-1.5 dB" if normalize else "none (source loudness kept)"
    )
    lines = [
        f"  Encoding      : {report.sample_rate} Hz, {report.channels} channel(s), {report.subtype}",
        f"  Duration      : {report.duration_seconds:.2f} s",
    ]
    if report.peak_dbfs is not None and report.rms_dbfs is not None:
        lines.append(
            f"  Peak / RMS    : {report.peak_dbfs:.1f} dBFS / {report.rms_dbfs:.1f} dBFS"
        )
    lines += [
        f"  Clipped       : {report.clipped_samples} sample(s)",
        f"  DC offset     : {report.dc_offset:+.4f}",
        f"  Silence       : {report.silence_ratio:.0%}",
        f"  Normalization : {normalization}",
    ]

    notes = reference_warnings(report, normalize=normalize)
    if notes:
        lines.append("  Warnings:")
        lines.extend(f"    - {note}" for note in notes)
    return "\n".join(lines)


def _as_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: object) -> float | None:
    try:
        result = float(str(value))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def probe_streams(ffprobe: str, input_path: Path) -> list[dict[str, Any]] | None:
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_streams",
        "-of",
        "json",
        str(input_path),
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None

    streams = payload.get("streams")
    return streams if isinstance(streams, list) else None


def read_stream_banner(ffmpeg: str, input_path: Path) -> str | None:
    """Return FFmpeg's stream banner for an input, used when ffprobe is unavailable.

    FFmpeg exits with status 1 because no output was requested, but the banner
    with codec and sample-rate details is still printed to stderr.
    """
    command = [ffmpeg, "-hide_banner", "-nostdin", "-i", str(input_path)]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return None
    return result.stderr or None


def _sample_rates_from_banner(banner: str) -> list[int]:
    return [int(match) for match in _AUDIO_INFO_RE.findall(banner)]


def describe_audio_stream(stream: dict[str, Any]) -> str:
    parts = [str(stream.get("codec_name") or "unknown")]
    sample_rate = _as_int(stream.get("sample_rate"))
    if sample_rate:
        parts.append(f"{sample_rate} Hz")
    channels = _as_int(stream.get("channels"))
    if channels:
        parts.append(f"{channels} ch")
    bit_rate = _as_int(stream.get("bit_rate"))
    if bit_rate:
        parts.append(f"{bit_rate // 1000} kb/s")
    return ", ".join(parts)


def print_streams(streams: list[dict[str, Any]]) -> None:
    audio_index = 0
    video_index = 0
    for index, stream in enumerate(streams):
        kind = str(stream.get("codec_type") or "unknown")
        codec = str(stream.get("codec_name") or "unknown")
        details: list[str] = []
        if kind == "audio":
            label = f"a:{audio_index}"
            audio_index += 1
            sample_rate = _as_int(stream.get("sample_rate"))
            channels = _as_int(stream.get("channels"))
            if sample_rate:
                details.append(f"{sample_rate} Hz")
            if channels:
                details.append(f"{channels} ch")
        elif kind == "video":
            label = f"v:{video_index}"
            video_index += 1
            width = _as_int(stream.get("width"))
            height = _as_int(stream.get("height"))
            if width and height:
                details.append(f"{width}x{height}")
        else:
            label = ""
        suffix = f" ({', '.join(details)})" if details else ""
        print(f"#{index:<3} {label:>4}  {kind:5s} {codec}{suffix}")


def _effective_clip_duration(
    source_stream: dict[str, Any] | None,
    start_seconds: float | None,
    duration_seconds: float | None,
) -> float | None:
    if duration_seconds is not None:
        return duration_seconds
    if source_stream is None:
        return None
    source_duration = _as_float(source_stream.get("duration"))
    if source_duration is None:
        return None
    return max(source_duration - (start_seconds or 0.0), 0.0)


def _ffmpeg_missing_message() -> str:
    return (
        "FFmpeg was not found. Install it and make sure it is on PATH, or install the "
        "optional Python fallback with 'pip install imageio-ffmpeg'.\n"
        "  Windows: winget install Gyan.FFmpeg  (or scoop install ffmpeg / choco install ffmpeg)\n"
        "  macOS:   brew install ffmpeg\n"
        "  Linux:   sudo apt install ffmpeg"
    )


def _validate_output_extension(path: Path, *, copy: bool) -> None:
    suffix = path.suffix.lower()
    if copy:
        if suffix in ("", ".wav"):
            raise ExtractError(
                "--copy writes the compressed stream as-is; use an output such as .m4a, .mka, or .aac."
            )
        return
    if suffix != ".wav":
        raise ExtractError(
            f"Only WAV output is supported without --copy: {path}. Use a .wav output path."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract_audio.py",
        description="Extract a clean mono WAV reference clip from a video or audio file for voice cloning.",
        epilog="Example: python scripts/extract_audio.py interview.mp4 --start 12 --duration 8",
    )
    parser.add_argument("input", type=Path, help="Input media file, for example an MP4.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path. Defaults to <input stem>.extracted.wav for WAV output.",
    )
    parser.add_argument(
        "--start",
        help="Start offset in the source, as seconds or [[HH:]MM:]SS. Defaults to the start.",
    )
    parser.add_argument(
        "--duration",
        help="Number of seconds to extract. Defaults to the rest of the file.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        help="Output sample rate in Hz. Defaults to the source rate (no resampling).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        choices=(1, 2),
        help="Output channel count. Defaults to 1 (mono).",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply loudness normalization to -16 LUFS. Enabled by default; use --no-normalize to keep the source loudness.",
    )
    parser.add_argument(
        "--stream-index",
        type=int,
        default=0,
        help="Audio stream to extract when the file has several (0 = first). See --list-streams.",
    )
    parser.add_argument(
        "--list-streams",
        action="store_true",
        help="Print the input's streams and exit.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy the compressed audio stream without re-encoding instead of producing a WAV.",
    )
    parser.add_argument(
        "--ffmpeg",
        help=f"Path to the FFmpeg executable. Overrides the {FFMPEG_ENV_VAR} environment variable.",
    )
    parser.add_argument(
        "--ffprobe",
        help=f"Path to the ffprobe executable. Overrides the {FFPROBE_ENV_VAR} environment variable.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    input_path: Path = args.input
    if not input_path.exists():
        raise ExtractError(f"Input file does not exist: {input_path}")
    if not input_path.is_file():
        raise ExtractError(f"Input path is not a file: {input_path}")

    if args.stream_index < 0:
        raise ExtractError("--stream-index must be zero or greater.")

    ffmpeg = resolve_ffmpeg(args.ffmpeg)
    if not ffmpeg:
        raise ExtractError(_ffmpeg_missing_message())
    ffprobe = resolve_ffprobe(args.ffprobe, ffmpeg)

    if args.list_streams:
        if not ffprobe:
            raise ExtractError(
                "--list-streams requires ffprobe. Install it alongside FFmpeg or pass --ffprobe."
            )
        streams = probe_streams(ffprobe, input_path)
        if streams is None:
            raise ExtractError(f"ffprobe could not read the input file: {input_path}")
        print_streams(streams)
        return 0

    if args.copy and args.output is None:
        raise ExtractError(
            "--copy requires an explicit --output path, for example -o audio.m4a."
        )

    output_path: Path = args.output or input_path.with_name(
        f"{input_path.stem}.extracted.wav"
    )
    _validate_output_extension(output_path, copy=args.copy)
    if output_path.exists() and not args.force:
        raise ExtractError(
            f"Output file already exists: {output_path}. Use --force to overwrite."
        )

    if args.copy and (
        args.sample_rate is not None
        or args.channels is not None
        or args.normalize is not None
    ):
        raise ExtractError(
            "--copy cannot be combined with --sample-rate, --channels, or --normalize."
        )

    if args.sample_rate is not None and args.sample_rate <= 0:
        raise ExtractError("--sample-rate must be greater than zero.")

    start_seconds = _optional_time(args.start, "--start")
    duration_seconds = _optional_time(args.duration, "--duration")
    if duration_seconds is not None and duration_seconds <= 0.0:
        raise ExtractError("--duration must be greater than zero.")

    audio_streams: list[dict[str, Any]] | None = None
    banner_rates: list[int] | None = None
    if ffprobe:
        streams = probe_streams(ffprobe, input_path)
        if streams is not None:
            audio_streams = [
                stream for stream in streams if stream.get("codec_type") == "audio"
            ]
            if not audio_streams:
                raise ExtractError(f"The input file has no audio streams: {input_path}")
    else:
        banner = read_stream_banner(ffmpeg, input_path)
        if banner and "Audio:" not in banner and "Video:" in banner:
            raise ExtractError(f"The input file has no audio streams: {input_path}")
        if banner:
            banner_rates = _sample_rates_from_banner(banner) or None

    if audio_streams is not None:
        audio_count: int | None = len(audio_streams)
    elif banner_rates is not None:
        audio_count = len(banner_rates)
    else:
        audio_count = None

    if audio_count is not None and args.stream_index >= audio_count:
        hint = " Use --list-streams to inspect them." if ffprobe else ""
        raise ExtractError(
            f"--stream-index {args.stream_index} is out of range; the file has "
            f"{audio_count} audio stream(s).{hint}"
        )
    if audio_count is not None and audio_count > 1:
        print(
            f"Note: the input has {audio_count} audio streams; extracting "
            f"a:{args.stream_index}. Use --stream-index to pick another."
        )

    source_stream = audio_streams[args.stream_index] if audio_streams else None
    source_sample_rate: int | None = None
    if source_stream is not None:
        source_sample_rate = _as_int(source_stream.get("sample_rate"))
        print(f"Source audio : {describe_audio_stream(source_stream)}")
    elif banner_rates and args.stream_index < len(banner_rates):
        source_sample_rate = banner_rates[args.stream_index]
        print(
            f"Source audio : {source_sample_rate} Hz "
            "(read from the FFmpeg stream banner; ffprobe is not available)"
        )

    normalize = True if args.normalize is None else args.normalize
    channels = args.channels if args.channels is not None else 1

    # loudnorm writes 192 kHz output unless the encoder rate is pinned, so always
    # pin a rate when normalizing.
    output_sample_rate = args.sample_rate
    if normalize and not args.copy and output_sample_rate is None:
        output_sample_rate = source_sample_rate
        if output_sample_rate is None:
            print(
                "Note: could not detect the source sample rate; loudnorm writes 192 kHz "
                "output. Use --sample-rate to pick a rate."
            )

    clip_duration = _effective_clip_duration(
        source_stream, start_seconds, duration_seconds
    )
    if (
        normalize
        and not args.copy
        and clip_duration is not None
        and clip_duration < RECOMMENDED_MIN_SECONDS
    ):
        print(
            "Note: loudness normalization is less accurate on clips shorter than 3 s; "
            "consider --no-normalize."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.copy:
        command = build_copy_command(
            ffmpeg,
            input_path,
            output_path,
            stream_index=args.stream_index,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )
        print(f"Copying audio stream without re-encoding: {input_path} -> {output_path}")
    else:
        command = build_extract_command(
            ffmpeg,
            input_path,
            output_path,
            stream_index=args.stream_index,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            channels=channels,
            sample_rate=output_sample_rate,
            normalize=normalize,
        )
        print(f"Extracting audio: {input_path} -> {output_path}")
    run_command(command)

    if not output_path.exists():
        raise ExtractError("FFmpeg finished but no output file was written.")

    print(f"Wrote reference clip: {output_path}")
    if args.copy:
        return 0

    try:
        report = analyze_audio_file(output_path)
    except RuntimeError as exc:
        raise ExtractError(f"Could not read the extracted file for analysis: {exc}") from exc

    if report is None:
        print("Quality report skipped: install numpy and soundfile to enable it.")
    else:
        print(format_report(report, normalize=normalize))
    print(
        'Next: pass this file as --reference-audio to "voiceclone synth" '
        "(with --reference-text-file for Qwen)."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except ExtractError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
