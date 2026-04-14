from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import sys
import warnings

from .backends.registry import create_backend
from .domain import AudioInspection, CLIError, GenerationSettings
from .text import split_text_into_chunks


class VoiceCloneEngine:
    def __init__(self, settings: GenerationSettings) -> None:
        self.settings = settings
        self.backend = create_backend(settings)

    def synthesize_document(
        self,
        target_text: str,
        reference_audio: Path,
        reference_transcript: str | None,
    ) -> tuple[Any, int, list[str]]:
        self.backend.validate_inputs(reference_transcript)
        chunks = split_text_into_chunks(
            target_text, max_chars=self.settings.chunk_max_chars
        )
        if not chunks:
            raise CLIError("Target text file is empty after normalization.")

        prepared_reference = self.backend.prepare_reference(
            reference_audio=reference_audio,
            reference_transcript=reference_transcript,
        )

        generated_segments: list[Any] = []
        sample_rate: int | None = None

        for index, chunk in enumerate(chunks, start=1):
            print(
                f"Synthesizing chunk {index}/{len(chunks)} with backend '{self.settings.backend_name}'...",
                file=sys.stderr,
            )
            waveform, sample_rate = self.backend.synthesize_chunk(
                text=chunk,
                prepared_reference=prepared_reference,
            )
            generated_segments.append(waveform)

        if sample_rate is None:
            raise CLIError("Model did not return a sample rate.")

        combined_audio = concatenate_audio_segments(
            generated_segments, sample_rate, self.settings.silence_ms
        )
        return combined_audio, sample_rate, chunks


def inspect_audio_file(path: Path) -> AudioInspection:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise CLIError(
            "soundfile is not installed in the current environment."
        ) from exc

    try:
        info = sf.info(str(path))
    except RuntimeError:
        try:
            import audioread
        except ImportError:
            try:
                import librosa
            except ImportError as exc:
                raise CLIError(f"Unable to inspect audio file: {path}") from exc

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                audio, sample_rate = librosa.load(str(path), sr=None, mono=False)
            if getattr(audio, "ndim", 1) == 1:
                channels = 1
            else:
                channels = int(audio.shape[0])
            duration_seconds = float(librosa.get_duration(y=audio, sr=sample_rate))
            return AudioInspection(
                duration_seconds=duration_seconds,
                sample_rate=sample_rate,
                channels=channels,
            )

        try:
            with audioread.audio_open(str(path)) as input_file:
                duration_seconds = float(input_file.duration)
                sample_rate = getattr(input_file, "samplerate", None)
                channels = getattr(input_file, "channels", None)
        except OSError as exc:
            raise CLIError(f"Unable to inspect audio file: {path}") from exc

        return AudioInspection(
            duration_seconds=duration_seconds,
            sample_rate=int(sample_rate) if sample_rate else None,
            channels=int(channels) if channels else None,
        )

    duration_seconds = None
    if info.samplerate:
        duration_seconds = info.frames / info.samplerate

    return AudioInspection(
        duration_seconds=duration_seconds,
        sample_rate=info.samplerate or None,
        channels=info.channels or None,
    )


def validate_reference_audio(path: Path) -> AudioInspection:
    if not path.exists():
        raise CLIError(f"Reference audio file does not exist: {path}")
    if not path.is_file():
        raise CLIError(f"Reference audio path is not a file: {path}")

    inspection = inspect_audio_file(path)
    if inspection.duration_seconds is None:
        return inspection

    if inspection.duration_seconds < 1.0:
        raise CLIError("Reference audio is too short. Use at least 1 second of speech.")
    if inspection.duration_seconds > 30.0:
        warnings.warn(
            "Reference audio is longer than 30 seconds. The CLI will continue, but prompt extraction may be slower and a shorter speech excerpt is usually better for cloning quality.",
            stacklevel=2,
        )
    elif inspection.duration_seconds < 3.0 or inspection.duration_seconds > 10.0:
        warnings.warn(
            "Reference audio is outside the recommended 3 to 10 second range; cloning quality may vary.",
            stacklevel=2,
        )

    return inspection


def concatenate_audio_segments(
    segments: list[Any], sample_rate: int, silence_ms: int
) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise CLIError("numpy is not installed in the current environment.") from exc

    normalized_segments: list[Any] = []
    for segment in segments:
        array = np.asarray(segment, dtype=np.float32)
        if array.ndim > 1 and array.shape[0] == 1:
            array = array.squeeze(0)
        normalized_segments.append(array)

    if not normalized_segments:
        raise CLIError("No audio segments were generated.")

    silence_samples = max(int(sample_rate * silence_ms / 1000), 0)
    silence = np.zeros(silence_samples, dtype=np.float32) if silence_samples else None

    parts: list[Any] = []
    for index, segment in enumerate(normalized_segments):
        if index and silence is not None:
            parts.append(silence)
        parts.append(segment)

    return np.concatenate(parts)


def write_audio_file(path: Path, audio: Any, sample_rate: int) -> None:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise CLIError(
            "soundfile is not installed in the current environment."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate)


def build_metadata(
    *,
    settings: GenerationSettings,
    reference_audio: Path,
    reference_transcript_path: Path | None,
    target_text_path: Path,
    output_path: Path,
    audio_inspection: AudioInspection,
    chunk_count: int,
    output_duration_seconds: float,
) -> dict[str, Any]:
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": settings.backend_name,
        "model_name": settings.model_name,
        "language": settings.language,
        "chunk_max_chars": settings.chunk_max_chars,
        "silence_ms": settings.silence_ms,
        "backend_options": dict(settings.backend_options),
        "reference_audio": str(reference_audio.resolve()),
        "reference_text_file": (
            str(reference_transcript_path.resolve())
            if reference_transcript_path
            else None
        ),
        "target_text_file": str(target_text_path.resolve()),
        "output_audio": str(output_path.resolve()),
        "reference_audio_duration_seconds": audio_inspection.duration_seconds,
        "reference_audio_sample_rate": audio_inspection.sample_rate,
        "reference_audio_channels": audio_inspection.channels,
        "output_duration_seconds": output_duration_seconds,
        "chunk_count": chunk_count,
    }
    metadata.update(settings.backend_options)
    return metadata


def write_metadata_file(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
