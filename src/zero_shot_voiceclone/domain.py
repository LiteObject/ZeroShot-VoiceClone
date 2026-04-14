from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class CLIError(RuntimeError):
    """Expected user-facing CLI error."""


@dataclass(slots=True, frozen=True)
class AudioInspection:
    duration_seconds: float | None
    sample_rate: int | None
    channels: int | None


@dataclass(slots=True)
class GenerationSettings:
    backend_name: str
    model_name: str
    language: str
    chunk_max_chars: int
    silence_ms: int
    backend_options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class BackendCapabilities:
    requires_reference_transcript: bool = False
    supports_transcriptless_clone: bool = False
    supports_auto_language: bool = True
