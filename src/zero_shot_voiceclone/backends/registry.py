from __future__ import annotations

from dataclasses import dataclass

from .base import VoiceCloneBackend
from .qwen import QwenVoiceCloneBackend
from .xtts import XTTSVoiceCloneBackend
from ..domain import CLIError, GenerationSettings


@dataclass(frozen=True)
class BackendDefinition:
    name: str
    backend_cls: type[VoiceCloneBackend]
    default_model: str
    description: str


_BACKENDS: dict[str, BackendDefinition] = {
    "qwen": BackendDefinition(
        name="qwen",
        backend_cls=QwenVoiceCloneBackend,
        default_model="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        description="Official Qwen3-TTS voice cloning backend.",
    ),
    "xtts": BackendDefinition(
        name="xtts",
        backend_cls=XTTSVoiceCloneBackend,
        default_model="coqui/XTTS-v2",
        description="Coqui XTTS-v2 voice cloning backend.",
    ),
}


def create_backend(settings: GenerationSettings) -> VoiceCloneBackend:
    definition = get_backend_definition(settings.backend_name)
    return definition.backend_cls(settings)


def get_backend_definition(name: str) -> BackendDefinition:
    try:
        return _BACKENDS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_BACKENDS))
        raise CLIError(
            f"Unsupported backend '{name}'. Supported backends: {supported}"
        ) from exc


def get_backend_names() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def get_default_model(name: str) -> str:
    return get_backend_definition(name).default_model
