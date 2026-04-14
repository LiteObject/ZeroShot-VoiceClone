from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..domain import BackendCapabilities, CLIError, GenerationSettings


class VoiceCloneBackend(ABC):
    backend_name = "base"
    capabilities = BackendCapabilities()

    def __init__(self, settings: GenerationSettings) -> None:
        self.settings = settings

    def validate_inputs(self, _reference_transcript: str | None) -> None:
        if (
            self.settings.language == "Auto"
            and not self.capabilities.supports_auto_language
        ):
            raise CLIError(
                f"The {self.backend_name} backend does not support automatic language selection."
            )

    @abstractmethod
    def prepare_reference(
        self, reference_audio: Path, reference_transcript: str | None
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def synthesize_chunk(self, text: str, prepared_reference: Any) -> tuple[Any, int]:
        raise NotImplementedError
