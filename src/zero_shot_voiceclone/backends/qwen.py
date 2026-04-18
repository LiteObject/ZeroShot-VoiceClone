from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import VoiceCloneBackend
from ..domain import BackendCapabilities, CLIError


class QwenVoiceCloneBackend(VoiceCloneBackend):
    backend_name = "qwen"
    capabilities = BackendCapabilities(
        requires_reference_transcript=False,
        supports_transcriptless_clone=True,
        supports_auto_language=True,
    )

    def __init__(self, settings) -> None:
        super().__init__(settings)
        self._model: Any | None = None

    def validate_inputs(self, reference_transcript: str | None) -> None:
        super().validate_inputs(reference_transcript)
        if reference_transcript is None and not self._x_vector_only_mode:
            raise CLIError(
                "The qwen backend requires --reference-text-file unless --x-vector-only is enabled."
            )

    def prepare_reference(
        self, reference_audio: Path, reference_transcript: str | None
    ) -> Any:
        model = self._load_model()
        prompt_kwargs: dict[str, Any] = {"ref_audio": str(reference_audio)}
        if reference_transcript is not None:
            prompt_kwargs["ref_text"] = reference_transcript
        if self._x_vector_only_mode:
            prompt_kwargs["x_vector_only_mode"] = True
        return model.create_voice_clone_prompt(**prompt_kwargs)

    def synthesize_chunk(self, text: str, prepared_reference: Any) -> tuple[Any, int]:
        model = self._load_model()
        wavs, sample_rate = model.generate_voice_clone(
            text=text,
            language=self.settings.language,
            voice_clone_prompt=prepared_reference,
        )
        if not wavs:
            raise CLIError("Model returned no audio for the requested chunk.")
        return wavs[0], sample_rate

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            import torch
        except ImportError as exc:
            raise CLIError(
                "PyTorch is not installed in the current environment."
            ) from exc

        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise CLIError(
                "qwen-tts is not installed in the current environment."
            ) from exc

        dtype = self._resolve_dtype(torch)
        model_kwargs: dict[str, Any] = {
            "device_map": self._device,
            "dtype": dtype,
        }

        attn_implementation = self._resolve_attn_implementation(torch, dtype)
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = Qwen3TTSModel.from_pretrained(
            self.settings.model_name, **model_kwargs
        )
        return self._model

    def _resolve_dtype(self, torch: Any) -> Any:
        if self._dtype == "float32":
            return torch.float32
        if self._dtype == "float16":
            return torch.float16
        if self._dtype == "bfloat16":
            return torch.bfloat16

        if self._device == "cpu":
            return torch.float32

        if torch.cuda.is_available():
            return torch.bfloat16

        return torch.float32

    def _resolve_attn_implementation(self, torch: Any, dtype: Any) -> str | None:
        requested = self._attn_implementation
        if requested == "none":
            return None
        if requested != "auto":
            return requested

        uses_cuda = self._device.startswith("cuda") or (
            self._device == "auto" and torch.cuda.is_available()
        )
        if uses_cuda and dtype in {torch.float16, torch.bfloat16}:
            try:
                import importlib.util

                if importlib.util.find_spec("flash_attn") is not None:
                    return "flash_attention_2"
            except (ImportError, ModuleNotFoundError):
                pass
        return None

    @property
    def _device(self) -> str:
        return str(self.settings.backend_options.get("device", "auto"))

    @property
    def _dtype(self) -> str:
        return str(self.settings.backend_options.get("dtype", "auto"))

    @property
    def _attn_implementation(self) -> str:
        return str(self.settings.backend_options.get("attn_implementation", "auto"))

    @property
    def _x_vector_only_mode(self) -> bool:
        return bool(self.settings.backend_options.get("x_vector_only_mode", False))
