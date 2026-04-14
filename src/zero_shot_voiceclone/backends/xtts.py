from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import warnings

from .base import VoiceCloneBackend
from ..domain import BackendCapabilities, CLIError

XTTS_DEFAULT_SAMPLE_RATE = 24000
XTTS_MODEL_ALIASES = {
    "coqui/XTTS-v2": "tts_models/multilingual/multi-dataset/xtts_v2",
    "tts_models/multilingual/multi-dataset/xtts_v2": "tts_models/multilingual/multi-dataset/xtts_v2",
    "xtts": "xtts",
}
XTTS_LANGUAGE_ALIASES = {
    "ar": "ar",
    "arabic": "ar",
    "cs": "cs",
    "czech": "cs",
    "de": "de",
    "german": "de",
    "en": "en",
    "english": "en",
    "es": "es",
    "spanish": "es",
    "fr": "fr",
    "french": "fr",
    "hi": "hi",
    "hindi": "hi",
    "hu": "hu",
    "hungarian": "hu",
    "it": "it",
    "italian": "it",
    "ja": "ja",
    "japanese": "ja",
    "ko": "ko",
    "korean": "ko",
    "nl": "nl",
    "dutch": "nl",
    "pl": "pl",
    "polish": "pl",
    "pt": "pt",
    "portuguese": "pt",
    "ru": "ru",
    "russian": "ru",
    "tr": "tr",
    "turkish": "tr",
    "zh": "zh-cn",
    "zh-cn": "zh-cn",
    "zh_cn": "zh-cn",
    "chinese": "zh-cn",
    "mandarin": "zh-cn",
}
XTTS_SUPPORTED_LANGUAGES = tuple(sorted(set(XTTS_LANGUAGE_ALIASES.values())))


class XTTSVoiceCloneBackend(VoiceCloneBackend):
    backend_name = "xtts"
    capabilities = BackendCapabilities(
        requires_reference_transcript=False,
        supports_transcriptless_clone=True,
        supports_auto_language=False,
    )

    def __init__(self, settings) -> None:
        super().__init__(settings)
        self._tts: Any | None = None

    def validate_inputs(self, reference_transcript: str | None) -> None:
        super().validate_inputs(reference_transcript)
        if reference_transcript is not None:
            warnings.warn(
                "The xtts backend ignores --reference-text-file and only uses the reference audio sample.",
                stacklevel=2,
            )

    def prepare_reference(
        self, reference_audio: Path, reference_transcript: str | None
    ) -> Any:
        _ = reference_transcript
        return {
            "speaker_wav": [str(reference_audio)],
            "language": self._normalized_language,
        }

    def synthesize_chunk(self, text: str, prepared_reference: Any) -> tuple[Any, int]:
        tts = self._load_model()

        try:
            import numpy as np
        except ImportError as exc:
            raise CLIError(
                "numpy is not installed in the current environment."
            ) from exc

        synth_kwargs: dict[str, Any] = {
            "text": text,
            "speaker_wav": prepared_reference["speaker_wav"],
            "language": prepared_reference["language"],
        }
        if self._split_sentences is not None:
            synth_kwargs["split_sentences"] = self._split_sentences

        waveform = np.asarray(tts.tts(**synth_kwargs), dtype=np.float32)
        if waveform.size == 0:
            raise CLIError("XTTS returned no audio for the requested chunk.")

        sample_rate = self._output_sample_rate(tts)
        return waveform, sample_rate

    def _load_model(self) -> Any:
        if self._tts is not None:
            return self._tts

        try:
            import torch
        except ImportError as exc:
            raise CLIError(
                "PyTorch is not installed in the current environment."
            ) from exc

        try:
            self._patch_xtts_audio_loader()
            from TTS.api import TTS
        except ImportError as exc:
            raise CLIError(
                "The xtts backend requires the Coqui XTTS runtime with codec support. Install it with `pip install -e .[xtts]`. "
                f"Underlying import error: {exc}"
            ) from exc

        device = self._resolved_device(torch)

        try:
            tts = TTS(self._resolved_model_name)
        except Exception as exc:
            raise CLIError(
                f"Unable to load XTTS model '{self.settings.model_name}'."
            ) from exc

        if hasattr(tts, "to"):
            tts = tts.to(device)

        self._tts = tts
        return self._tts

    def _patch_xtts_audio_loader(self) -> None:
        try:
            import torch
            from TTS.tts.models import xtts as xtts_module
        except ImportError:
            return

        if getattr(xtts_module, "_zero_shot_voiceclone_loader_patched", False):
            return

        def load_audio(audiopath: str | os.PathLike[Any], sampling_rate: int):
            audio_array, sample_rate = self._read_audio_array(Path(audiopath))
            audio_tensor = torch.from_numpy(audio_array)

            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            elif (
                audio_tensor.ndim == 2 and audio_tensor.shape[0] > audio_tensor.shape[1]
            ):
                audio_tensor = audio_tensor.transpose(0, 1)

            if audio_tensor.size(0) != 1:
                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

            if sample_rate != sampling_rate:
                audio_tensor = xtts_module.torchaudio.functional.resample(
                    audio_tensor, sample_rate, sampling_rate
                )

            if torch.any(audio_tensor > 10) or not torch.any(audio_tensor < 0):
                xtts_module.logger.error(
                    "Error with %s. Max=%.2f min=%.2f",
                    audiopath,
                    audio_tensor.max(),
                    audio_tensor.min(),
                )

            audio_tensor = audio_tensor.to(dtype=torch.float32)
            audio_tensor.clip_(-1, 1)
            return audio_tensor

        xtts_module.load_audio = load_audio
        xtts_module._zero_shot_voiceclone_loader_patched = True

    def _read_audio_array(self, path: Path) -> tuple[Any, int]:
        try:
            import numpy as np
        except ImportError as exc:
            raise CLIError(
                "numpy is not installed in the current environment."
            ) from exc

        try:
            import soundfile as sf
        except ImportError:
            sf = None

        if sf is not None:
            try:
                audio_array, sample_rate = sf.read(
                    str(path), dtype="float32", always_2d=False
                )
                return self._normalize_audio_array(audio_array, np), int(sample_rate)
            except RuntimeError:
                pass

        try:
            import librosa
        except ImportError:
            librosa = None

        if librosa is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                try:
                    audio_array, sample_rate = librosa.load(
                        str(path), sr=None, mono=False
                    )
                except Exception:
                    audio_array = None
                else:
                    return self._normalize_audio_array(audio_array, np), int(
                        sample_rate
                    )

        try:
            import audioread
        except ImportError as exc:
            raise CLIError(
                f"Unable to decode XTTS reference audio: {path}. Install librosa or provide a PCM WAV file."
            ) from exc

        pcm_chunks: list[Any] = []
        try:
            with audioread.audio_open(str(path)) as input_file:
                sample_rate = int(input_file.samplerate)
                channels = int(input_file.channels)
                for buffer in input_file:
                    pcm_chunks.append(np.frombuffer(buffer, dtype="<i2"))
        except OSError as exc:
            raise CLIError(f"Unable to decode XTTS reference audio: {path}") from exc

        if not pcm_chunks:
            raise CLIError(f"XTTS reference audio produced no samples: {path}")

        pcm = np.concatenate(pcm_chunks).astype(np.float32) / 32768.0
        if channels > 1:
            pcm = pcm.reshape(-1, channels).T

        return self._normalize_audio_array(pcm, np), sample_rate

    def _normalize_audio_array(self, audio_array: Any, np: Any) -> Any:
        array = np.asarray(audio_array, dtype=np.float32)
        if array.ndim == 0:
            raise CLIError("XTTS reference audio contains no samples.")
        if array.ndim > 2:
            raise CLIError("XTTS reference audio has an unsupported channel layout.")
        return array

    def _resolved_device(self, torch: Any) -> str:
        if self._device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _output_sample_rate(self, tts: Any) -> int:
        synthesizer = getattr(tts, "synthesizer", None)
        value = getattr(synthesizer, "output_sample_rate", None)
        if value:
            return int(value)
        return XTTS_DEFAULT_SAMPLE_RATE

    @property
    def _resolved_model_name(self) -> str:
        return XTTS_MODEL_ALIASES.get(
            self.settings.model_name, self.settings.model_name
        )

    @property
    def _normalized_language(self) -> str:
        key = self.settings.language.strip().lower()
        try:
            return XTTS_LANGUAGE_ALIASES[key]
        except KeyError as exc:
            supported = ", ".join(XTTS_SUPPORTED_LANGUAGES)
            raise CLIError(
                f"Unsupported XTTS language '{self.settings.language}'. Use an explicit code such as 'en' or 'zh-cn'. Supported XTTS language codes: {supported}"
            ) from exc

    @property
    def _device(self) -> str:
        return str(self.settings.backend_options.get("device", "auto"))

    @property
    def _split_sentences(self) -> bool | None:
        value = self.settings.backend_options.get("split_sentences")
        if value is None:
            return None
        return bool(value)
